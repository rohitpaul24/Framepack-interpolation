from diffusers_helper.hf_login import login

import os
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
import cv2
import shutil
import traceback
from pathlib import Path
from typing import Optional

# Import all necessary modules
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import crop_or_pad_yield_mask, generate_timestamp, save_bcthw_as_mp4, resize_and_center_crop, resize_and_pad, remove_padding, soft_append_bcthw
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, unload_complete_models, load_model_as_complete, fake_diffusers_current_device, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
import numpy as np
import einops
from PIL import Image

# Import API modules
from api import (
    ProcessVideoRequest, ProcessVideoResponse, ConfigResponse, ConfigUpdateRequest,
    HealthResponse, ErrorResponse, VideoOutputs,
    APIError, ValidationError, ProcessingError, ModelNotLoadedError,
    InvalidDurationError, VideoTooShortError,
    api_error_handler, general_exception_handler,
    RequestLoggingMiddleware, log_video_processing, log_config_update, log_error, logger,
    Job, JobStatus, start_worker, queue_job, get_queue_size
)

# Additional imports for async job system
import uuid

# Initialize FastAPI app
app = FastAPI(
    title="Video Loop Generation API",
    version="1.0.0",
    description="API for generating seamless video loops using AI interpolation"
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)

# Add exception handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Global variables for models and config
models = {}
high_vram = False
outputs_folder = './outputs/'
uploads_folder = './uploads/'
os.makedirs(outputs_folder, exist_ok=True)
os.makedirs(uploads_folder, exist_ok=True)

# Global configuration (can be updated via API)
server_config = {
    'prompt': 'silent lips',
    'n_prompt': '',
    'seed': 42,
    'total_second_length': 0.5,
    'latent_window_size': 7,
    'steps': 10,
    'cfg': 1.0,
    'gs': 10.0,
    'rs': 0.0,
    'gpu_memory_preservation': 6.0,
    'use_teacache': True,
    'mp4_crf': 16
}

# Mock stream for worker function
class MockQueue:
    def __init__(self):
        self.last_file = None
        self.status = "running"
    
    def push(self, item):
        flag = item[0]
        if flag == 'file':
            self.last_file = item[1]
            print(f"Generated File: {self.last_file}")
        elif flag == 'progress':
            desc = item[1][1] if len(item[1]) > 1 else ""
            if desc: print(f"Progress: {desc}")
        elif flag == 'end':
            self.status = "ended"

    def top(self):
        return None

class MockStream:
    def __init__(self):
        self.output_queue = MockQueue()
        self.input_queue = MockQueue()

stream = MockStream()


@app.on_event("startup")
async def load_models():
    """Load all models once at startup"""
    global models, high_vram
    
    print("\n" + "="*80)
    print("INITIALIZING MODELS - This will take a few minutes...")
    print("="*80 + "\n")
    
    try:
        free_mem_gb = get_cuda_free_memory_gb(gpu)
        high_vram = free_mem_gb > 60
        
        print(f'Free VRAM: {free_mem_gb} GB')
        print(f'High-VRAM Mode: {high_vram}\n')
        
        # Load text encoders
        print("Loading text encoders...")
        text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        
        # Load tokenizers
        print("Loading tokenizers...")
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        
        # Load VAE
        print("Loading VAE...")
        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
        
        # Load image encoder
        print("Loading image encoder...")
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
        
        # Load transformer
        print("Loading transformer...")
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
        
        # Set to eval mode
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()
        
        # Configure VAE
        if not models.get('high_vram', False):
            vae.enable_slicing()
            vae.enable_tiling()
        
        # Configure transformer
        transformer.high_quality_fp32_output_for_inference = True
        
        # Set dtypes
        transformer.to(dtype=torch.bfloat16)
        vae.to(dtype=torch.float16)
        image_encoder.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        text_encoder_2.to(dtype=torch.float16)
        
        # Disable gradients
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        image_encoder.requires_grad_(False)
        transformer.requires_grad_(False)
        
        
        # Model loading - match demo_gradio.py logic
        if not models.get('high_vram', False):
            DynamicSwapInstaller.install_model(transformer, device=gpu)
            DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        else:
            text_encoder.to(gpu)
            text_encoder_2.to(gpu)
            image_encoder.to(gpu)
            vae.to(gpu)
            transformer.to(gpu)
        
        # Store models globally
        models['transformer'] = transformer
        models['vae'] = vae
        models['text_encoder'] = text_encoder
        models['text_encoder_2'] = text_encoder_2
        models['image_encoder'] = image_encoder
        models['tokenizer'] = tokenizer
        models['tokenizer_2'] = tokenizer_2
        models['feature_extractor'] = feature_extractor
        models['high_vram'] = high_vram
        
        # Cache prompt embeddings at startup (optimization)
        print("\nPre-computing prompt embeddings...")
        prompt = 'silent lips'
        n_prompt = ''
        
        if not high_vram:
            from diffusers_helper.memory import fake_diffusers_current_device, load_model_as_complete
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        
        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # Store cached embeddings
        models['cached_prompt_embeds'] = {
            'llama_vec': llama_vec,
            'llama_attention_mask': llama_attention_mask,
            'clip_l_pooler': clip_l_pooler,
            'llama_vec_n': llama_vec_n,
            'llama_attention_mask_n': llama_attention_mask_n,
            'clip_l_pooler_n': clip_l_pooler_n
        }
        
        print("✓ Prompt embeddings cached")

        # Unload text encoders to free VRAM (they won't be needed again)
        if not high_vram:
            from diffusers_helper.memory import unload_complete_models
            unload_complete_models(text_encoder, text_encoder_2)
            print("✓ Text encoders unloaded")

        
        print("\n" + "="*80)
        print("✓ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR LOADING MODELS: {e}")
        traceback.print_exc()
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if server and models are ready"""
    if not models:
        raise ModelNotLoadedError()
    
    return HealthResponse(
        status="ready",
        models_loaded=len(models),
        high_vram=models.get('high_vram', False),
        gpu_memory_gb=get_cuda_free_memory_gb(gpu),
        cached_prompts=[server_config['prompt']]
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current server configuration"""
    return ConfigResponse(**server_config)


@app.post("/config")
async def update_config(config_update: ConfigUpdateRequest):
    """Update server configuration"""
    global server_config
    
    updated_fields = []
    prompt_changed = False
    
    # Update configuration
    for field, value in config_update.dict(exclude_unset=True).items():
        if value is not None:
            server_config[field] = value
            updated_fields.append(field)
            if field == 'prompt':
                prompt_changed = True
    
    # Re-cache prompt embeddings if prompt changed
    if prompt_changed:
        logger.info("Prompt changed, re-caching embeddings...")
        try:
            # Re-compute embeddings
            if not models.get('high_vram', False):
                fake_diffusers_current_device(models['text_encoder'], gpu)
                load_model_as_complete(models['text_encoder_2'], target_device=gpu)
            
            llama_vec, clip_l_pooler = encode_prompt_conds(
                server_config['prompt'], 
                models['text_encoder'], 
                models['text_encoder_2'], 
                models['tokenizer'], 
                models['tokenizer_2']
            )
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            
            # Update cached embeddings
            models['cached_prompt_embeds'] = {
                'llama_vec': llama_vec,
                'llama_attention_mask': llama_attention_mask,
                'clip_l_pooler': clip_l_pooler,
                'llama_vec_n': llama_vec_n,
                'llama_attention_mask_n': llama_attention_mask_n,
                'clip_l_pooler_n': clip_l_pooler_n
            }
            
            # Unload text encoders
            if not models.get('high_vram', False):
                unload_complete_models(models['text_encoder'], models['text_encoder_2'])
            
            logger.info("✓ Prompt embeddings re-cached")
        except Exception as e:
            log_error("CONFIG_UPDATE_ERROR", f"Failed to re-cache embeddings: {e}")
            raise ProcessingError(f"Failed to update prompt: {e}")
    
    log_config_update({k: server_config[k] for k in updated_fields})
    
    return {
        "status": "success",
        "message": "Configuration updated",
        "updated_fields": updated_fields,
        "requires_restart": False
    }


@app.post("/process_video", response_model=ProcessVideoResponse)
async def process_video(
    file: UploadFile = File(...),
    duration: Optional[float] = Form(None),
    include_loops: bool = Form(False),
    include_trimmed: bool = Form(False),
    include_interpolations: bool = Form(True)
):
    """
    Process uploaded video and generate loops at each second mark.
    
    Args:
        file: Video file to process
        duration: Length of each interpolation video in seconds (0.1-10.0). Default: server config (0.5s)
        include_loops: Include loopable videos in response
        include_trimmed: Include trimmed videos in response
        include_interpolations: Include interpolation videos in response
    
    Returns:
        JSON with paths to generated files
    """
    if not models:
        raise ModelNotLoadedError()
    
    # Validate duration
    if duration is not None and (duration < 0.5 or duration > 10.0):
        raise InvalidDurationError(duration)
    
    # Save uploaded file
    video_name = Path(file.filename).stem
    upload_path = os.path.join(outputs_folder, f"upload_{video_name}.mp4")
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing video: {file.filename}")
        
        # Process video using loop generation
        result = process_video_multi_loops_api(
            upload_path,
            duration=duration,
            include_loops=include_loops,
            include_trimmed=include_trimmed,
            include_interpolations=include_interpolations
        )
        
        log_video_processing(result['video_name'], result['duration'], result['loops_generated'])
        
        return ProcessVideoResponse(**result)
        
    except APIError:
        raise
    except Exception as e:
        log_error("PROCESSING_ERROR", str(e))
        raise ProcessingError(f"Failed to process video: {e}")
    finally:
        # Clean up uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)


@app.get("/outputs/{video_name}/{folder}/{filename}")
async def download_file(video_name: str, folder: str, filename: str):
    """Download generated files"""
    file_path = os.path.join(outputs_folder, folder, video_name, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


# ==============================================================================
#  Video Processing Functions (copied from get_loop.py)
# ==============================================================================

def extract_frame_at_timestamp(video_path, timestamp_sec):
    """Extract a single frame at a specific timestamp from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame at {timestamp_sec}s")
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    
    return frame_rgb, height, width


def worker(input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, pad_info=None):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not models.get('high_vram', False):
            unload_complete_models(
                models['image_encoder'], models['vae'], models['transformer']
            )

        # Use cached prompt embeddings (optimization)
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Loading cached embeddings ...'))))
        
        cached = models['cached_prompt_embeds']
        llama_vec = cached['llama_vec']
        llama_attention_mask = cached['llama_attention_mask']
        clip_l_pooler = cached['clip_l_pooler']
        llama_vec_n = cached['llama_vec_n']
        llama_attention_mask_n = cached['llama_attention_mask_n']
        clip_l_pooler_n = cached['clip_l_pooler_n']

        # Processing input image (start frame) - Use padding instead of cropping
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        
        # Use padding if pad_info not provided
        if pad_info is None:
            input_image_np, pad_info = resize_and_pad(input_image, target_width=width, target_height=height)
        else:
            input_image_np, _ = resize_and_pad(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_start.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # Processing end image (if provided)
        has_end_image = end_image is not None
        if has_end_image:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...'))))
            
            H_end, W_end, C_end = end_image.shape
            end_image_np, _ = resize_and_pad(end_image, target_width=width, target_height=height)  # Use same bucket size
            
            Image.fromarray(end_image_np).save(os.path.join(outputs_folder, f'{job_id}_end.png'))
            
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not models.get('high_vram', False):
            load_model_as_complete(models['vae'], target_device=gpu)

        start_latent = vae_encode(input_image_pt, models['vae'])
        
        if has_end_image:
            end_latent = vae_encode(end_image_pt, models['vae'])

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not models.get('high_vram', False):
            load_model_as_complete(models['image_encoder'], target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, models['feature_extractor'], models['image_encoder'])
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, models['feature_extractor'], models['image_encoder'])
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # Combine both image embeddings or use a weighted approach
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2

        # Dtype
        llama_vec = llama_vec.to(models['transformer'].dtype)
        llama_vec_n = llama_vec_n.to(models['transformer'].dtype)
        clip_l_pooler = clip_l_pooler.to(models['transformer'].dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(models['transformer'].dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(models['transformer'].dtype)

        # Sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        # 将迭代器转换为列表
        latent_paddings = list(reversed(range(total_latent_sections)))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # Use end image latent for the first section if provided
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not models.get('high_vram', False):
                unload_complete_models()
                move_model_to_device_with_memory_preservation(models['transformer'], target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                models['transformer'].initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                models['transformer'].initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=models['transformer'],
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not models.get('high_vram', False):
                offload_model_from_device_for_memory_preservation(models['transformer'], target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(models['vae'], target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, models['vae']).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], models['vae']).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not models.get('high_vram', False):
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            
            # Remove padding from generated video if padding was applied
            if pad_info is not None and (pad_info['pad_top'] > 0 or pad_info['pad_left'] > 0):
                # Convert tensor to numpy for padding removal
                history_pixels_np = history_pixels.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
                history_pixels_np = ((history_pixels_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                
                # Remove padding from each frame
                unpadded_frames = []
                for frame in history_pixels_np:
                    unpadded_frame = remove_padding(frame, pad_info)
                    unpadded_frames.append(unpadded_frame)
                
                # Convert back to tensor format for saving
                unpadded_np = np.stack(unpadded_frames, axis=0)  # (T, H, W, C)
                unpadded_tensor = torch.from_numpy(unpadded_np).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
                unpadded_tensor = (unpadded_tensor.float() / 127.5 - 1.0)
                
                save_bcthw_as_mp4(unpadded_tensor, output_filename, fps=30, crf=mp4_crf)
            else:
                # No padding, save as normal
                save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not models.get('high_vram', False):
            unload_complete_models(
                models['image_encoder'], models['vae'], models['transformer']
            )

    stream.output_queue.push(('end', None))
    return


def generate_interpolation_loop(start_frame, end_frame, output_path, config):
    """Generate interpolation video from start_frame to end_frame."""
    print(f"  Generating interpolation: {os.path.basename(output_path)}")
    
    # Reset stream before generation
    stream.output_queue.last_file = None
    stream.output_queue.status = "running"
    
    worker(
        input_image=start_frame,
        end_image=end_frame,
        prompt=config.get('prompt', 'silent lips'),
        n_prompt=config.get('n_prompt', ''),
        seed=config.get('seed', 42),
        total_second_length=config.get('total_second_length', 0.5),
        latent_window_size=config.get('latent_window_size', 6),
        steps=config.get('steps', 10),
        cfg=config.get('cfg', 1.0),
        gs=config.get('gs', 8.0),
        rs=config.get('rs', 0.0),
        gpu_memory_preservation=config.get('gpu_memory_preservation', 6.0),
        use_teacache=config.get('use_teacache', True),
        mp4_crf=config.get('mp4_crf', 16)
    )
    
    generated_path = stream.output_queue.last_file
    if not generated_path:
        raise ValueError("Interpolation generation failed")
    
    if generated_path != output_path:
        shutil.move(generated_path, output_path)
    
    return output_path


def trim_video_to_timestamp(input_video, output_path, end_timestamp, fps):
    """Trim video from start to just before the specified timestamp."""
    import subprocess
    
    print(f"  Trimming video to {end_timestamp}s: {os.path.basename(output_path)}")
    
    frame_duration = 1.0 / fps
    trim_time = end_timestamp - frame_duration
    
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-t", str(trim_time),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    return output_path


def concatenate_videos(video1_path, video2_path, output_path):
    """Concatenate two videos."""
    import subprocess
    
    print(f"  Concatenating videos: {os.path.basename(output_path)}")
    
    list_file = "concat_list.txt"
    with open(list_file, "w") as f:
        f.write(f"file '{os.path.abspath(video1_path)}'\n")
        f.write(f"file '{os.path.abspath(video2_path)}'\n")
    
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    
    if os.path.exists(list_file):
        os.remove(list_file)
    
    return output_path


def process_video_multi_loops_api(
    input_video_path,
    duration=None,
    include_loops=False,
    include_trimmed=False,
    include_interpolations=True
):
    """
    Process video and create loops at each second mark.
    Returns dict with results for API response.
    
    Args:
        input_video_path: Path to input video
        duration: Process only first N seconds (None = entire video)
        include_loops: Include loopable videos in response
        include_trimmed: Include trimmed videos in response
        include_interpolations: Include interpolations in response
    """
    # Get video info
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ProcessingError("Could not open video")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    logger.info(f"Video Info: {video_duration:.2f}s, {width}x{height}, {fps:.2f} FPS")
    
    # Validate video length
    if video_duration < 1.0:
        raise VideoTooShortError(video_duration)
    
    # Create output folders
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    interpolations_folder = os.path.join(outputs_folder, 'interpolations', video_name)
    loopable_folder = os.path.join(outputs_folder, 'loopable_videos', video_name)
    trimmed_folder = os.path.join(outputs_folder, 'trimmed_videos', video_name)
    
    os.makedirs(interpolations_folder, exist_ok=True)
    os.makedirs(loopable_folder, exist_ok=True)
    os.makedirs(trimmed_folder, exist_ok=True)
    
    # Extract frame at 0 seconds
    logger.info("Extracting reference frame at 0s...")
    frame_0sec, _, _ = extract_frame_at_timestamp(input_video_path, 0)
    
    # Process all intermediate seconds (1s, 2s, 3s, ... up to floor(duration))
    num_seconds = int(video_duration)
    
    # Use global server configuration
    config = server_config.copy()
    
    # Override total_second_length if duration parameter is provided
    if duration is not None:
        config['total_second_length'] = duration
    
    interpolations = []
    loopable_videos = []
    trimmed_videos = []
    
    # Process each second mark
    for sec in range(1, num_seconds):
        logger.info(f"Processing {sec}s mark ({sec}/{num_seconds-1})")
        
        frame_at_sec, _, _ = extract_frame_at_timestamp(input_video_path, sec)
        
        # Always generate interpolation (needed for loops)
        interpolation_path = os.path.join(interpolations_folder, f"{video_name}_loop_{sec}sec_to_0sec.mp4")
        generate_interpolation_loop(frame_at_sec, frame_0sec, interpolation_path, config)
        
        if include_interpolations:
            interpolations.append(f"/outputs/{video_name}/interpolations/{os.path.basename(interpolation_path)}")
        
        # Generate trimmed and loopable videos if requested
        if include_trimmed or include_loops:
            trimmed_path = os.path.join(trimmed_folder, f"{video_name}_trimmed_{sec}sec.mp4")
            trim_video_to_timestamp(input_video_path, trimmed_path, sec, fps)
            
            if include_trimmed:
                trimmed_videos.append(f"/outputs/{video_name}/trimmed_videos/{os.path.basename(trimmed_path)}")
        
        if include_loops:
            loopable_path = os.path.join(loopable_folder, f"{video_name}_loopable_{sec}sec.mp4")
            concatenate_videos(trimmed_path, interpolation_path, loopable_path)
            loopable_videos.append(f"/outputs/{video_name}/loopable_videos/{os.path.basename(loopable_path)}")
        
        logger.info(f"✓ Completed {sec}s mark")
    
    # Build response with only requested outputs
    outputs = VideoOutputs()
    if include_interpolations:
        outputs.interpolations = interpolations
    if include_loops:
        outputs.loopable_videos = loopable_videos
    if include_trimmed:
        outputs.trimmed_videos = trimmed_videos
    
    return {
        "status": "success",
        "video_name": video_name,
        "duration": video_duration,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "loops_generated": num_seconds - 1,
        "outputs": outputs.dict(exclude_none=True)
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Starting Video Loop Generation Server")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
