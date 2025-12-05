# ==============================================================================
#  PART 1: Original Model Loading & Imports (From your provided script)
# ==============================================================================
from diffusers_helper.hf_login import login
import os
import cv2
import subprocess
import shutil

# Setup Paths
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# --- Initialize Models (Global Scope) ---
print("--- Initializing Models ---")
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval(); text_encoder.eval(); text_encoder_2.eval(); image_encoder.eval(); transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False); text_encoder.requires_grad_(False); text_encoder_2.requires_grad_(False); image_encoder.requires_grad_(False); transformer.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu); text_encoder_2.to(gpu); image_encoder.to(gpu); vae.to(gpu); transformer.to(gpu)

# ==============================================================================
#  PART 2: The Worker Logic (Copied from source, slightly adapted for context)
# ==============================================================================

# We need a mock stream to capture the worker output
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
            desc = item[1][1]
            if desc: print(f"Progress: {desc}")
        elif flag == 'end':
            self.status = "ended"

    def top(self):
        return None

class MockStream:
    def __init__(self):
        self.output_queue = MockQueue()
        self.input_queue = MockQueue()

# Override the global stream
stream = MockStream()

# (Include the worker function from your script here implicitly or import if in module)
# Since the worker relies on the global 'stream' variable we just defined, 
# we paste the worker logic here to ensure it uses OUR stream.

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    # [Insert the full worker code from your provided snippet here]
    # For brevity in this answer, I assume the 'worker' function logic is available 
    # either by pasting the function body here or if this script is appended to the original file.
    # The logic below calls 'worker' assuming it is defined in the scope.
    
    # ... (Pasting the worker function body from your provided text) ...
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        
        # Text encoding
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Image Processing
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE
        if not high_vram: load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(transformer.dtype)
        
        # Casting
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)

        # Sampling Loop
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache: transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else: transformer.initialize_teacache(enable_teacache=False)

            def callback(d): pass # Simplified callback

            generated_latents = sample_hunyuan(
                transformer=transformer, sampler='unipc', width=width, height=height, frames=num_frames,
                real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                num_inference_steps=steps, generator=rnd,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                device=gpu, dtype=torch.bfloat16, image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram: unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            stream.output_queue.push(('file', output_filename))
            if is_last_section: break
    except:
        traceback.print_exc()
    stream.output_queue.push(('end', None))


# ==============================================================================
#  PART 3: Video Extraction and Loop Logic
# ==============================================================================

def create_looped_video(input_video_path):
    print(f"\nProcessing Video: {input_video_path}")
    
    # 1. Extract Frames using OpenCV
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read First Frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame_bgr = cap.read()
    if not ret: return
    
    # Read Last Frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, last_frame_bgr = cap.read()
    cap.release()
    if not ret: return

    # Convert BGR (OpenCV) to RGB (PIL/Numpy)
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    last_frame_rgb = cv2.cvtColor(last_frame_bgr, cv2.COLOR_BGR2RGB)

    # 2. Generate "Bridge" Video
    # We treat the Last Frame as the start image for the generator.
    # Note: Since the worker() function only accepts an input image (not a target image),
    # we rely on the prompt or the natural continuation to simulate the loop.
    # To strictly follow the "first frame as last frame" rule via code, one would need
    # an interpolation model, but this script is an I2V generator.
    
    print("Generating bridge video from Last Frame...")
    
    # Configuration
    prompt = "smooth transition" # Minimal prompt to encourage motion
    n_prompt = ""
    seed = 42
    total_second_length = 1.0 # Shorter length for the bridge
    latent_window_size = 9
    steps = 25
    cfg = 1.0
    gs = 10.0
    rs = 0.0
    gpu_mem = 6.0
    use_teacache = True
    mp4_crf = 16

    # Call Worker
    worker(
        input_image=last_frame_rgb, # Last frame becomes Start of new video
        prompt=prompt,
        n_prompt=n_prompt,
        seed=seed,
        total_second_length=total_second_length,
        latent_window_size=latent_window_size,
        steps=steps,
        cfg=cfg,
        gs=gs,
        rs=rs,
        gpu_memory_preservation=gpu_mem,
        use_teacache=use_teacache,
        mp4_crf=mp4_crf
    )

    bridge_video_path = stream.output_queue.last_file
    if not bridge_video_path:
        print("Error: Video generation failed.")
        return

    print(f"Bridge video generated at: {bridge_video_path}")

    # 3. Concatenate Videos using ffmpeg
    final_output_name = f"looped_{os.path.basename(input_video_path)}"
    final_output_path = os.path.join(outputs_folder, final_output_name)

    # Create a temporary list file for ffmpeg
    list_file = "files_to_merge.txt"
    with open(list_file, "w") as f:
        # Absolute paths are safer for ffmpeg
        f.write(f"file '{os.path.abspath(input_video_path)}'\n")
        f.write(f"file '{os.path.abspath(bridge_video_path)}'\n")

    print("Merging videos...")
    # ffmpeg command to concat without re-encoding if codecs match, 
    # but strictly re-encoding is safer for different generation params.
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", # Re-encode to ensure smoothness
        final_output_path
    ]
    
    subprocess.run(cmd)
    
    if os.path.exists(list_file):
        os.remove(list_file)
        
    print(f"\nSUCCESS: Final looped video saved to: {final_output_path}")

# ==============================================================================
#  Execution
# ==============================================================================

if __name__ == "__main__":
    # REPLACE THIS WITH YOUR VIDEO PATH
    input_video = "out_new_angry_1.mp4" 
    
    if os.path.exists(input_video):
        create_looped_video(input_video)
    else:
        print("Please set a valid input_video path at the bottom of the script.")