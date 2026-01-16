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
def worker(input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
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

        # Image Processing (start frame)
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # Processing end image (if provided)
        has_end_image = end_image is not None
        if has_end_image:
            H_end, W_end, C_end = end_image.shape
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE
        if not high_vram: load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)
        
        if has_end_image:
            end_latent = vae_encode(end_image_pt, vae)

        # CLIP Vision
        if not high_vram: load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # Combine both image embeddings or use a weighted approach
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2
        
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        
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
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

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
#  PART 3: Utility Functions for Multi-Second Loop Generation
# ==============================================================================

def extract_frame_at_timestamp(video_path, timestamp_sec):
    """
    Extract a single frame at a specific timestamp from a video.
    
    Args:
        video_path: Path to input video
        timestamp_sec: Timestamp in seconds
        
    Returns:
        tuple: (frame_rgb, height, width)
    """
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


def generate_interpolation_loop(start_frame, end_frame, output_path, config):
    """
    Generate interpolation video from start_frame to end_frame.
    
    Args:
        start_frame: Starting frame (numpy array)
        end_frame: Ending frame (numpy array)
        output_path: Path to save interpolation video
        config: Dictionary with generation parameters
        
    Returns:
        str: Path to generated interpolation video
    """
    print(f"  Generating interpolation: {os.path.basename(output_path)}")
    
    # Call worker function
    worker(
        input_image=start_frame,
        end_image=end_frame,
        prompt=config.get('prompt', 'smooth transition'),
        n_prompt=config.get('n_prompt', ''),
        seed=config.get('seed', 42),
        total_second_length=config.get('total_second_length', 1.0),
        latent_window_size=config.get('latent_window_size', 9),
        steps=config.get('steps', 25),
        cfg=config.get('cfg', 1.0),
        gs=config.get('gs', 10.0),
        rs=config.get('rs', 0.0),
        gpu_memory_preservation=config.get('gpu_memory_preservation', 6.0),
        use_teacache=config.get('use_teacache', True),
        mp4_crf=config.get('mp4_crf', 16)
    )
    
    # Get generated file from stream
    generated_path = stream.output_queue.last_file
    if not generated_path:
        raise ValueError("Interpolation generation failed")
    
    # Move to desired output path
    if generated_path != output_path:
        shutil.move(generated_path, output_path)
    
    return output_path


def trim_video_to_timestamp(input_video, output_path, end_timestamp, fps):
    """
    Trim video from start to just before the specified timestamp.
    Excludes the last frame to avoid duplication when concatenating with interpolation.
    
    Args:
        input_video: Path to input video
        output_path: Path to save trimmed video
        end_timestamp: End time in seconds
        fps: Frames per second of the video
        
    Returns:
        str: Path to trimmed video
    """
    print(f"  Trimming video to {end_timestamp}s: {os.path.basename(output_path)}")
    
    # Calculate the exact time to trim to (exclude the last frame)
    # Subtract one frame duration to avoid including the frame at end_timestamp
    frame_duration = 1.0 / fps
    trim_time = end_timestamp - frame_duration
    
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-t", str(trim_time),  # Trim to just before end_timestamp
        "-c:v", "libx264", "-pix_fmt", "yuv420p",  # Re-encode for precise frame control
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    return output_path


def concatenate_videos(video1_path, video2_path, output_path):
    """
    Concatenate two videos.
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Path to save concatenated video
        
    Returns:
        str: Path to concatenated video
    """
    print(f"  Concatenating videos: {os.path.basename(output_path)}")
    
    # Create temporary list file
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


def process_video_multi_loops(input_video_path):
    """
    Main function to process video and create loops at each second mark.
    
    For a 5-second video, creates:
    - Interpolations: 1sec→0sec, 2sec→0sec, 3sec→0sec, 4sec→0sec
    - Loopable videos: [0-1sec]+loop, [0-2sec]+loop, [0-3sec]+loop, [0-4sec]+loop
    
    Args:
        input_video_path: Path to input video
    """
    print(f"\n{'='*80}")
    print(f"Processing Video: {input_video_path}")
    print(f"{'='*80}\n")
    
    # Get video info
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Video Info:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {frame_count}\n")
    
    # Calculate number of second marks
    num_seconds = int(duration)
    if num_seconds < 1:
        print("Error: Video must be at least 1 second long.")
        return
    
    # Create output folders
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    interpolations_folder = os.path.join(outputs_folder, 'interpolations', video_name)
    loopable_folder = os.path.join(outputs_folder, 'loopable_videos', video_name)
    trimmed_folder = os.path.join(outputs_folder, 'trimmed_videos', video_name)
    
    os.makedirs(interpolations_folder, exist_ok=True)
    os.makedirs(loopable_folder, exist_ok=True)
    os.makedirs(trimmed_folder, exist_ok=True)
    
    print(f"Output folders created:")
    print(f"  Interpolations: {interpolations_folder}")
    print(f"  Loopable videos: {loopable_folder}")
    print(f"  Trimmed videos: {trimmed_folder}\n")
    
    # Extract frame at 0 seconds (reference frame)
    print("Extracting reference frame at 0s...")
    frame_0sec, _, _ = extract_frame_at_timestamp(input_video_path, 0)
    
    # Configuration for interpolation generation
    config = {
        'prompt': 'smooth transition',
        'n_prompt': '',
        'seed': 42,
        'total_second_length': 1.0,
        'latent_window_size': 9,
        'steps': 25,
        'cfg': 1.0,
        'gs': 10.0,
        'rs': 0.0,
        'gpu_memory_preservation': 6.0,
        'use_teacache': True,
        'mp4_crf': 16
    }
    
    # Process each second mark
    for sec in range(1, num_seconds):
        print(f"\n{'-'*80}")
        print(f"Processing {sec}s mark ({sec}/{num_seconds-1})")
        print(f"{'-'*80}")
        
        # Extract frame at this second
        print(f"Extracting frame at {sec}s...")
        frame_at_sec, _, _ = extract_frame_at_timestamp(input_video_path, sec)
        
        # Generate interpolation from this second back to 0
        interpolation_path = os.path.join(
            interpolations_folder, 
            f"{video_name}_loop_{sec}sec_to_0sec.mp4"
        )
        generate_interpolation_loop(frame_at_sec, frame_0sec, interpolation_path, config)
        
        # Trim original video to this second
        trimmed_path = os.path.join(
            trimmed_folder,
            f"{video_name}_trimmed_{sec}sec.mp4"
        )
        trim_video_to_timestamp(input_video_path, trimmed_path, sec, fps)
        
        # Concatenate trimmed + interpolation
        loopable_path = os.path.join(
            loopable_folder,
            f"{video_name}_loopable_{sec}sec.mp4"
        )
        concatenate_videos(trimmed_path, interpolation_path, loopable_path)
        
        print(f"✓ Completed {sec}s mark")
    
    print(f"\n{'='*80}")
    print(f"SUCCESS! Generated {num_seconds-1} loopable videos")
    print(f"{'='*80}")
    print(f"\nOutput locations:")
    print(f"  Interpolations: {interpolations_folder}")
    print(f"  Loopable videos: {loopable_folder}")
    print(f"  Trimmed videos: {trimmed_folder}\n")


# ==============================================================================
#  PART 4: Legacy Function (kept for reference)
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
    prompt = "silent lips, mouth shut, static tripod shot" # Minimal prompt to encourage motion
    n_prompt = "talking"
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

    # Call Worker with first frame as end_image to create seamless loop
    worker(
        input_image=last_frame_rgb,  # Last frame becomes Start of new video
        end_image=first_frame_rgb,   # First frame becomes End target for seamless loop
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
    input_video = "test_vid.mp4" 
    
    if os.path.exists(input_video):
        # Use the new multi-second loop generation
        process_video_multi_loops(input_video)
        
        # To use the legacy single-loop function, uncomment below:
        # create_looped_video(input_video)
    else:
        print(f"Error: Video file not found: {input_video}")
        print("Please set a valid input_video path at the bottom of the script.")