"""
API Request/Response Models and Validation

This module defines Pydantic models for request validation and response formatting.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ProcessVideoRequest(BaseModel):
    """Request model for /process_video endpoint"""
    duration: Optional[int] = Field(
        None, 
        ge=1, 
        le=60, 
        description="Process only first N seconds of video (1-60). If not specified, processes entire video."
    )
    include_loops: bool = Field(
        False, 
        description="Include loopable videos in response"
    )
    include_trimmed: bool = Field(
        False, 
        description="Include trimmed videos in response"
    )
    include_interpolations: bool = Field(
        True, 
        description="Include interpolation videos in response"
    )


class VideoOutputs(BaseModel):
    """Video outputs structure"""
    interpolations: Optional[List[str]] = None
    loopable_videos: Optional[List[str]] = None
    trimmed_videos: Optional[List[str]] = None


class ProcessVideoResponse(BaseModel):
    """Response model for /process_video endpoint"""
    status: str
    video_name: str
    duration: float
    resolution: str
    fps: float
    loops_generated: int
    outputs: VideoOutputs


class ConfigResponse(BaseModel):
    """Response model for /config endpoint"""
    prompt: str
    n_prompt: str
    seed: int
    total_second_length: float
    latent_window_size: int
    steps: int
    cfg: float
    gs: float
    rs: float
    gpu_memory_preservation: float
    use_teacache: bool
    mp4_crf: int


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration"""
    prompt: Optional[str] = Field(None, description="Prompt for generation")
    total_second_length: Optional[float] = Field(None, ge=0.1, le=10.0, description="Video length in seconds")
    latent_window_size: Optional[int] = Field(None, ge=1, le=33, description="Latent window size")
    steps: Optional[int] = Field(None, ge=1, le=100, description="Number of inference steps")
    cfg: Optional[float] = Field(None, ge=1.0, le=32.0, description="CFG scale")
    gs: Optional[float] = Field(None, ge=1.0, le=32.0, description="Distilled CFG scale")
    rs: Optional[float] = Field(None, ge=0.0, le=1.0, description="CFG re-scale")
    gpu_memory_preservation: Optional[float] = Field(None, ge=0.0, le=128.0, description="GPU memory to preserve (GB)")
    use_teacache: Optional[bool] = Field(None, description="Enable TeaCache")
    mp4_crf: Optional[int] = Field(None, ge=0, le=51, description="MP4 compression quality")


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    models_loaded: int
    high_vram: bool
    gpu_memory_gb: float
    cached_prompts: List[str]


class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
