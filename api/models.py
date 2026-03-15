from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TaskStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELED   = "canceled"


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Description of the video to generate")
    seed: int = Field(default=42, description="Random seed (-1 for random)")
    resolution_preset: str = Field(
        default="1280 × 768   (720p Landscape)",
        description="Resolution preset string (e.g. '1280 × 768   (720p Landscape)')",
    )
    custom_height: int = Field(default=1024, ge=64, le=2160, description="Custom height (must be divisible by 64)")
    custom_width: int  = Field(default=1536, ge=64, le=3840, description="Custom width (must be divisible by 64)")
    use_custom_resolution: bool = Field(default=False, description="Flag to use custom height and width")
    num_frames: int = Field(
        default=121, ge=9, le=721, description="Number of frames (must be 8k+1, e.g. 9, 17, ..., 121)"
    )
    frame_rate: float = Field(default=24.0, ge=8.0, le=30.0, description="Frame rate in fps")
    image_path: Optional[str]   = Field(None, description="Path to conditioning image on server disk")
    image_base64: Optional[str] = Field(None, description="Base64 encoded conditioning image (e.g. 'data:image/png;base64,...')")
    image_strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Image conditioning strength")
    image_crf: int        = Field(default=33,  ge=0,   le=51,   description="Image CRF compression (0=lossless, 33=default)")
    enhance_prompt: bool  = Field(default=False, description="Enhance prompt with AI text encoder")


class VideoGenerationResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier for tracking")
    message: str = Field(..., description="Status message")


class TaskStatusResponse(BaseModel):
    task_id:      str        = Field(..., description="Unique task identifier")
    status:       TaskStatus = Field(..., description="Current status of the task")
    progress:     float      = Field(default=0.0, description="Progress (0.0 to 1.0)")
    message:      str        = Field(default="",  description="Status message or error details")
    video_url:    Optional[str] = Field(None, description="URL path to access the generated video")
    is_cancelled: bool       = Field(default=False, description="Whether the task was canceled")


class SystemLoadRequest(BaseModel):
    checkpoint_path: Optional[str] = Field(None, description="Path to the distilled checkpoint. Leave empty to auto-download.")
    upsampler_path:  Optional[str] = Field(None, description="Path to the spatial upsampler. Leave empty to auto-download.")
    gemma_path:      Optional[str] = Field(None, description="Path to Gemma root directory. Leave empty to auto-download.")

    # NOTE: quantization is intentionally FIXED to fp8-cast server-side.
    # This field is accepted for API compatibility but its value is IGNORED.
    # The server always enforces fp8-cast for Blackwell 96GB VRAM optimization.
    # Passing any other value (e.g. 'none', 'fp8-scaled-mm') has NO effect.
    quantization: str = Field(
        default="fp8-cast",
        description=(
            "IGNORED by server. Quantization is always fp8-cast (built-in). "
            "Field retained for API schema compatibility only."
        ),
    )

    lora_path:     Optional[str] = Field(None, description="Optional path to LoRA weights (.safetensors)")
    lora_strength: float         = Field(default=1.0, description="Strength of LoRA applied")


class SystemStatusResponse(BaseModel):
    pipeline_loaded: bool = Field(..., description="True if pipeline is currently active in memory")
    message:         str  = Field(..., description="Current state of the system pipeline")
    active_tasks:    int  = Field(..., description="Number of tasks currently in queue or processing")
    quantization:    str  = Field(default="fp8-cast", description="Active quantization policy (always fp8-cast)")
