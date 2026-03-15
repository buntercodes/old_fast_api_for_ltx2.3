import base64
import logging
import os
import re
import time
import uuid
import asyncio
from pathlib import Path
from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
#  LTX-2.3 imports (workspace packages)
# ---------------------------------------------------------------------------
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy

from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.media_io import encode_video

from ltx_core.types import LatentState, Audio
from ltx_core.components.protocols import DiffusionStepProtocol
from collections.abc import Iterator, Callable

from api.models import SystemLoadRequest, VideoGenerationRequest
from api.task_manager import TaskState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CPU_TASKS = int(os.environ.get("MAX_CPU_TASKS", 2))
CPU_LOCK = asyncio.Semaphore(MAX_CPU_TASKS)

# ---------------------------------------------------------------------------
#  FP8-CAST: Built-in enforced quantization policy for Blackwell 96GB
#
#  fp8-cast is ALWAYS applied on this server. It is NOT optional and NOT
#  configurable at runtime. This decision is intentional:
#    - Reduces transformer VRAM by ~40% vs BF16
#    - Enables MAX_CONCURRENT_GENS=6 on 96GB without OOM
#    - fp8-cast is safe on all FP8-capable GPUs (Ada, Hopper, Blackwell)
#    - fp8-scaled-mm (Hopper TensorRT-LLM only) is intentionally excluded
#
#  Do NOT change this unless you are moving to a non-FP8 GPU.
# ---------------------------------------------------------------------------
ENFORCED_QUANTIZATION = QuantizationPolicy.fp8_cast()
ENFORCED_QUANTIZATION_NAME = "fp8-cast"

# ---------------------------------------------------------------------------
#  Preset resolutions
# ---------------------------------------------------------------------------
PRESET_RESOLUTIONS = {
    # ── Landscape ──
    "1920 × 1088  (1080p Landscape)": (1088, 1920),
    "1280 × 768   (720p Landscape)":  (768,  1280),
    "1536 × 1024  (3:2 Landscape)":   (1024, 1536),
    "1024 × 768   (4:3 Landscape)":   (768,  1024),
    "832 × 576    (Fast Landscape)":   (576,  832),
    # ── Portrait ──
    "1088 × 1920  (1080p Portrait)":  (1920, 1088),
    "768 × 1280   (720p Portrait)":   (1280, 768),
    "1024 × 1536  (2:3 Portrait)":    (1536, 1024),
    "768 × 1024   (3:4 Portrait)":    (1024, 768),
    "576 × 832    (Fast Portrait)":    (832,  576),
    # ── Square ──
    "1024 × 1024  (Square)":          (1024, 1024),
    "768 × 768    (Square Small)":    (768,  768),
}


# ---------------------------------------------------------------------------
#  FP8 verification helper
# ---------------------------------------------------------------------------
def verify_and_log_fp8_status() -> bool:
    """
    Verifies that the GPU supports FP8 and logs a detailed status report.
    Called once during pipeline load so the server startup logs are clear.
    Returns True if FP8 is confirmed supported, False otherwise.
    """
    logger.info("=" * 70)
    logger.info("  QUANTIZATION STATUS CHECK")
    logger.info("=" * 70)

    if not torch.cuda.is_available():
        logger.error("  [QUANTIZATION] ❌ CUDA is NOT available — cannot use FP8.")
        logger.info("=" * 70)
        return False

    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    compute_major, compute_minor = capability

    logger.info(f"  GPU Detected     : {gpu_name}")
    logger.info(f"  Compute Capability: {compute_major}.{compute_minor}")
    logger.info(f"  Total VRAM       : {total_vram_gb:.1f} GB")

    # FP8 requires compute capability 8.9+ (Ada Lovelace) or higher
    # Blackwell is 10.x, Hopper is 9.0, Ada is 8.9
    fp8_supported = (compute_major > 8) or (compute_major == 8 and compute_minor >= 9)

    if fp8_supported:
        logger.info(f"  FP8 Support      : ✅ YES (sm_{compute_major}{compute_minor} >= sm_89)")
        logger.info(f"  Quantization     : ✅ fp8-cast ACTIVE — transformer weights cast to FP8")
        logger.info(f"  VRAM Benefit     : ~40%% reduction vs BF16 on transformer")
        logger.info(f"  Concurrency      : Optimized for MAX_CONCURRENT_GENS on {total_vram_gb:.0f}GB VRAM")
    else:
        logger.warning(f"  FP8 Support      : ⚠️  NO (sm_{compute_major}{compute_minor} < sm_89)")
        logger.warning(f"  Quantization     : ⚠️  fp8-cast requested but GPU may not support it fully")
        logger.warning(f"  Recommendation   : Upgrade to Ada Lovelace (RTX 40xx) or Blackwell GPU")

    logger.info("=" * 70)
    return fp8_supported


# ---------------------------------------------------------------------------
#  Blackwell-optimized pipeline (models resident in VRAM)
# ---------------------------------------------------------------------------
class BlackwellDistilledPipeline(DistilledPipeline):
    """
    Subclass that keeps all models resident in 96GB VRAM.
    fp8-cast quantization is always applied — enforced at construction time.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transformer = None
        self._video_encoder = None
        self._upsampler = None
        self._vae_decoder = None
        self._audio_decoder = None
        self._vocoder = None

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        from ltx_pipelines.utils import (
            assert_resolution,
            combined_image_conditionings,
            denoise_audio_video,
            encode_prompts,
            euler_denoising_loop,
            simple_denoising_func,
        )
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_core.types import VideoPixelShape
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio

        assert_resolution(height=height, width=width, is_two_stage=True)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        with torch.no_grad():
            cleanup_memory()

            (ctx_p,) = encode_prompts(
                [prompt],
                self.model_ledger,
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            )
            video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

            if self._video_encoder is None:
                self._video_encoder = self.model_ledger.video_encoder()
            if self._transformer is None:
                self._transformer = self.model_ledger.transformer()

            video_encoder = self._video_encoder
            transformer = self._transformer

            stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

            base_denoise_fn = simple_denoising_func(
                video_context=video_context,
                audio_context=audio_context,
                transformer=transformer,
            )

            def custom_denoise_fn(*args, **kwargs):
                if cancel_callback and cancel_callback():
                    raise InterruptedError("Task was canceled by the user.")
                return base_denoise_fn(*args, **kwargs)

            def denoising_loop(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper: DiffusionStepProtocol,
            ) -> tuple[LatentState, LatentState]:
                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper,
                    denoise_fn=custom_denoise_fn,
                )

            stage_1_output_shape = VideoPixelShape(
                batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=frame_rate,
            )
            stage_1_conditionings = combined_image_conditionings(
                images=images, height=stage_1_output_shape.height, width=stage_1_output_shape.width,
                video_encoder=video_encoder, dtype=dtype, device=self.device,
            )

            video_state, audio_state = denoise_audio_video(
                output_shape=stage_1_output_shape, conditionings=stage_1_conditionings,
                noiser=noiser, sigmas=stage_1_sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
                components=self.pipeline_components, dtype=dtype, device=self.device,
            )

            if self._upsampler is None:
                self._upsampler = self.model_ledger.spatial_upsampler()

            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=self._upsampler
            )

            stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
            stage_2_output_shape = VideoPixelShape(
                batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
            )
            stage_2_conditionings = combined_image_conditionings(
                images=images, height=stage_2_output_shape.height, width=stage_2_output_shape.width,
                video_encoder=video_encoder, dtype=dtype, device=self.device,
            )
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_2_output_shape, conditionings=stage_2_conditionings,
                noiser=noiser, sigmas=stage_2_sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
                components=self.pipeline_components, dtype=dtype, device=self.device,
                noise_scale=stage_2_sigmas[0], initial_video_latent=upscaled_video_latent,
                initial_audio_latent=audio_state.latent,
            )

            if self._vae_decoder is None:
                self._vae_decoder = self.model_ledger.video_decoder()
            if self._audio_decoder is None:
                self._audio_decoder = self.model_ledger.audio_decoder()
            if self._vocoder is None:
                self._vocoder = self.model_ledger.vocoder()

            torch.cuda.synchronize()
            cleanup_memory()

            decoded_video = vae_decode_video(
                video_state.latent, self._vae_decoder, tiling_config, generator
            )
            decoded_audio = vae_decode_audio(
                audio_state.latent, self._audio_decoder, self._vocoder
            )

            cleanup_memory()

            return decoded_video, decoded_audio


# ---------------------------------------------------------------------------
#  Global pipeline state
# ---------------------------------------------------------------------------
_pipeline: Optional[BlackwellDistilledPipeline] = None
_pipeline_config: dict = {}


def get_pipeline_status() -> Tuple[bool, str]:
    if _pipeline is not None:
        return True, f"Pipeline is loaded and ready. Quantization: {ENFORCED_QUANTIZATION_NAME} (built-in)."
    return False, "Pipeline is not loaded. Please configure model paths via system settings first."


def load_pipeline(config_req: SystemLoadRequest) -> str:
    """
    Build (or reuse) the global BlackwellDistilledPipeline instance.

    IMPORTANT: The quantization field in SystemLoadRequest is IGNORED.
    fp8-cast is always enforced server-side for Blackwell 96GB VRAM optimization.
    Any value passed in config_req.quantization has no effect.
    """
    global _pipeline, _pipeline_config

    # Normalize config for cache comparison — quantization is always fp8-cast
    config = config_req.model_dump()
    config["quantization"] = ENFORCED_QUANTIZATION_NAME  # normalize for cache key

    if _pipeline is not None and _pipeline_config == config:
        logger.info("Reusing existing pipeline instance (fp8-cast active).")
        return f"Reusing existing pipeline instance. Quantization: {ENFORCED_QUANTIZATION_NAME} (built-in, always active)."

    logger.info("Building new BlackwellDistilledPipeline with enforced fp8-cast …")

    # Auto-download models if paths not provided
    if not config_req.checkpoint_path or not config_req.upsampler_path or not config_req.gemma_path:
        logger.info("One or more model paths not provided. Attempting to auto-download models...")
        from api.downloader import download_required_models
        downloaded = download_required_models()
        if not config_req.checkpoint_path:
            config_req.checkpoint_path = downloaded["checkpoint_path"]
        if not config_req.upsampler_path:
            config_req.upsampler_path = downloaded["upsampler_path"]
        if not config_req.gemma_path:
            config_req.gemma_path = downloaded["gemma_path"]

    if not Path(config_req.checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {config_req.checkpoint_path}")
    if not Path(config_req.upsampler_path).is_file():
        raise FileNotFoundError(f"Upsampler not found at: {config_req.upsampler_path}")
    if not Path(config_req.gemma_path).is_dir():
        raise FileNotFoundError(f"Gemma root not found at: {config_req.gemma_path}")

    # Verify FP8 support and log detailed status before loading
    verify_and_log_fp8_status()

    # LoRA (optional)
    loras: list[LoraPathStrengthAndSDOps] = []
    if config_req.lora_path and Path(config_req.lora_path).is_file():
        loras.append(
            LoraPathStrengthAndSDOps(
                path=str(Path(config_req.lora_path).resolve()),
                strength=config_req.lora_strength,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )

    logger.info(f"Loading pipeline with quantization={ENFORCED_QUANTIZATION_NAME} (enforced, not configurable)")

    _pipeline = BlackwellDistilledPipeline(
        distilled_checkpoint_path=str(Path(config_req.checkpoint_path).resolve()),
        spatial_upsampler_path=str(Path(config_req.upsampler_path).resolve()),
        gemma_root=str(Path(config_req.gemma_path).resolve()),
        loras=loras,
        quantization=ENFORCED_QUANTIZATION,   # Always fp8-cast, hardcoded
    )
    _pipeline_config = config

    # Log post-load VRAM usage
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb  = torch.cuda.memory_reserved()  / (1024 ** 3)
        total_gb     = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free_gb      = total_gb - reserved_gb
        logger.info("=" * 70)
        logger.info("  PIPELINE LOAD COMPLETE — VRAM REPORT")
        logger.info("=" * 70)
        logger.info(f"  Quantization Applied : {ENFORCED_QUANTIZATION_NAME} ✅")
        logger.info(f"  VRAM Allocated       : {allocated_gb:.2f} GB")
        logger.info(f"  VRAM Reserved        : {reserved_gb:.2f} GB")
        logger.info(f"  VRAM Free            : {free_gb:.2f} GB / {total_gb:.1f} GB total")
        logger.info(f"  MAX_CONCURRENT_GENS  : {os.environ.get('MAX_CONCURRENT_GENS', 4)}")
        logger.info(f"  MAX_CPU_TASKS        : {MAX_CPU_TASKS}")
        logger.info("=" * 70)

    return f"Blackwell pipeline loaded successfully. Quantization: {ENFORCED_QUANTIZATION_NAME} (built-in, always active)."


def auto_load_from_env() -> bool:
    """Attempt to load the pipeline using environment variables or default paths."""
    checkpoint = os.environ.get("LTX23_DISTILLED_CHECKPOINT") or "models/ltx-2.3-22b-distilled.safetensors"
    upsampler  = os.environ.get("LTX23_SPATIAL_UPSAMPLER")    or "models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    gemma      = os.environ.get("LTX23_GEMMA_ROOT")           or "models/gemma-3-12b-it-qat-q4_0-unquantized"

    # NOTE: LTX23_QUANTIZATION env var is intentionally ignored.
    # fp8-cast is always enforced regardless of environment configuration.
    if os.environ.get("LTX23_QUANTIZATION") and os.environ.get("LTX23_QUANTIZATION") != ENFORCED_QUANTIZATION_NAME:
        logger.warning(
            f"LTX23_QUANTIZATION is set to '{os.environ.get('LTX23_QUANTIZATION')}' "
            f"but is IGNORED. Server always uses '{ENFORCED_QUANTIZATION_NAME}' (built-in)."
        )

    if Path(checkpoint).exists() and Path(upsampler).exists() and Path(gemma).exists():
        try:
            from api.models import SystemLoadRequest
            req = SystemLoadRequest(
                checkpoint_path=checkpoint,
                upsampler_path=upsampler,
                gemma_path=gemma,
                # quantization field is ignored inside load_pipeline — fp8-cast always applied
            )
            load_pipeline(req)
            logger.info(f"Pipeline auto-loaded. Quantization: {ENFORCED_QUANTIZATION_NAME} (built-in).")
            return True
        except Exception as e:
            logger.warning(f"Failed to auto-load pipeline: {e}")
    else:
        logger.info("Auto-load skipped: One or more model files not found at default or environment paths.")
    return False


# ---------------------------------------------------------------------------
#  Core generation task
# ---------------------------------------------------------------------------
@torch.inference_mode()
async def process_generation_task(task: TaskState) -> None:
    """Execute a generation task securely within the async task queue."""
    from api.task_manager import task_manager

    if _pipeline is None:
        raise ValueError("Pipeline is not loaded.")

    request = task.request

    # Resolve resolution
    if request.use_custom_resolution:
        height, width = request.custom_height, request.custom_width
    else:
        height, width = PRESET_RESOLUTIONS.get(request.resolution_preset, (1024, 1536))

    if height % 64 != 0 or width % 64 != 0:
        raise ValueError(f"Resolution {height}x{width} must be divisible by 64.")

    num_frames = request.num_frames
    if (num_frames - 1) % 8 != 0:
        snapped = ((num_frames - 1) // 8) * 8 + 1
        raise ValueError(f"Frame count must satisfy 8k+1. Try {snapped} instead of {num_frames}.")

    images: list[ImageConditioningInput] = []

    if request.image_base64:
        try:
            head_and_data = request.image_base64.split(",", 1)
            header  = head_and_data[0] if len(head_and_data) > 1 else ""
            data_str = head_and_data[1] if len(head_and_data) > 1 else head_and_data[0]
            img_data = base64.b64decode(data_str)

            mime_match = re.search(r"data:image/(\w+)", header)
            mime_type  = mime_match.group(1).lower() if mime_match else "png"
            ext_map = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp",
                       "gif": "gif", "bmp": "bmp", "tiff": "tiff", "avif": "avif"}
            ext = ext_map.get(mime_type, mime_type)

            upload_dir = OUTPUT_DIR / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved_path = upload_dir / f"{task.task_id}.{ext}"

            with open(saved_path, "wb") as f:
                f.write(img_data)
            request.image_path = str(saved_path)
            logger.info(f"Saved uploaded image as {saved_path} (MIME: {mime_type})")
        except Exception as e:
            logger.error(f"Failed to decode and save base64 image: {e}")
            raise ValueError("Invalid Base64 image data")

    if request.image_path and Path(request.image_path).is_file():
        images.append(
            ImageConditioningInput(
                path=str(Path(request.image_path).resolve()),
                frame_idx=0,
                strength=float(request.image_strength),
                crf=int(request.image_crf),
            )
        )

    seed = request.seed
    if seed < 0:
        seed = torch.randint(0, 2**31, (1,)).item()

    duration_sec = num_frames / request.frame_rate
    task_manager.update_progress(task.task_id, 0.0, f"Initialising | fp8-cast active | {height}x{width} @ {request.frame_rate}fps")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    status_parts = [
        f"Resolution: {height}x{width}",
        f"Frames: {num_frames}",
        f"FPS: {request.frame_rate}",
        f"Est. Duration: {duration_sec:.1f}s",
        f"Seed: {seed}",
        f"Quantization: {ENFORCED_QUANTIZATION_NAME}",
    ]
    if images:
        status_parts.append(f"Image Conditioning Strength: {request.image_strength}")

    task_manager.update_progress(task.task_id, 0.05, "Waiting for CPU/Gemma slot…")

    async with CPU_LOCK:
        task_manager.update_progress(task.task_id, 0.1, "Encoding prompt & building latents…")

        loop = asyncio.get_running_loop()

        def _run_pipeline():
            with torch.no_grad():
                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
                video, audio = _pipeline(
                    prompt=request.prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=request.frame_rate,
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=request.enhance_prompt,
                    cancel_callback=lambda: task.is_cancelled,
                )
                return video, audio, video_chunks_number

        try:
            video, audio, video_chunks_number = await loop.run_in_executor(None, _run_pipeline)
        except InterruptedError:
            logger.info(f"Task {task.task_id} interrupted during GPU generation.")
            return

    if task.is_cancelled:
        return

    task_manager.update_progress(task.task_id, 0.85, "Encoding video to MP4…")

    output_filename = f"ltx2.3_{task.task_id[:8]}.mp4"
    output_path = str(OUTPUT_DIR / output_filename)

    def _run_encoding():
        if task.is_cancelled:
            raise InterruptedError("Task was canceled before encoding completed.")
        with torch.no_grad():
            cleanup_memory()
            encode_video(
                video=video,
                fps=int(request.frame_rate),
                audio=audio,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )

    try:
        await loop.run_in_executor(None, _run_encoding)
    except InterruptedError:
        logger.info(f"Task {task.task_id} interrupted during video encoding.")
        return

    elapsed = time.perf_counter() - t0
    status_parts.append(f"Completed in {elapsed:.1f}s")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024 ** 3)
        status_parts.append(f"Peak VRAM: {peak_vram:.2f} GB | Free VRAM: {free_vram:.2f} GB")

    success_message = " | ".join(status_parts)
    video_url = f"/videos/{output_filename}"

    logger.info(f"Task {task.task_id} done: {success_message}")
    task_manager.mark_completed(task.task_id, video_url=video_url, message=success_message)
