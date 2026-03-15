import os
import logging
from contextlib import asynccontextmanager

# Enforce CUDA memory config to avoid fragmentation on Blackwell
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.task_manager import task_manager
from api.engine import process_generation_task, ENFORCED_QUANTIZATION_NAME

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""

    logger.info("=" * 70)
    logger.info("  LTX-2.3 FASTAPI SERVER — STARTING UP")
    logger.info("=" * 70)

    # ── Concurrency limits ──────────────────────────────────────────────────
    from api.task_manager import MAX_CONCURRENT_GENS
    from api.engine import MAX_CPU_TASKS
    logger.info(f"  MAX_CONCURRENT_GENS (GPU Queue) : {MAX_CONCURRENT_GENS}")
    logger.info(f"  MAX_CPU_TASKS (CPU Encoder/IO)  : {MAX_CPU_TASKS}")

    # ── Quantization notice ─────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  QUANTIZATION POLICY")
    logger.info("=" * 70)
    logger.info(f"  Mode     : {ENFORCED_QUANTIZATION_NAME} (built-in, always enforced)")
    logger.info(f"  Scope    : Transformer weights cast to FP8 at inference time")
    logger.info(f"  Benefit  : ~40% VRAM reduction vs BF16 → higher concurrency")
    logger.info(f"  Override : NOT possible — hardcoded for Blackwell 96GB safety")
    logger.info("=" * 70)

    # ── CUDA / GPU info ─────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name   = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cap        = torch.cuda.get_device_capability(0)
            logger.info("  GPU INFO")
            logger.info(f"  GPU              : {gpu_name}")
            logger.info(f"  Compute Cap      : sm_{cap[0]}{cap[1]}")
            logger.info(f"  Total VRAM       : {total_vram:.1f} GB")
            fp8_ok = (cap[0] > 8) or (cap[0] == 8 and cap[1] >= 9)
            logger.info(f"  FP8 Capable      : {'✅ YES' if fp8_ok else '❌ NO — upgrade GPU'}")
        else:
            logger.warning("  ⚠️  CUDA not available — GPU generation will fail!")
    except Exception as e:
        logger.warning(f"  Could not query GPU info: {e}")

    logger.info("=" * 70)

    # ── Model download ──────────────────────────────────────────────────────
    try:
        from api.downloader import download_required_models
        logger.info("Checking and downloading required models if not present...")
        download_required_models()
    except Exception as e:
        logger.error(f"Error while downloading models during startup: {e}")

    # ── Pipeline auto-load ──────────────────────────────────────────────────
    try:
        from api.engine import auto_load_from_env
        if auto_load_from_env():
            logger.info(f"Pipeline ready. Quantization: {ENFORCED_QUANTIZATION_NAME} ✅")
        else:
            logger.info("Pipeline not auto-loaded. Use POST /api/v1/system/load to initialize.")
    except Exception as e:
        logger.error(f"Error during pipeline auto-load: {e}")

    # ── Start task queue workers ────────────────────────────────────────────
    task_manager.start(processor_callback=process_generation_task)
    logger.info("Task queue workers started. Server is ready to accept requests.")
    logger.info("=" * 70)

    yield

    # ── Graceful shutdown ───────────────────────────────────────────────────
    logger.info("Shutting down task queue workers...")
    await task_manager.stop()
    logger.info("Server shut down cleanly.")


app = FastAPI(
    title="LTX-2.3 Video Generation API",
    description=(
        "Professional-grade FastAPI server for LTX-2.3 Distilled Pipeline. "
        f"Quantization: fp8-cast (built-in, always active for Blackwell 96GB VRAM optimization)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory as static file server
app.mount("/videos", StaticFiles(directory="outputs"), name="videos")

# Include API routes
app.include_router(router)


@app.get("/")
def read_root():
    return {
        "service": "LTX-2.3 Video Generation API",
        "quantization": f"{ENFORCED_QUANTIZATION_NAME} (built-in, always active)",
        "docs": "/docs",
        "status": "online",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
