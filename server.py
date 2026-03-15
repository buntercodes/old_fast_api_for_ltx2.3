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
from api.engine import process_generation_task

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    logger.info("Starting LTX-2.3 FastAPI Server...")
    
    from api.task_manager import MAX_CONCURRENT_GENS
    from api.engine import MAX_CPU_TASKS
    logger.info(f"Concurrency Limits -> MAX_CONCURRENT_GENS (GPU Queue): {MAX_CONCURRENT_GENS}")
    logger.info(f"Concurrency Limits -> MAX_CPU_TASKS (CPU Encoder/IO): {MAX_CPU_TASKS}")
    
    # Pre-download required models if missing
    try:
        from api.downloader import download_required_models
        logger.info("Checking and downloading required models if they don't exist...")
        download_required_models()
    except Exception as e:
        logger.error(f"Error while downloading models during startup: {e}")
        
    # Auto-load pipeline if models are available
    try:
        from api.engine import auto_load_from_env
        if auto_load_from_env():
            logger.info("Blackwell Optimized Pipeline ready on startup.")
        else:
            logger.info("Pipeline not auto-loaded. Please use /api/v1/system/load if models are present.")
    except Exception as e:
        logger.error(f"Error during pipeline auto-load: {e}")
        
    # Start the task queue workers passing our core generation function
    task_manager.start(processor_callback=process_generation_task)
    
    yield
    
    # Graceful shutdown
    logger.info("Shutting down workers...")
    await task_manager.stop()


app = FastAPI(
    title="LTX-2.3 Video Generation API",
    description="Professional-grade FastAPI server for LTX-2.3 Distilled Pipeline.",
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

# Mount outputs directory as a static file server to access generated videos
app.mount("/videos", StaticFiles(directory="outputs"), name="videos")

# Include API routes
app.include_router(router)


@app.get("/")
def read_root():
    return {
        "service": "LTX-2.3 Video Generation API",
        "docs": "/docs",
        "status": "online"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
