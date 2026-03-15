from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.models import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    TaskStatusResponse,
    SystemLoadRequest,
    SystemStatusResponse,
)
from api.task_manager import task_manager
from api.engine import load_pipeline, get_pipeline_status, ENFORCED_QUANTIZATION_NAME

router = APIRouter(prefix="/api/v1")


@router.post("/system/load", response_model=SystemStatusResponse)
def api_load_system(request: SystemLoadRequest):
    """
    Initialize or update the LTX-2.3 pipeline in memory.

    Note: The `quantization` field in the request body is ignored.
    The server always uses fp8-cast (built-in) for Blackwell 96GB optimization.
    """
    try:
        msg = load_pipeline(request)
        is_loaded, _ = get_pipeline_status()
        return SystemStatusResponse(
            pipeline_loaded=is_loaded,
            message=msg,
            active_tasks=task_manager.queue.qsize(),
            quantization=ENFORCED_QUANTIZATION_NAME,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/system/status", response_model=SystemStatusResponse)
def api_system_status():
    """
    Check if the pipeline is currently loaded.
    Always reports quantization as fp8-cast (built-in, not configurable).
    """
    is_loaded, msg = get_pipeline_status()
    return SystemStatusResponse(
        pipeline_loaded=is_loaded,
        message=msg,
        active_tasks=task_manager.queue.qsize(),
        quantization=ENFORCED_QUANTIZATION_NAME,
    )


@router.post("/generate", response_model=VideoGenerationResponse)
async def api_generate_video(request: VideoGenerationRequest):
    """Submit a video generation task to the background queue."""
    is_loaded, msg = get_pipeline_status()
    if not is_loaded:
        raise HTTPException(
            status_code=400,
            detail="Pipeline not loaded. Call POST /api/v1/system/load first.",
        )
    try:
        task_id = await task_manager.submit_task(request)
        return VideoGenerationResponse(
            task_id=task_id,
            message=f"Task successfully queued. Quantization: {ENFORCED_QUANTIZATION_NAME} (built-in).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue task: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def api_get_task_status(task_id: str):
    """Poll the status of a specific generation task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found.")

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        message=task.message,
        video_url=task.video_url,
        is_cancelled=task.is_cancelled,
    )


@router.delete("/tasks/{task_id}")
def api_cancel_task(task_id: str):
    """Cancel a specific video generation task."""
    success = task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Task ID not found, or it is already completed/failed and cannot be cancelled.",
        )
    return {"message": "Task successfully canceled."}
