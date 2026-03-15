import asyncio
import os
import uuid
import logging
from typing import Dict, Optional, Callable, Awaitable
from api.models import TaskStatus, VideoGenerationRequest

logger = logging.getLogger(__name__)


class TaskState:
    def __init__(self, request: VideoGenerationRequest):
        self.task_id = str(uuid.uuid4())
        self.request = request
        self.status = TaskStatus.QUEUED
        self.progress = 0.0
        self.message = "Task queued."
        self.video_url: Optional[str] = None
        self.is_cancelled: bool = False


class TaskManager:
    def __init__(self, max_concurrent_tasks: int = 4):
        self.tasks: Dict[str, TaskState] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.processor_callback: Optional[Callable[[TaskState], Awaitable[None]]] = None
        self._workers: list[asyncio.Task] = []
        self._is_running = False

    def get_task(self, task_id: str) -> Optional[TaskState]:
        return self.tasks.get(task_id)

    async def submit_task(self, request: VideoGenerationRequest) -> str:
        state = TaskState(request)
        self.tasks[state.task_id] = state
        await self.queue.put(state.task_id)
        logger.info(f"Task {state.task_id} queued. Queue depth: {self.queue.qsize()}")
        return state.task_id

    def update_progress(self, task_id: str, progress: float, message: str):
        if task_id in self.tasks:
            self.tasks[task_id].progress = progress
            self.tasks[task_id].message = message

    def mark_completed(self, task_id: str, video_url: str, message: str = "Video generation complete."):
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].progress = 1.0
            self.tasks[task_id].video_url = video_url
            self.tasks[task_id].message = message
            logger.info(f"Task {task_id} completed: {video_url}")

    def mark_failed(self, task_id: str, error_message: str):
        if task_id in self.tasks:
            if self.tasks[task_id].is_cancelled:
                return  # Skip failing a canceled task
            self.tasks[task_id].status = TaskStatus.FAILED
            self.tasks[task_id].message = error_message
            logger.error(f"Task {task_id} failed: {error_message}")

    def cancel_task(self, task_id: str) -> bool:
        if task_id in self.tasks:
            state = self.tasks[task_id]
            if state.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False  # Too late to cancel
            state.is_cancelled = True
            state.status = TaskStatus.CANCELED
            state.message = "Task was canceled by the user."
            logger.info(f"Task {task_id} canceled.")
            return True
        return False

    async def _worker_loop(self):
        while self._is_running:
            try:
                task_id = await self.queue.get()
                state = self.tasks.get(task_id)

                if not state or not self.processor_callback:
                    self.queue.task_done()
                    continue

                if state.is_cancelled:
                    logger.info(f"Task {task_id} skipped — canceled while in queue.")
                    self.queue.task_done()
                    continue

                state.status = TaskStatus.PROCESSING
                state.message = "Starting generation..."

                async with self.semaphore:
                    logger.info(f"Processing task {task_id} | Active semaphore slots used.")
                    try:
                        if state.is_cancelled:
                            logger.info(f"Task {task_id} canceled just before GPU execution.")
                            continue
                        await self.processor_callback(state)
                    except Exception as e:
                        logger.exception(f"Exception while processing task {task_id}")
                        self.mark_failed(task_id, str(e))

                self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker encountered unexpected error: {e}")

    def start(self, processor_callback: Callable[[TaskState], Awaitable[None]]):
        if self._is_running:
            return
        self.processor_callback = processor_callback
        self._is_running = True

        # Spawn one worker per semaphore slot
        num_workers = self.semaphore._value
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop())
            self._workers.append(task)

        logger.info(f"TaskManager started with {num_workers} worker(s). Max concurrent: {num_workers}")

    async def stop(self):
        self._is_running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("TaskManager stopped cleanly.")


# ---------------------------------------------------------------------------
#  Global instance
# ---------------------------------------------------------------------------
MAX_CONCURRENT_GENS = int(os.environ.get("MAX_CONCURRENT_GENS", 4))
task_manager = TaskManager(max_concurrent_tasks=MAX_CONCURRENT_GENS)
