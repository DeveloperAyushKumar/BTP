"""Job Manager - Handles generation request lifecycle."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationJob:
    """Represents a queued generation job."""
    job_id: str
    image: Image.Image
    preset_name: str
    seed: int
    num_frames: int = 14
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None


class JobManager:
    """Manages the lifecycle of generation jobs."""

    def __init__(self):
        self._current_job: Optional[GenerationJob] = None
        self._lock = threading.Lock()

    @property
    def current_job(self) -> Optional[GenerationJob]:
        return self._current_job

    def create_job(
        self,
        image: Image.Image,
        preset_name: str,
        seed: int = 42,
        **kwargs,
    ) -> GenerationJob:
        """Create a new generation job."""
        import uuid
        job = GenerationJob(
            job_id=str(uuid.uuid4())[:8],
            image=image,
            preset_name=preset_name,
            seed=seed,
            **kwargs,
        )
        self._current_job = job
        logger.info(f"Job created: {job.job_id}")
        return job

    def update_progress(self, step: int, total: int, elapsed: float):
        """Update progress of current job."""
        if self._current_job:
            self._current_job.progress = step / total

    def complete_job(self, result: Any):
        """Mark current job as completed."""
        if self._current_job:
            self._current_job.status = JobStatus.COMPLETED
            self._current_job.result = result
            self._current_job.progress = 1.0

    def fail_job(self, error: str):
        """Mark current job as failed."""
        if self._current_job:
            self._current_job.status = JobStatus.FAILED
            self._current_job.error = error
            logger.error(f"Job failed: {error}")
