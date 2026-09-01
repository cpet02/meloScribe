"""Background job management.

Transcription takes minutes, so it cannot run inside a request handler: the
browser would time out and the user would have no idea whether anything was
happening. Jobs therefore run on a worker thread and the UI polls for progress.

A thread pool is the right size of tool here. The work is GPU- and
subprocess-bound rather than CPU-bound in Python, so the GIL is not the
constraint, and a single-user local app does not need Celery and a broker.
Concurrency is capped at one by default because two simultaneous separations
will simply fight over the same GPU memory.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class JobStatus(str, Enum):
    QUEUED = 'queued'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


@dataclass
class Job:
    """One transcription request and everything known about its progress."""
    id: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = 'queued'
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    params: Dict[str, Any] = field(default_factory=dict)
    _cancelled: bool = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def public(self) -> Dict[str, Any]:
        """The JSON view. Never includes the result payload, which can be
        large - it is fetched separately once the job is done."""
        data = {
            'id': self.id,
            'filename': self.filename,
            'status': self.status.value,
            'progress': round(self.progress, 4),
            'message': self.message,
            'created_at': self.created_at,
            'finished_at': self.finished_at,
            'error': self.error,
            'params': self.params,
        }
        if self.status == JobStatus.DONE and self.result is not None:
            data['summary'] = self.result.to_dict()
        return data


class JobCancelled(RuntimeError):
    """Raised inside a worker when the user cancels the job."""


class JobStore:
    """Thread-safe registry of jobs plus the pool that runs them."""

    def __init__(self, max_workers: int = 1, history: int = 50):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix='meloscribe')
        self._history = history

    def create(self, filename: str, params: Dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, params=params)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at,
                          reverse=True)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation.

        Cooperative: the worker checks the flag at each progress callback, so a
        job stops at the next stage boundary rather than instantly. Killing a
        thread mid-separation would leave partial files behind, and the cache
        would then have to defend against them.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in (JobStatus.DONE, JobStatus.FAILED):
                return False
            job._cancelled = True
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.message = 'cancelled'
            return True

    def submit(self, job: Job, work: Callable[[Job, Callable], Any]) -> None:
        """Queue `work`, handing it the job and a progress reporter."""

        def report(fraction: float, message: str) -> None:
            if job.cancelled:
                raise JobCancelled()
            job.progress = max(0.0, min(1.0, fraction))
            job.message = message

        def run() -> None:
            if job.cancelled:
                return
            job.status = JobStatus.RUNNING
            job.message = 'starting'
            try:
                job.result = work(job, report)
                job.status = JobStatus.DONE
                job.progress = 1.0
                job.message = 'complete'
            except JobCancelled:
                job.status = JobStatus.CANCELLED
                job.message = 'cancelled'
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = f"{exc.__class__.__name__}: {exc}"
                job.message = 'failed'
                # Kept server-side only: the traceback is for the log, while
                # the client gets the readable one-line error above.
                traceback.print_exc()
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()

        self._pool.submit(run)

    def _prune(self) -> None:
        """Drop the oldest finished jobs once history is exceeded."""
        if len(self._jobs) <= self._history:
            return
        finished = sorted(
            (j for j in self._jobs.values()
             if j.status in (JobStatus.DONE, JobStatus.FAILED,
                             JobStatus.CANCELLED)),
            key=lambda j: j.created_at)
        for job in finished[:len(self._jobs) - self._history]:
            self._jobs.pop(job.id, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
