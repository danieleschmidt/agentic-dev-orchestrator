"""AgentWorker — mock worker that executes tasks with configurable success rate."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from .task import Task, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class WorkResult:
    """Result returned by a worker after attempting a task."""

    task_id: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"WorkResult({status} task={self.task_id!r}, {self.duration_ms:.0f}ms)"


class AgentWorker:
    """A mock agent worker that simulates task execution.

    Parameters
    ----------
    name:
        Human-readable worker identifier.
    success_rate:
        Probability (0.0–1.0) that a task succeeds on a given attempt.
    simulate_delay_ms:
        Simulated execution time per task (milliseconds). Set to 0 for tests.
    rng_seed:
        Optional seed for reproducible results in tests.
    """

    def __init__(
        self,
        name: str,
        success_rate: float = 0.8,
        simulate_delay_ms: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError("success_rate must be between 0.0 and 1.0")
        self.name = name
        self.success_rate = success_rate
        self.simulate_delay_ms = simulate_delay_ms
        self._rng = random.Random(rng_seed)
        self._tasks_completed = 0
        self._tasks_failed = 0

    @property
    def tasks_completed(self) -> int:
        return self._tasks_completed

    @property
    def tasks_failed(self) -> int:
        return self._tasks_failed

    def execute(self, task: Task) -> WorkResult:
        """Execute a task and return a WorkResult.

        Updates task.status, task.result / task.error in-place.
        """
        logger.info("[%s] Starting task %s: %s", self.name, task.id, task.title)
        task.status = TaskStatus.IN_PROGRESS

        start = time.monotonic()
        if self.simulate_delay_ms > 0:
            time.sleep(self.simulate_delay_ms / 1000.0)
        duration_ms = (time.monotonic() - start) * 1000

        success = self._rng.random() < self.success_rate

        if success:
            task.status = TaskStatus.COMPLETED
            task.result = f"Worker {self.name!r} completed: {task.title}"
            task.error = None
            self._tasks_completed += 1
            logger.info("[%s] ✓ Task %s completed (%.0fms)", self.name, task.id, duration_ms)
            return WorkResult(
                task_id=task.id,
                success=True,
                output=task.result,
                duration_ms=duration_ms,
            )
        else:
            task.error = f"Worker {self.name!r} failed on attempt {task.retries + 1}"
            self._tasks_failed += 1
            logger.warning("[%s] ✗ Task %s failed: %s", self.name, task.id, task.error)
            return WorkResult(
                task_id=task.id,
                success=False,
                error=task.error,
                duration_ms=duration_ms,
            )

    def __repr__(self) -> str:
        return (
            f"AgentWorker(name={self.name!r}, success_rate={self.success_rate:.0%}, "
            f"done={self._tasks_completed}, failed={self._tasks_failed})"
        )
