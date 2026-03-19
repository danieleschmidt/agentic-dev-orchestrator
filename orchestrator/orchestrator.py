"""Orchestrator — dispatches WSJF-ranked tasks to workers, handles retries."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .backlog import TaskBacklog
from .prioritizer import WSJFPrioritizer
from .task import Task, TaskStatus
from .worker import AgentWorker, WorkResult

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorStats:
    dispatched: int = 0
    completed: int = 0
    failed: int = 0
    retried: int = 0
    cycles: int = 0


class Orchestrator:
    """Dispatches tasks from a TaskBacklog to AgentWorkers in WSJF order.

    On each cycle:
    1. Rank all eligible tasks by WSJF score.
    2. Dispatch the highest-priority task to an available worker (round-robin).
    3. On failure: retry up to task.max_retries times, then mark FAILED.

    Parameters
    ----------
    backlog:
        The task backlog to draw from.
    workers:
        List of AgentWorker instances.
    tasks_per_cycle:
        How many tasks to dispatch per run() cycle (default: one per worker).
    """

    def __init__(
        self,
        backlog: TaskBacklog,
        workers: List[AgentWorker],
        tasks_per_cycle: Optional[int] = None,
    ) -> None:
        if not workers:
            raise ValueError("At least one worker is required.")
        self.backlog = backlog
        self.workers = workers
        self.tasks_per_cycle = tasks_per_cycle or len(workers)
        self._prioritizer = WSJFPrioritizer()
        self._worker_index = 0
        self.stats = OrchestratorStats()
        self._results: List[WorkResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[WorkResult]:
        """Run one orchestration cycle: dispatch up to tasks_per_cycle tasks.

        Returns the list of WorkResults from this cycle.
        """
        self.stats.cycles += 1
        cycle_results: List[WorkResult] = []

        ranked = self._prioritizer.rank(self.backlog.all())
        to_dispatch = ranked[: self.tasks_per_cycle]

        if not to_dispatch:
            logger.info("Cycle %d: no eligible tasks.", self.stats.cycles)
            return cycle_results

        logger.info(
            "Cycle %d: dispatching %d task(s) across %d worker(s).",
            self.stats.cycles,
            len(to_dispatch),
            len(self.workers),
        )

        for task in to_dispatch:
            worker = self._next_worker()
            result = self._dispatch(task, worker)
            cycle_results.append(result)
            self._results.append(result)

        return cycle_results

    def run_until_done(self, max_cycles: int = 100) -> List[WorkResult]:
        """Run cycles until the backlog has no eligible tasks or max_cycles hit."""
        for _ in range(max_cycles):
            if not self._has_eligible_tasks():
                break
            self.run()
        return self._results

    def print_summary(self) -> None:
        """Print a summary of orchestration results."""
        s = self.stats
        bl = self.backlog.stats()
        print("\n" + "=" * 60)
        print("ORCHESTRATION SUMMARY")
        print("=" * 60)
        print(f"  Cycles run   : {s.cycles}")
        print(f"  Dispatched   : {s.dispatched}")
        print(f"  Completed    : {s.completed}")
        print(f"  Retried      : {s.retried}")
        print(f"  Failed       : {s.failed}")
        print(f"  Backlog stats: {bl}")
        print("=" * 60)

        print("\nWorker Performance:")
        for w in self.workers:
            print(f"  {w.name}: {w.tasks_completed} completed, {w.tasks_failed} failed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch(self, task: Task, worker: AgentWorker) -> WorkResult:
        self.stats.dispatched += 1
        result = worker.execute(task)

        if result.success:
            self.stats.completed += 1
        else:
            task.retries += 1
            if task.retries <= task.max_retries:
                task.status = TaskStatus.RETRYING
                task.error = result.error
                self.stats.retried += 1
                logger.info(
                    "Task %s queued for retry (%d/%d).",
                    task.id,
                    task.retries,
                    task.max_retries,
                )
            else:
                task.status = TaskStatus.FAILED
                self.stats.failed += 1
                logger.error(
                    "Task %s permanently failed after %d attempts.",
                    task.id,
                    task.retries,
                )

        return result

    def _next_worker(self) -> AgentWorker:
        """Round-robin worker selection."""
        worker = self.workers[self._worker_index % len(self.workers)]
        self._worker_index += 1
        return worker

    def _has_eligible_tasks(self) -> bool:
        eligible = {TaskStatus.PENDING, TaskStatus.RETRYING}
        return any(t.status in eligible for t in self.backlog.all())
