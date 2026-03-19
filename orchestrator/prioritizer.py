"""WSJF Prioritizer — ranks tasks by Weighted Shortest Job First score."""

from __future__ import annotations

from typing import List

from .task import Task, TaskStatus


class WSJFPrioritizer:
    """Ranks tasks by WSJF score (highest first).

    WSJF = (Business Value + Time Criticality + Risk Reduction) / Job Size

    Only PENDING and RETRYING tasks are eligible for dispatch.
    """

    ELIGIBLE_STATUSES = {TaskStatus.PENDING, TaskStatus.RETRYING}

    def rank(self, tasks: List[Task]) -> List[Task]:
        """Return tasks sorted by WSJF score descending (highest priority first)."""
        eligible = [t for t in tasks if t.status in self.ELIGIBLE_STATUSES]
        return sorted(eligible, key=lambda t: t.wsjf_score, reverse=True)

    def top(self, tasks: List[Task], n: int = 1) -> List[Task]:
        """Return the top-n highest priority tasks."""
        return self.rank(tasks)[:n]

    def print_ranked(self, tasks: List[Task]) -> None:
        """Print a ranked table of eligible tasks."""
        ranked = self.rank(tasks)
        if not ranked:
            print("No eligible tasks.")
            return

        header = f"{'#':<4} {'ID':<10} {'WSJF':>6}  {'BV':>4} {'TC':>4} {'RR':>4} {'JS':>4}  {'Title'}"
        print(header)
        print("-" * len(header))
        for i, task in enumerate(ranked, 1):
            print(
                f"{i:<4} {task.id:<10} {task.wsjf_score:>6.2f}  "
                f"{task.business_value:>4.0f} {task.time_criticality:>4.0f} "
                f"{task.risk_reduction:>4.0f} {task.job_size:>4.0f}  {task.title}"
            )
