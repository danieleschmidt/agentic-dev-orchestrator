"""Tests for WSJFPrioritizer."""

from orchestrator.task import Task, TaskStatus
from orchestrator.prioritizer import WSJFPrioritizer


def task(title, bv, tc, rr, js, status=TaskStatus.PENDING):
    t = Task(title=title, description="", business_value=bv,
             time_criticality=tc, risk_reduction=rr, job_size=js)
    t.status = status
    return t


class TestWSJFPrioritizer:
    def setup_method(self):
        self.p = WSJFPrioritizer()

    def test_ranks_highest_wsjf_first(self):
        low  = task("Low",  bv=1, tc=1, rr=1, js=3)   # score = 1
        high = task("High", bv=8, tc=8, rr=8, js=2)   # score = 12
        mid  = task("Mid",  bv=5, tc=3, rr=2, js=2)   # score = 5

        ranked = self.p.rank([low, high, mid])
        assert ranked[0].title == "High"
        assert ranked[1].title == "Mid"
        assert ranked[2].title == "Low"

    def test_only_includes_eligible_statuses(self):
        pending  = task("Pending",  bv=5, tc=5, rr=5, js=1, status=TaskStatus.PENDING)
        retrying = task("Retrying", bv=5, tc=5, rr=5, js=1, status=TaskStatus.RETRYING)
        done     = task("Done",     bv=9, tc=9, rr=9, js=1, status=TaskStatus.COMPLETED)
        failed   = task("Failed",   bv=9, tc=9, rr=9, js=1, status=TaskStatus.FAILED)

        ranked = self.p.rank([pending, retrying, done, failed])
        titles = [t.title for t in ranked]
        assert "Pending" in titles
        assert "Retrying" in titles
        assert "Done" not in titles
        assert "Failed" not in titles

    def test_top_returns_n_highest(self):
        tasks = [task(f"T{i}", bv=i, tc=1, rr=1, js=1) for i in range(1, 6)]
        top2 = self.p.top(tasks, n=2)
        assert len(top2) == 2
        assert top2[0].business_value == 5
        assert top2[1].business_value == 4

    def test_empty_returns_empty(self):
        assert self.p.rank([]) == []

    def test_all_completed_returns_empty(self):
        tasks = [task("X", bv=5, tc=5, rr=5, js=1, status=TaskStatus.COMPLETED)]
        assert self.p.rank(tasks) == []
