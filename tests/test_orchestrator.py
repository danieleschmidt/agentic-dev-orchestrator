"""Tests for Orchestrator."""

import pytest
from orchestrator.task import Task, TaskStatus
from orchestrator.backlog import TaskBacklog
from orchestrator.worker import AgentWorker
from orchestrator.orchestrator import Orchestrator


def make_task(title="T", bv=5, tc=3, rr=2, js=2):
    return Task(title=title, description="", business_value=bv,
                time_criticality=tc, risk_reduction=rr, job_size=js)


def backlog_with(*tasks):
    bl = TaskBacklog()
    for t in tasks:
        bl.add(t)
    return bl


class TestOrchestratorBasic:
    def test_no_workers_raises(self):
        with pytest.raises(ValueError, match="At least one worker"):
            Orchestrator(TaskBacklog(), [])

    def test_completes_all_tasks_with_perfect_worker(self):
        bl = backlog_with(make_task("A"), make_task("B"), make_task("C"))
        worker = AgentWorker("w", success_rate=1.0, rng_seed=0)
        orch = Orchestrator(bl, [worker], tasks_per_cycle=3)
        orch.run_until_done()
        assert len(bl.completed()) == 3
        assert orch.stats.completed == 3
        assert orch.stats.failed == 0

    def test_failing_worker_retries_and_fails(self):
        t = make_task("FailMe")
        t.max_retries = 2
        bl = backlog_with(t)
        worker = AgentWorker("w", success_rate=0.0, rng_seed=0)
        orch = Orchestrator(bl, [worker])
        orch.run_until_done(max_cycles=10)
        assert t.status == TaskStatus.FAILED
        assert orch.stats.retried == 2  # tried max_retries times before failing

    def test_dispatches_in_wsjf_order(self):
        """Higher WSJF task should be dispatched first."""
        low  = make_task("Low",  bv=1, tc=1, rr=1, js=3)   # WSJF ≈ 1
        high = make_task("High", bv=8, tc=8, rr=8, js=1)   # WSJF = 24

        dispatched_order = []

        class TrackingWorker(AgentWorker):
            def execute(self, task):
                dispatched_order.append(task.title)
                return super().execute(task)

        bl = backlog_with(low, high)
        worker = TrackingWorker("tracker", success_rate=1.0, rng_seed=0)
        orch = Orchestrator(bl, [worker], tasks_per_cycle=1)
        orch.run_until_done()
        assert dispatched_order[0] == "High"
        assert dispatched_order[1] == "Low"

    def test_cycles_tracked(self):
        bl = backlog_with(make_task(), make_task())
        worker = AgentWorker("w", success_rate=1.0, rng_seed=0)
        orch = Orchestrator(bl, [worker], tasks_per_cycle=1)
        orch.run_until_done()
        assert orch.stats.cycles >= 2

    def test_empty_backlog_runs_one_cycle(self):
        orch = Orchestrator(TaskBacklog(), [AgentWorker("w")])
        results = orch.run()
        assert results == []

    def test_round_robin_worker_selection(self):
        used = []

        class NamedWorker(AgentWorker):
            def execute(self, task):
                used.append(self.name)
                return super().execute(task)

        workers = [NamedWorker(f"w{i}", success_rate=1.0, rng_seed=i) for i in range(3)]
        tasks = [make_task(f"T{i}") for i in range(6)]
        bl = backlog_with(*tasks)
        orch = Orchestrator(bl, workers, tasks_per_cycle=6)
        orch.run()
        # Each worker should have been used exactly twice
        for w in workers:
            assert used.count(w.name) == 2
