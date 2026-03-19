"""Tests for AgentWorker."""

import pytest
from orchestrator.task import Task, TaskStatus
from orchestrator.worker import AgentWorker


def make_task():
    return Task(title="Test", description="", business_value=5,
                time_criticality=3, risk_reduction=2, job_size=2)


class TestAgentWorker:
    def test_always_success(self):
        worker = AgentWorker("w", success_rate=1.0, rng_seed=0)
        task = make_task()
        result = worker.execute(task)
        assert result.success
        assert task.status == TaskStatus.COMPLETED
        assert task.result is not None
        assert task.error is None

    def test_always_failure(self):
        worker = AgentWorker("w", success_rate=0.0, rng_seed=0)
        task = make_task()
        result = worker.execute(task)
        assert not result.success
        assert task.error is not None

    def test_tracks_completed(self):
        worker = AgentWorker("w", success_rate=1.0, rng_seed=0)
        for _ in range(5):
            worker.execute(make_task())
        assert worker.tasks_completed == 5
        assert worker.tasks_failed == 0

    def test_tracks_failed(self):
        worker = AgentWorker("w", success_rate=0.0, rng_seed=0)
        for _ in range(3):
            worker.execute(make_task())
        assert worker.tasks_failed == 3
        assert worker.tasks_completed == 0

    def test_invalid_success_rate_raises(self):
        with pytest.raises(ValueError):
            AgentWorker("bad", success_rate=1.5)
        with pytest.raises(ValueError):
            AgentWorker("bad", success_rate=-0.1)

    def test_result_contains_worker_name(self):
        worker = AgentWorker("alpha", success_rate=1.0, rng_seed=0)
        task = make_task()
        result = worker.execute(task)
        assert "alpha" in result.output

    def test_deterministic_with_seed(self):
        w1 = AgentWorker("x", success_rate=0.5, rng_seed=99)
        w2 = AgentWorker("x", success_rate=0.5, rng_seed=99)
        results1 = [w1.execute(make_task()).success for _ in range(10)]
        results2 = [w2.execute(make_task()).success for _ in range(10)]
        assert results1 == results2
