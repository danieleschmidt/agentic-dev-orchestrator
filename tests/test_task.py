"""Tests for Task dataclass and WSJF scoring."""

import pytest
from orchestrator.task import Task, TaskStatus


def make_task(**kwargs):
    defaults = dict(title="Test Task", description="desc", business_value=5,
                    time_criticality=3, risk_reduction=2, job_size=2)
    defaults.update(kwargs)
    return Task(**defaults)


class TestTaskCreation:
    def test_default_status_is_pending(self):
        t = make_task()
        assert t.status == TaskStatus.PENDING

    def test_auto_id_assigned(self):
        t = make_task()
        assert t.id and len(t.id) == 8

    def test_job_size_zero_raises(self):
        with pytest.raises(ValueError, match="job_size must be > 0"):
            make_task(job_size=0)

    def test_job_size_negative_raises(self):
        with pytest.raises(ValueError):
            make_task(job_size=-1)


class TestWSJFScore:
    def test_basic_score(self):
        t = make_task(business_value=5, time_criticality=3, risk_reduction=2, job_size=2)
        assert t.wsjf_score == pytest.approx((5 + 3 + 2) / 2)

    def test_higher_cost_of_delay_scores_higher(self):
        low = make_task(business_value=1, time_criticality=1, risk_reduction=1, job_size=1)
        high = make_task(business_value=13, time_criticality=8, risk_reduction=5, job_size=1)
        assert high.wsjf_score > low.wsjf_score

    def test_larger_job_size_scores_lower(self):
        small = make_task(business_value=5, time_criticality=3, risk_reduction=2, job_size=1)
        large = make_task(business_value=5, time_criticality=3, risk_reduction=2, job_size=10)
        assert small.wsjf_score > large.wsjf_score

    def test_to_dict_includes_wsjf_score(self):
        t = make_task()
        d = t.to_dict()
        assert "wsjf_score" in d
        assert d["wsjf_score"] == pytest.approx(t.wsjf_score, abs=0.01)


class TestTaskSerialization:
    def test_from_dict_roundtrip(self):
        t = make_task(title="Roundtrip", business_value=8, time_criticality=5,
                      risk_reduction=3, job_size=4)
        d = t.to_dict()
        t2 = Task.from_dict(d)
        assert t2.title == t.title
        assert t2.business_value == t.business_value
        assert t2.wsjf_score == pytest.approx(t.wsjf_score)

    def test_from_dict_defaults(self):
        t = Task.from_dict({"title": "Minimal"})
        assert t.business_value == 1.0
        assert t.job_size == 1.0
        assert t.status == TaskStatus.PENDING

    def test_from_dict_status_string(self):
        t = Task.from_dict({"title": "Done", "status": "completed"})
        assert t.status == TaskStatus.COMPLETED
