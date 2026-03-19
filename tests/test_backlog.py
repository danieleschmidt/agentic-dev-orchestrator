"""Tests for TaskBacklog."""

import json
import pytest
from orchestrator.task import Task, TaskStatus
from orchestrator.backlog import TaskBacklog


def make_task(title="Task", bv=5, tc=3, rr=2, js=2):
    return Task(title=title, description="", business_value=bv,
                time_criticality=tc, risk_reduction=rr, job_size=js)


class TestTaskBacklogCRUD:
    def setup_method(self):
        self.bl = TaskBacklog()
        self.t1 = make_task("Alpha")
        self.t2 = make_task("Beta")

    def test_add_and_len(self):
        self.bl.add(self.t1)
        assert len(self.bl) == 1

    def test_add_duplicate_id_raises(self):
        self.bl.add(self.t1)
        with pytest.raises(ValueError, match="already exists"):
            self.bl.add(self.t1)

    def test_get_existing(self):
        self.bl.add(self.t1)
        assert self.bl.get(self.t1.id) is self.t1

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            self.bl.get("nonexistent")

    def test_remove(self):
        self.bl.add(self.t1)
        removed = self.bl.remove(self.t1.id)
        assert removed is self.t1
        assert len(self.bl) == 0

    def test_update_field(self):
        self.bl.add(self.t1)
        self.bl.update(self.t1.id, business_value=10)
        assert self.bl.get(self.t1.id).business_value == 10

    def test_update_bad_field_raises(self):
        self.bl.add(self.t1)
        with pytest.raises(AttributeError):
            self.bl.update(self.t1.id, nonexistent_field=99)

    def test_contains(self):
        self.bl.add(self.t1)
        assert self.t1.id in self.bl
        assert "missing" not in self.bl


class TestTaskBacklogFilters:
    def setup_method(self):
        self.bl = TaskBacklog()
        self.t1 = make_task("A")
        self.t2 = make_task("B")
        self.t2.status = TaskStatus.COMPLETED
        self.bl.add(self.t1)
        self.bl.add(self.t2)

    def test_pending(self):
        assert len(self.bl.pending()) == 1

    def test_completed(self):
        assert len(self.bl.completed()) == 1

    def test_filter(self):
        result = self.bl.filter(lambda t: t.title == "A")
        assert len(result) == 1 and result[0].title == "A"

    def test_stats(self):
        s = self.bl.stats()
        assert s["total"] == 2
        assert s["by_status"]["pending"] == 1
        assert s["by_status"]["completed"] == 1


class TestTaskBacklogJSON:
    def test_import_json_list(self):
        bl = TaskBacklog()
        data = [
            {"title": "T1", "business_value": 5, "time_criticality": 3,
             "risk_reduction": 2, "job_size": 2},
            {"title": "T2", "business_value": 8, "time_criticality": 5,
             "risk_reduction": 3, "job_size": 3},
        ]
        added = bl.import_json(data)
        assert len(added) == 2
        assert len(bl) == 2

    def test_import_json_string(self):
        bl = TaskBacklog()
        data = json.dumps([{"title": "From string", "job_size": 1}])
        added = bl.import_json(data)
        assert len(added) == 1

    def test_import_skips_duplicate_ids(self):
        bl = TaskBacklog()
        t = make_task("Original")
        bl.add(t)
        data = [{"id": t.id, "title": "Duplicate", "job_size": 1}]
        added = bl.import_json(data)
        assert len(added) == 0  # skipped
        assert bl.get(t.id).title == "Original"

    def test_export_roundtrip(self):
        bl = TaskBacklog()
        bl.import_json([{"title": "Export me", "business_value": 7, "job_size": 2}])
        exported = bl.export_json()
        data = json.loads(exported)
        assert len(data) == 1
        assert data[0]["title"] == "Export me"
        assert "wsjf_score" in data[0]
