"""TaskBacklog — CRUD operations, filtering, and JSON import/export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

from .task import Task, TaskStatus


class TaskBacklog:
    """Manages a collection of Tasks.

    Supports add, remove, get, update, filter, and bulk import/export.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, task: Task) -> Task:
        """Add a task to the backlog. Raises if ID already exists."""
        if task.id in self._tasks:
            raise ValueError(f"Task with id {task.id!r} already exists.")
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task:
        """Retrieve a task by ID. Raises KeyError if not found."""
        if task_id not in self._tasks:
            raise KeyError(f"No task with id {task_id!r}.")
        return self._tasks[task_id]

    def update(self, task_id: str, **kwargs) -> Task:
        """Update fields on an existing task."""
        task = self.get(task_id)
        for key, value in kwargs.items():
            if not hasattr(task, key):
                raise AttributeError(f"Task has no attribute {key!r}.")
            setattr(task, key, value)
        return task

    def remove(self, task_id: str) -> Task:
        """Remove and return a task by ID."""
        task = self.get(task_id)
        del self._tasks[task_id]
        return task

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all(self) -> List[Task]:
        return list(self._tasks.values())

    def filter(self, predicate: Callable[[Task], bool]) -> List[Task]:
        return [t for t in self._tasks.values() if predicate(t)]

    def by_status(self, status: TaskStatus) -> List[Task]:
        return self.filter(lambda t: t.status == status)

    def pending(self) -> List[Task]:
        return self.by_status(TaskStatus.PENDING)

    def completed(self) -> List[Task]:
        return self.by_status(TaskStatus.COMPLETED)

    def failed(self) -> List[Task]:
        return self.by_status(TaskStatus.FAILED)

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    # ------------------------------------------------------------------
    # Bulk import / export
    # ------------------------------------------------------------------

    def import_json(self, data: list | str | Path) -> List[Task]:
        """Import tasks from a JSON list, JSON string, or file path.

        Returns the list of newly added tasks.
        """
        if isinstance(data, Path):
            with data.open() as f:
                raw = json.load(f)
        elif isinstance(data, str):
            stripped = data.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                # Inline JSON string
                raw = json.loads(stripped)
            else:
                # Treat as a file path
                with Path(data).open() as f:
                    raw = json.load(f)
        else:
            raw = data

        if not isinstance(raw, list):
            raise ValueError("JSON import expects a list of task objects.")

        added: List[Task] = []
        for item in raw:
            task = Task.from_dict(item)
            # Allow re-import without collision by skipping existing IDs
            if task.id in self._tasks:
                continue
            self._tasks[task.id] = task
            added.append(task)
        return added

    def export_json(self, path: Optional[str | Path] = None) -> str:
        """Export all tasks as JSON. Optionally write to a file."""
        payload = json.dumps([t.to_dict() for t in self.all()], indent=2)
        if path:
            Path(path).write_text(payload)
        return payload

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for task in self._tasks.values():
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        return {
            "total": len(self._tasks),
            "by_status": counts,
        }
