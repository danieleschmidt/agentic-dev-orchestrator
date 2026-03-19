"""Task dataclass and status enum for the orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """A coding task with WSJF scoring attributes.

    WSJF score = (business_value + time_criticality + risk_reduction) / job_size

    All numeric fields use a 1-10 Fibonacci-ish scale (1, 2, 3, 5, 8, 13, 20).
    job_size must be > 0 to avoid division by zero.
    """

    title: str
    description: str

    # WSJF cost-of-delay components (1-20 scale)
    business_value: float = 1.0
    time_criticality: float = 1.0
    risk_reduction: float = 1.0

    # WSJF job size (effort/duration estimate, must be > 0)
    job_size: float = 1.0

    # Managed fields
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: TaskStatus = field(default=TaskStatus.PENDING)
    retries: int = 0
    max_retries: int = 2
    result: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.job_size <= 0:
            raise ValueError(f"job_size must be > 0, got {self.job_size}")

    @property
    def wsjf_score(self) -> float:
        """Calculate WSJF score: cost of delay / job size."""
        cost_of_delay = self.business_value + self.time_criticality + self.risk_reduction
        return cost_of_delay / self.job_size

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "business_value": self.business_value,
            "time_criticality": self.time_criticality,
            "risk_reduction": self.risk_reduction,
            "job_size": self.job_size,
            "wsjf_score": round(self.wsjf_score, 3),
            "status": self.status.value,
            "retries": self.retries,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create a Task from a dictionary (e.g. JSON import)."""
        status_raw = data.get("status", "pending")
        status = TaskStatus(status_raw) if isinstance(status_raw, str) else status_raw
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            title=data["title"],
            description=data.get("description", ""),
            business_value=float(data.get("business_value", 1.0)),
            time_criticality=float(data.get("time_criticality", 1.0)),
            risk_reduction=float(data.get("risk_reduction", 1.0)),
            job_size=float(data.get("job_size", 1.0)),
            status=status,
            retries=int(data.get("retries", 0)),
            max_retries=int(data.get("max_retries", 2)),
        )

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id!r}, title={self.title!r}, "
            f"wsjf={self.wsjf_score:.2f}, status={self.status.value})"
        )
