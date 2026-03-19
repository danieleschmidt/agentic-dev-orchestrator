# agentic-dev-orchestrator

A task orchestrator for agentic development workflows. Manages a backlog of coding tasks, prioritizes them using **WSJF (Weighted Shortest Job First)**, and dispatches them to agent workers.

Zero external dependencies — stdlib only.

---

## What is WSJF?

WSJF is a prioritization framework from SAFe (Scaled Agile Framework). It answers: *which task delivers the most value for the least effort?*

```
WSJF = Cost of Delay / Job Size

Cost of Delay = Business Value + Time Criticality + Risk Reduction
```

Each component is scored on a relative scale (1, 2, 3, 5, 8, 13, 20 — Fibonacci-ish):

| Component | Question |
|---|---|
| **Business Value** | How much value does this deliver to the customer/business? |
| **Time Criticality** | How much does delay cost us? Is there a deadline? |
| **Risk Reduction** | Does this reduce a technical, security, or business risk? |
| **Job Size** | How much effort does this take? (smaller = better denominator) |

**Higher score = dispatch first.** A high-value, urgent, low-effort task beats a large, low-urgency feature.

### Example

| Task | BV | TC | RR | JS | WSJF |
|---|---|---|---|---|---|
| Fix auth bug | 13 | 20 | 13 | 3 | **15.33** ← dispatch first |
| Add dark mode | 3 | 1 | 1 | 3 | 1.67 ← dispatch last |

---

## Architecture

```
orchestrator/
├── task.py          # Task dataclass + TaskStatus enum
├── prioritizer.py   # WSJFPrioritizer — ranks tasks by score
├── backlog.py       # TaskBacklog — CRUD + JSON import/export
├── worker.py        # AgentWorker — mock executor with configurable success rate
└── orchestrator.py  # Orchestrator — dispatch loop, retry handling
```

### Components

**`Task`** — dataclass holding WSJF attributes, status, retry count, and result.

**`WSJFPrioritizer`** — stateless ranker. Filters eligible tasks (PENDING/RETRYING), sorts by WSJF score descending.

**`TaskBacklog`** — dict-backed store with add/get/update/remove, status filters, and JSON bulk import/export.

**`AgentWorker`** — mock worker that simulates execution with a configurable success rate (0.0–1.0). Deterministic via `rng_seed`. Tracks completed/failed counts.

**`Orchestrator`** — drives the dispatch loop:
1. Rank all eligible tasks by WSJF
2. Dispatch top-N to workers (round-robin)
3. On failure: retry up to `task.max_retries`, then mark FAILED
4. Repeat until no eligible tasks remain or `max_cycles` hit

---

## Quick Start

```python
from orchestrator import Task, TaskBacklog, AgentWorker, Orchestrator

# Build a backlog
backlog = TaskBacklog()
backlog.import_json([
    {"title": "Fix auth bug",  "business_value": 13, "time_criticality": 20,
     "risk_reduction": 13, "job_size": 3},
    {"title": "Add dark mode", "business_value": 3,  "time_criticality": 1,
     "risk_reduction": 1,  "job_size": 3},
])

# Create workers
workers = [
    AgentWorker("alpha", success_rate=0.9),
    AgentWorker("beta",  success_rate=0.8),
]

# Run orchestration
orch = Orchestrator(backlog, workers)
orch.run_until_done()
orch.print_summary()
```

Run the included demo (10 tasks, 3 workers):

```bash
python demo.py
```

---

## Running Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

46 tests across task, prioritizer, backlog, worker, and orchestrator.

---

## Extending

To plug in a real worker (e.g. a Claude Code agent, subprocess, API call):

```python
from orchestrator.worker import AgentWorker, WorkResult
from orchestrator.task import Task, TaskStatus

class MyRealWorker(AgentWorker):
    def execute(self, task: Task) -> WorkResult:
        # Call your actual agent here
        try:
            output = my_agent.run(task.title, task.description)
            task.status = TaskStatus.COMPLETED
            task.result = output
            return WorkResult(task_id=task.id, success=True, output=output)
        except Exception as e:
            task.error = str(e)
            return WorkResult(task_id=task.id, success=False, error=str(e))
```

---

## License

MIT
