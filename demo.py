#!/usr/bin/env python3
"""Demo: 10-task backlog, 3 workers, full WSJF-ordered orchestration cycle."""

import logging
import json
from orchestrator import (
    Task,
    TaskBacklog,
    WSJFPrioritizer,
    AgentWorker,
    Orchestrator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)

# ------------------------------------------------------------------
# 1. Build a 10-task backlog (mix of priorities)
# ------------------------------------------------------------------

TASKS_JSON = json.dumps([
    {
        "title": "Fix critical auth bug",
        "description": "Users are getting 401s on valid tokens after password reset.",
        "business_value": 13, "time_criticality": 20, "risk_reduction": 13, "job_size": 3,
    },
    {
        "title": "Add OAuth2 login",
        "description": "Support Google and GitHub OAuth login flows.",
        "business_value": 8, "time_criticality": 5, "risk_reduction": 3, "job_size": 8,
    },
    {
        "title": "Database connection pooling",
        "description": "Replace per-request DB connections with a connection pool.",
        "business_value": 8, "time_criticality": 8, "risk_reduction": 8, "job_size": 5,
    },
    {
        "title": "Write onboarding docs",
        "description": "Create developer onboarding guide for new hires.",
        "business_value": 3, "time_criticality": 2, "risk_reduction": 1, "job_size": 2,
    },
    {
        "title": "Upgrade deprecated dependencies",
        "description": "Several packages are 2 major versions behind with known CVEs.",
        "business_value": 5, "time_criticality": 8, "risk_reduction": 13, "job_size": 3,
    },
    {
        "title": "Add rate limiting to API",
        "description": "Prevent abuse with per-IP rate limiting middleware.",
        "business_value": 8, "time_criticality": 8, "risk_reduction": 8, "job_size": 5,
    },
    {
        "title": "Migrate to structured logging",
        "description": "Replace print statements with structured JSON logs.",
        "business_value": 3, "time_criticality": 2, "risk_reduction": 3, "job_size": 2,
    },
    {
        "title": "Build CI/CD pipeline",
        "description": "Automate test, lint, build, and deploy with GitHub Actions.",
        "business_value": 8, "time_criticality": 5, "risk_reduction": 5, "job_size": 5,
    },
    {
        "title": "Add dark mode to UI",
        "description": "Users have requested a dark mode toggle.",
        "business_value": 3, "time_criticality": 1, "risk_reduction": 1, "job_size": 3,
    },
    {
        "title": "Implement task retry logic",
        "description": "Worker tasks should auto-retry on transient failures.",
        "business_value": 5, "time_criticality": 5, "risk_reduction": 8, "job_size": 3,
    },
])

backlog = TaskBacklog()
backlog.import_json(TASKS_JSON)

print(f"\n{'='*60}")
print(f"BACKLOG LOADED: {len(backlog)} tasks")
print(f"{'='*60}")

# ------------------------------------------------------------------
# 2. Show WSJF ranking before execution
# ------------------------------------------------------------------

prioritizer = WSJFPrioritizer()
print("\nWSJF RANKING (before execution):")
print("  BV=Business Value, TC=Time Criticality, RR=Risk Reduction, JS=Job Size")
print()
prioritizer.print_ranked(backlog.all())

# ------------------------------------------------------------------
# 3. Create 3 workers with different success rates
# ------------------------------------------------------------------

workers = [
    AgentWorker("alpha",   success_rate=0.9, rng_seed=42),
    AgentWorker("beta",    success_rate=0.7, rng_seed=7),
    AgentWorker("gamma",   success_rate=0.8, rng_seed=13),
]

# ------------------------------------------------------------------
# 4. Run the orchestrator to completion
# ------------------------------------------------------------------

orch = Orchestrator(backlog, workers, tasks_per_cycle=3)

print(f"\n{'='*60}")
print("ORCHESTRATION STARTING")
print(f"{'='*60}\n")

orch.run_until_done(max_cycles=20)

# ------------------------------------------------------------------
# 5. Print results
# ------------------------------------------------------------------

orch.print_summary()

print("\nFINAL TASK STATES:")
for task in sorted(backlog.all(), key=lambda t: t.wsjf_score, reverse=True):
    icon = {"completed": "✓", "failed": "✗", "pending": "?", "retrying": "↻", "in_progress": "…"}.get(
        task.status.value, "?"
    )
    retries = f" (retried {task.retries}x)" if task.retries else ""
    print(f"  {icon} [{task.wsjf_score:5.2f}] {task.title}{retries}")
