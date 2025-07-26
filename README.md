# agentic-dev-orchestrator

<!-- IMPORTANT: Replace 'your-github-username-or-org' with your actual GitHub details -->
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-github-username-or-org/agentic-dev-orchestrator/ci.yml?branch=main)](https://github.com/your-github-username-or-org/agentic-dev-orchestrator/actions)
[![Coverage Status](https://img.shields.io/coveralls/github/your-github-username-or-org/agentic-dev-orchestrator)](https://coveralls.io/github/your-github-username-or-org/agentic-dev-orchestrator)
[![License](https://img.shields.io/github/license/your-github-username-or-org/agentic-dev-orchestrator)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://semver.org)

A CLI and GitHub Action that unifies multi-agent development workflows, integrating frameworks like AutoGen and CrewAI. It streamlines the coding process by automatically ranking tasks from a backlog using the Weighted Shortest Job First (WSJF) method and executing them through a sequence of specialized AI agents.

## ✨ Key Features

*   **WSJF-Ranked Backlog**: Ingests `backlog/*.json` files, prioritizing tasks using the full SAFe WSJF formula: `(User-Business Value + Time Criticality + Risk Reduction & Opportunity Enablement) / Job Size`.
*   **Multi-Agent Execution Graph**: Implements a pipeline of AI agents—Planner, Coder, Reviewer, and Merger—using AutoGen's communication channels.
*   **Safety & Escalation Hooks**: Includes a policy gate, unit-test coverage checks, and a human-in-the-loop pattern for resolving edge cases. This pattern is implemented via AutoGen's `UserProxyAgent`.

## 🏗️ Architecture

```mermaid
graph TD
    A[Backlog.json] -->|Select top WSJF| P(Planner Agent)
    P --> C(Coder Agent)
    C --> R(Reviewer Agent)
    R -- Fails --> H{Human-in-the-Loop via UserProxyAgent}
    R -- Passes --> G(GitHub PR)
Use code with caution.
Markdown
⚡ Quick Start
Install the orchestrator: pip install agentic-dev-orchestrator
Initialize the project: ado init
Set your GitHub token: export GITHUB_TOKEN='your_personal_access_token'
Run the orchestrator: ado run
🛠️ Configuration
Backlog Schema backlog/issue-123.json
Generated json
{
  "title": "Implement user authentication endpoint",
  "wsjf": {
    "user_business_value": 8,
    "time_criticality": 8,
    "risk_reduction_opportunity_enablement": 5,
    "job_size": 5
  },
  "description": "Create a new FastAPI endpoint at /auth/login."
}
Use code with caution.
Json
Environment Variables
Variable	Description
GITHUB_TOKEN	A GitHub Personal Access Token for creating PRs.
OPENAI_API_KEY	API key for the underlying LLM used by the agents.
📈 Roadmap
v0.1.0: Support for single-repository projects.
v0.2.0 (Monorepo Support): The orchestrator will discover sub-projects by looking for ado.yml configuration files in subdirectories.
v1.0.0: SaaS dashboard for managing workflows.
🤝 Contributing
We welcome contributions! Please see our organization-wide CONTRIBUTING.md for guidelines and our CODE_OF_CONDUCT.md. A CHANGELOG.md is maintained for version history.
See Also
observer-coordinator-insights: Uses this orchestration layer for HR analytics.
📝 License
This project is licensed under the Apache-2.0 License.
📚 References
AutoGen Human-in-the-Loop: AutoGen UserProxyAgent Docs
SAFe WSJF: Scaled Agile Framework Documentation
