# Getting Started with Agentic Dev Orchestrator

Welcome to the Agentic Dev Orchestrator (ADO)! This guide will help you get up and running quickly.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Git** installed and configured
- **GitHub account** with repository access
- **OpenAI API key** or other LLM provider credentials
- **GitHub Personal Access Token** with appropriate permissions

## 🚀 Quick Start (5 minutes)

### 1. Installation

#### Option A: Install from PyPI (Recommended)
```bash
pip install agentic-dev-orchestrator
```

#### Option B: Install from Source
```bash
git clone https://github.com/terragon-labs/agentic-dev-orchestrator.git
cd agentic-dev-orchestrator
pip install -e .
```

### 2. Initial Setup

```bash
# Initialize ADO in your project
ado init

# This creates:
# - .ado/ directory with configuration
# - backlog/ directory for task definitions
# - Sample backlog items
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# At minimum, set:
# - GITHUB_TOKEN
# - OPENAI_API_KEY (or other LLM provider)
# - GITHUB_REPO (format: owner/repo)
```

### 4. Create Your First Task

Create a file `backlog/my-first-task.json`:

```json
{
  "title": "Add a health check endpoint",
  "description": "Create a simple /health endpoint that returns service status",
  "wsjf": {
    "user_business_value": 7,
    "time_criticality": 5,
    "risk_reduction_opportunity_enablement": 6,
    "job_size": 3
  },
  "acceptance_criteria": [
    "Endpoint responds with 200 OK",
    "Returns JSON with service status",
    "Includes basic system information"
  ],
  "labels": ["enhancement", "api"]
}
```

### 5. Run ADO

```bash
# Process the backlog and execute the highest priority task
ado run

# Or run with specific options
ado run --max-items 1 --dry-run
```

## 📖 Understanding ADO

### How It Works

1. **Prioritization**: ADO reads all `.json` files in the `backlog/` directory and calculates WSJF scores
2. **Agent Pipeline**: The highest priority task flows through specialized AI agents:
   - **Planner**: Analyzes requirements and creates implementation plan
   - **Coder**: Implements the solution following the plan
   - **Reviewer**: Reviews code quality, tests, and adherence to requirements
   - **Merger**: Creates pull request and handles integration

3. **Safety Gates**: Built-in safeguards ensure code quality:
   - Test coverage requirements
   - Security scanning
   - Human escalation for complex issues
   - Policy compliance checks

### WSJF Prioritization

Weighted Shortest Job First (WSJF) formula:
```
WSJF = (User-Business Value + Time Criticality + Risk Reduction) / Job Size
```

- **Higher values** = higher priority
- **User-Business Value** (1-10): Direct value to users/business
- **Time Criticality** (1-10): Urgency and time sensitivity
- **Risk Reduction** (1-10): Risk mitigation and opportunity enablement
- **Job Size** (1-10): Estimated effort/complexity (lower = less effort)

## 🛠️ Configuration

### Project Structure

After initialization, your project will have:

```
your-project/
├── .ado/
│   ├── config.yml          # ADO configuration
│   ├── agents.yml          # Agent configurations
│   └── policies.yml        # Safety policies
├── backlog/
│   ├── sample-task.json    # Example task
│   └── your-tasks.json     # Your task definitions
├── .env                    # Environment variables
└── .gitignore             # Updated with ADO patterns
```

### Environment Configuration

Key environment variables:

```bash
# Required
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=sk-your_key_here
GITHUB_REPO=your-username/your-repo

# Optional but recommended
ADO_LOG_LEVEL=INFO
AGENT_TIMEOUT=300
MIN_TEST_COVERAGE=0.8
HUMAN_REVIEW_REQUIRED=false
```

### Agent Configuration

Customize agent behavior in `.ado/agents.yml`:

```yaml
planner:
  model: gpt-4
  temperature: 0.2
  max_tokens: 2000
  system_prompt: "You are a senior software architect..."

coder:
  model: gpt-4
  temperature: 0.1
  max_tokens: 4000
  system_prompt: "You are an expert software developer..."

reviewer:
  model: gpt-4
  temperature: 0.0
  max_tokens: 3000
  system_prompt: "You are a thorough code reviewer..."
```

## 📝 Writing Effective Backlog Items

### Best Practices

1. **Clear Titles**: Use action-oriented, specific titles
   - ✅ "Add user authentication to API endpoints"
   - ❌ "User stuff"

2. **Detailed Descriptions**: Provide context and requirements
   ```json
   {
     "description": "Implement JWT-based authentication for all API endpoints. Users should be able to login with email/password and receive a token for subsequent requests."
   }
   ```

3. **Acceptance Criteria**: Define success clearly
   ```json
   {
     "acceptance_criteria": [
       "POST /auth/login accepts email and password",
       "Returns JWT token on successful authentication",
       "Protected endpoints reject requests without valid token",
       "Token expiration is configurable"
     ]
   }
   ```

4. **Accurate WSJF Scoring**: Be honest and consistent
   - Consider actual business impact
   - Estimate effort realistically
   - Account for dependencies and risks

### Task Templates

#### Feature Implementation
```json
{
  "title": "Implement [feature name]",
  "description": "Detailed description of the feature and its purpose",
  "wsjf": {
    "user_business_value": 8,
    "time_criticality": 6,
    "risk_reduction_opportunity_enablement": 5,
    "job_size": 7
  },
  "acceptance_criteria": [
    "Specific, testable requirement 1",
    "Specific, testable requirement 2"
  ],
  "technical_notes": "Implementation hints or constraints",
  "labels": ["feature", "backend"]
}
```

#### Bug Fix
```json
{
  "title": "Fix [bug description]",
  "description": "Steps to reproduce and expected vs actual behavior",
  "wsjf": {
    "user_business_value": 6,
    "time_criticality": 9,
    "risk_reduction_opportunity_enablement": 8,
    "job_size": 3
  },
  "reproduction_steps": [
    "Step 1",
    "Step 2",
    "Observe error"
  ],
  "labels": ["bug", "priority-high"]
}
```

## 🔧 Advanced Usage

### Custom Agents

Create custom agents for specialized workflows:

```python
# custom_agents/security_agent.py
from ado.agents.base import BaseAgent

class SecurityAgent(BaseAgent):
    def execute(self, task, context):
        # Custom security review logic
        pass
```

### Policy Customization

Define custom policies in `.ado/policies.yml`:

```yaml
security_policies:
  - name: "no_hardcoded_secrets"
    enabled: true
    severity: "high"
  - name: "dependency_scan"
    enabled: true
    max_vulnerabilities: 0

quality_policies:
  - name: "test_coverage"
    threshold: 0.85
  - name: "code_complexity"
    max_complexity: 10
```

### Integration with CI/CD

Add ADO to your GitHub Actions:

```yaml
# .github/workflows/ado.yml
name: ADO Workflow
on:
  schedule:
    - cron: '0 9 * * 1-5'  # Weekdays at 9 AM

jobs:
  ado:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run ADO
        uses: terragon-labs/ado-action@v1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          max-items: 3
```

## 🚨 Troubleshooting

### Common Issues

#### Authentication Errors
```bash
Error: GitHub API authentication failed
```

**Solution**: Check your `GITHUB_TOKEN` permissions:
- Needs `repo` access for private repositories
- Needs `workflow` access for GitHub Actions
- Must not be expired

#### LLM API Errors
```bash
Error: OpenAI API request failed
```

**Solutions**:
1. Verify `OPENAI_API_KEY` is correct
2. Check API quota and billing
3. Try reducing `max_tokens` in agent config
4. Enable retry logic: `AGENT_MAX_RETRIES=5`

#### No Tasks Processed
```bash
Info: No backlog items found
```

**Solution**: Ensure JSON files are in `backlog/` directory with valid format

### Debug Mode

Enable detailed logging:

```bash
# Environment variable
ADO_LOG_LEVEL=DEBUG

# Command line
ado run --log-level DEBUG
```

### Getting Help

1. Check the [troubleshooting guide](../troubleshooting/)
2. Search [existing issues](https://github.com/terragon-labs/agentic-dev-orchestrator/issues)
3. Create a [new issue](https://github.com/terragon-labs/agentic-dev-orchestrator/issues/new)
4. Join our [community discussions](https://github.com/terragon-labs/agentic-dev-orchestrator/discussions)

## 🎯 Next Steps

Now that you're up and running:

1. **Create more backlog items** for your project
2. **Customize agent configurations** for your workflow
3. **Set up monitoring** to track ADO performance
4. **Integrate with CI/CD** for automated execution
5. **Explore advanced features** like custom policies

Happy orchestrating! 🚀

---

**Need help?** Check our [documentation](../), [examples](https://github.com/terragon-labs/ado-examples), or [community](https://github.com/terragon-labs/agentic-dev-orchestrator/discussions).