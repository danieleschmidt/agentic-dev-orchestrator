# ADO Quick Start Guide

Get up and running with the Agentic Development Orchestrator in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- Git configured with your GitHub credentials
- GitHub Personal Access Token
- OpenAI API Key (or compatible LLM service)

## Installation

Choose your preferred installation method:

### Option 1: PyPI (Recommended)
```bash
pip install agentic-dev-orchestrator
```

### Option 2: Docker
```bash
# Pull the image
docker pull ghcr.io/danieleschmidt/agentic-dev-orchestrator:latest

# Create an alias for easier use
alias ado='docker run --rm -v $(pwd):/workspace -w /workspace ghcr.io/danieleschmidt/agentic-dev-orchestrator:latest ado'
```

### Option 3: From Source
```bash
git clone https://github.com/danieleschmidt/agentic-dev-orchestrator.git
cd agentic-dev-orchestrator
pip install -e .
```

## Setup

### 1. Initialize ADO in your project
```bash
cd your-project
ado init
```

This creates:
- `backlog.yml` - Main backlog configuration
- `backlog/` - Directory for individual backlog items
- `docs/status/` - Execution reports and metrics
- `escalations/` - Human intervention logs

### 2. Configure Environment Variables

Create a `.env` file or export these variables:

```bash
# GitHub integration (required)
export GITHUB_TOKEN="ghp_your_github_personal_access_token"

# LLM service (required)
export OPENAI_API_KEY="sk-your_openai_api_key"

# Optional: Custom configuration
export ADO_CONFIG_PATH="/path/to/your/ado/config"
export ADO_LOG_LEVEL="INFO"
```

### 3. Add Your First Backlog Item

Create `backlog/my-first-task.json`:

```json
{
  "title": "Add user authentication endpoint",
  "type": "feature",
  "description": "Create a new FastAPI endpoint at /auth/login that validates user credentials",
  "acceptance_criteria": [
    "Endpoint responds to POST /auth/login",
    "Validates username and password",
    "Returns JWT token on success",
    "Returns 401 on invalid credentials",
    "Includes proper error handling"
  ],
  "wsjf": {
    "user_business_value": 8,
    "time_criticality": 6,
    "risk_reduction_opportunity_enablement": 5,
    "job_size": 5
  },
  "effort": 5,
  "status": "READY",
  "risk_tier": "medium"
}
```

## Usage

### Check Status
```bash
ado status
```

This shows:
- Total backlog items
- Items by status (NEW, READY, DOING, etc.)
- Top 3 items prioritized by WSJF score

### Run the Orchestrator
```bash
ado run
```

This executes the autonomous workflow:
1. **Planner Agent** - Analyzes the highest WSJF task and creates implementation plan
2. **Coder Agent** - Implements the code changes
3. **Reviewer Agent** - Reviews code quality and runs tests
4. **Merger Agent** - Creates GitHub PR (or escalates to human if issues found)

### Discover New Tasks
```bash
ado discover
```

Automatically scans your codebase for potential tasks like:
- TODO comments
- FIXME notes
- Missing tests
- Documentation gaps

## Understanding WSJF Prioritization

ADO uses Weighted Shortest Job First (WSJF) from SAFe methodology:

```
WSJF = (User-Business Value + Time Criticality + Risk Reduction & Opportunity Enablement) / Job Size
```

- **User-Business Value** (1-10): Impact on users and business
- **Time Criticality** (1-10): How time-sensitive is this work
- **Risk Reduction & Opportunity Enablement** (1-10): Risk mitigation and future opportunities
- **Job Size** (1-13): Effort required (Fibonacci scale)

Higher WSJF scores get prioritized first.

## Safety Features

ADO includes several safety mechanisms:

- **Policy Gates**: Prevents dangerous operations
- **Test Coverage**: Requires tests before merging
- **Security Scanning**: Automatic security checks
- **Human-in-the-Loop**: Escalates complex decisions
- **Rollback**: Can undo changes if issues detected

## Troubleshooting

### Common Issues

**"No GitHub token found"**
```bash
# Check your token is set
echo $GITHUB_TOKEN

# Or create a .env file
echo "GITHUB_TOKEN=your_token_here" > .env
```

**"OpenAI API key not configured"**
```bash
# Set your OpenAI key
export OPENAI_API_KEY="sk-your_key_here"
```

**"No backlog items found"**
```bash
# Check your backlog directory
ls backlog/

# Add a sample task
ado init  # Creates sample files
```

### Getting Help

```bash
# Show available commands
ado help

# Check logs
cat docs/status/last_execution.json

# View escalations (human interventions needed)
ls escalations/
```

## Next Steps

1. **Read the Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design
2. **Customize Configuration**: Check [docs/](docs/) for advanced configuration options
3. **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
4. **Security**: Review [SECURITY.md](SECURITY.md) for security considerations

## Need Help?

- 📖 **Documentation**: [Full documentation](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/danieleschmidt/agentic-dev-orchestrator/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/danieleschmidt/agentic-dev-orchestrator/discussions)

---

**Tip**: Start with small, well-defined tasks to get familiar with the system before tackling larger features.