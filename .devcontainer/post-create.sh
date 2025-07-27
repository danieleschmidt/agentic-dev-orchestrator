#!/bin/bash
# Post-create script for devcontainer setup

set -e

echo "🚀 Running post-create setup for Agentic Development Orchestrator..."

# Install Python dependencies in development mode
echo "📦 Installing Python package in development mode..."
pip install -e .

# Install additional development dependencies if they exist
if [ -f "requirements-dev.txt" ]; then
    echo "📦 Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p docs/status
mkdir -p escalations
mkdir -p .ado/cache
mkdir -p .ado/locks
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/e2e
mkdir -p coverage

# Set up git configuration if not already set
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up git configuration..."
    git config --global user.name "ADO Developer"
    git config --global user.email "developer@ado.local"
fi

# Initialize pre-commit cache
echo "🔄 Initializing pre-commit cache..."
pre-commit run --all-files || true

# Set up shell aliases and functions
echo "🐚 Setting up shell environment..."
cat >> ~/.bashrc << 'EOF'

# ADO-specific aliases and functions
alias ado="python ado.py"
alias ado-test="pytest -v"
alias ado-coverage="pytest --cov=. --cov-report=html"
alias ado-lint="ruff check . && mypy ."
alias ado-format="black . && ruff check --fix ."

# Development helpers
function ado-reset-backlog() {
    echo "Resetting backlog to default state..."
    git checkout -- backlog.yml
    rm -f backlog/*.json
    ado init
}

function ado-clean() {
    echo "Cleaning development artifacts..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    rm -rf .pytest_cache .coverage htmlcov .mypy_cache .tox build/ dist/ *.egg-info/
}

function ado-logs() {
    echo "Recent ADO logs:"
    find docs/status -name "*.json" -type f -print0 | xargs -0 ls -t | head -5
}

EOF

# Create a simple development config
echo "⚙️  Creating development configuration..."
cat > .env.example << 'EOF'
# Agentic Development Orchestrator Environment Variables

# GitHub Integration
GITHUB_TOKEN=your_github_personal_access_token_here
GITHUB_REPO=your-username/your-repo

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# ADO Configuration
ADO_LOG_LEVEL=DEBUG
ADO_CONFIG_PATH=.ado/config.json
ADO_WORKSPACE_PATH=/workspaces/agentic-dev-orchestrator

# Agent Configuration
AGENT_TIMEOUT=300
AGENT_MAX_RETRIES=3
AGENT_CONCURRENCY=2

# Safety and Security
ESCALATION_TIMEOUT=3600
HUMAN_REVIEW_REQUIRED=true
SECURITY_SCAN_REQUIRED=true

# Development Settings
PYTHON_ENV=development
DEBUG=true
TESTING=false

# Optional Integrations
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
DATADOG_API_KEY=your_datadog_api_key_here
SENTRY_DSN=your_sentry_dsn_here
EOF

# Set up basic ADO structure
echo "🏗️  Initializing ADO structure..."
python ado.py init 2>/dev/null || echo "ADO init completed with warnings"

echo "✅ Post-create setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Copy .env.example to .env and configure your API keys"
echo "2. Run 'ado status' to check the system"
echo "3. Add backlog items and run 'ado run' to test"
echo ""
echo "📚 Available commands:"
echo "  ado init      - Initialize ADO structure"
echo "  ado status    - Show backlog status"
echo "  ado run       - Execute autonomous processing"
echo "  ado discover  - Discover new backlog items"
echo ""
echo "🛠️  Development commands:"
echo "  ado-test      - Run tests"
echo "  ado-coverage  - Run tests with coverage"
echo "  ado-lint      - Run linting"
echo "  ado-format    - Format code"
echo "  ado-clean     - Clean artifacts"