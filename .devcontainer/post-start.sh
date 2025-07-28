#!/bin/bash

set -e

echo "🚀 Running post-start setup for Agentic Development Orchestrator..."

# Check if git safe directory is configured
echo "🔧 Ensuring git safe directory configuration..."
git config --global --add safe.directory /workspaces/agentic-dev-orchestrator 2>/dev/null || true

# Ensure environment is activated and dependencies are available
echo "🐍 Verifying Python environment..."
if command -v python &> /dev/null; then
    python --version
else
    echo "⚠️  Python not found in PATH"
fi

# Check if ADO CLI is working
echo "🔍 Verifying ADO installation..."
if python -c "import ado" 2>/dev/null; then
    echo "✅ ADO module is importable"
else
    echo "⚠️  ADO module not found, attempting reinstall..."
    pip install -e . --quiet
fi

# Start background services if needed
echo "🔄 Starting background services..."

# Start health check endpoint if configured
if [ "$ENABLE_HEALTH_CHECK" = "true" ]; then
    echo "🏥 Starting health check service on port ${HEALTH_CHECK_PORT:-8080}..."
    # Note: This would start a simple health check server
    # python -m ado.monitoring.health_check &
fi

# Start metrics collection if configured
if [ "$ENABLE_METRICS" = "true" ]; then
    echo "📊 Starting metrics collection on port ${METRICS_PORT:-9090}..."
    # Note: This would start a metrics endpoint
    # python -m ado.monitoring.metrics &
fi

# Check for required environment variables
echo "⚙️  Checking environment configuration..."
REQUIRED_VARS=("GITHUB_TOKEN" "OPENAI_API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "⚠️  Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "    - $var"
    done
    echo ""
    echo "💡 Please configure these in your .env file"
    echo "   Copy .env.example to .env and update the values"
else
    echo "✅ All required environment variables are set"
fi

# Update pre-commit hooks if needed
echo "🔄 Updating pre-commit hooks..."
pre-commit autoupdate --quiet || true

# Display system status
echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📈 System Status:"
echo "   Python: $(python --version 2>&1 | sed 's/Python //')"
echo "   Git: $(git --version | sed 's/git version //')"
echo "   Node.js: $(node --version 2>/dev/null || echo 'Not installed')"
echo "   Docker: $(docker --version 2>/dev/null | sed 's/Docker version //' | cut -d',' -f1 || echo 'Not available')"
echo ""

# Show quick help
echo "🚀 Quick Start:"
echo "   1. Configure your .env file with API keys"
echo "   2. Run: ado init"
echo "   3. Run: ado status"
echo "   4. Run: ado run"
echo ""
echo "📚 Documentation: https://github.com/terragon-labs/agentic-dev-orchestrator"
echo "🐛 Issues: https://github.com/terragon-labs/agentic-dev-orchestrator/issues"
echo ""