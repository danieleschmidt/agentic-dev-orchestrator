#!/bin/bash
# Post-start script run each time the container starts

set -e

echo "🔄 Running post-start setup..."

# Ensure proper permissions
sudo chown -R vscode:vscode /workspaces/agentic-dev-orchestrator/.ado 2>/dev/null || true

# Update Python path
export PYTHONPATH="/workspaces/agentic-dev-orchestrator:$PYTHONPATH"

# Start any background services if needed
# (Currently none, but placeholder for future)

# Check system health
echo "🔍 Performing system health checks..."

# Check Python installation
python --version

# Check key dependencies
echo "📋 Checking dependencies..."
python -c "import yaml; print('✅ PyYAML available')" 2>/dev/null || echo "❌ PyYAML not available"
python -c "import pytest; print('✅ pytest available')" 2>/dev/null || echo "❌ pytest not available"

# Check ADO CLI
echo "🧪 Testing ADO CLI..."
python ado.py --help > /dev/null 2>&1 && echo "✅ ADO CLI working" || echo "❌ ADO CLI not working"

# Show current status
if [ -f "backlog.yml" ]; then
    echo "📊 Current backlog status:"
    python ado.py status 2>/dev/null || echo "ℹ️  Run 'ado init' to initialize"
else
    echo "ℹ️  No backlog.yml found. Run 'ado init' to get started."
fi

# Display helpful information
echo ""
echo "🎯 Development environment ready!"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "📦 Packages installed: $(pip list | wc -l) packages"
echo ""
echo "💡 Quick start:"
echo "  1. Configure your .env file with API keys"
echo "  2. Run 'ado init' to initialize the workspace"
echo "  3. Run 'ado status' to check the backlog"
echo "  4. Add items to backlog/ and run 'ado run'"
echo ""