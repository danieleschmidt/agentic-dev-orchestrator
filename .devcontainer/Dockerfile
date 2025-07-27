# Development container for Agentic Development Orchestrator
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        vim \
        nano \
        jq \
        tree \
        htop \
        httpie \
        sqlite3 \
        postgresql-client \
        redis-tools \
        shellcheck \
        yamllint \
        graphviz \
        pandoc \
        ripgrep \
        fd-find \
        bat \
        exa \
        zoxide \
        fzf \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install additional Python tools
RUN pip install --upgrade pip setuptools wheel

# Install development tools globally
RUN pip install \
    pre-commit \
    black \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    tox \
    coverage \
    bandit \
    safety \
    pip-audit \
    sphinx \
    mkdocs \
    mkdocs-material \
    jupyter \
    ipykernel \
    jupyterlab

# Configure shell
RUN echo 'alias ll="exa -la"' >> /home/vscode/.bashrc \
    && echo 'alias la="exa -la"' >> /home/vscode/.bashrc \
    && echo 'alias cat="bat"' >> /home/vscode/.bashrc \
    && echo 'alias find="fd"' >> /home/vscode/.bashrc

# Set up pre-commit cache directory
RUN mkdir -p /home/vscode/.cache/pre-commit && chown -R vscode:vscode /home/vscode/.cache

# Create workspace directory
WORKDIR /workspaces/agentic-dev-orchestrator

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt* ./
RUN if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
RUN pip install -r requirements.txt

# Set ownership
RUN chown -R vscode:vscode /workspaces/agentic-dev-orchestrator