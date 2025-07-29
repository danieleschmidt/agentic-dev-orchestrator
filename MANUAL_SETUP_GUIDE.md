# Manual Setup Guide for SDLC Enhancement

## Overview

This guide provides step-by-step instructions to implement the advanced SDLC enhancements for the Agentic Development Orchestrator repository. Since GitHub Actions workflows require special permissions, these must be created manually.

## GitHub Actions Workflows Setup

### Step 1: Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### Step 2: Create CI/CD Pipeline (`ci.yml`)

Create `.github/workflows/ci.yml` with the following content:

```yaml
# CI/CD Pipeline for Agentic Development Orchestrator
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 6 * * *'
  workflow_dispatch:

env:
  PYTHON_VERSION: '3.11'

jobs:
  test-matrix:
    name: Test Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip setuptools wheel
        pip install -e ".[dev]"
    - name: Run tests
      run: pytest tests/ -v --cov=. --cov-report=xml
    - name: Upload coverage
      if: matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3

  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run linting
      run: |
        ruff check . --output-format=github
        black --check --diff .
        isort --check-only --diff .
    - name: Run type checking
      run: mypy . --ignore-missing-imports

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit[toml] safety pip-audit
        pip install -e ".[dev]"
    - name: Run Bandit security scan
      run: bandit -r . -f sarif -o bandit-results.sarif || true
    - name: Run Safety vulnerability scan
      run: safety check --json --output safety-report.json || true
    - name: Run pip-audit
      run: pip-audit --format=json --output=pip-audit-report.json || true
    - name: Upload SARIF results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: bandit-results.sarif

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [quality, security]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Check package
      run: twine check dist/*
```

### Step 3: Create Release Pipeline (`release.yml`)

Create `.github/workflows/release.yml`:

```yaml
name: Release Pipeline

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: write
  packages: write

jobs:
  release:
    name: Semantic Release
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        token: ${{ secrets.GITHUB_TOKEN }}
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        cache: 'npm'
    - name: Install semantic-release dependencies
      run: npm ci
    - name: Run semantic-release
      run: npx semantic-release
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        PYPI_API_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
```

### Step 4: Create Security Workflow (`security.yml`)

Create `.github/workflows/security.yml`:

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 2 * * *'
  push:
    branches: [ main ]
    paths:
      - 'requirements*.txt'
      - 'pyproject.toml'
      - 'Dockerfile'
  workflow_dispatch:

permissions:
  contents: read
  security-events: write

jobs:
  security-scan:
    name: Advanced Security Scan
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit[toml] semgrep safety pip-audit cyclonedx-bom
        pip install -e ".[dev]"
    - name: Run comprehensive security scan
      run: |
        bandit -r . -f sarif -o bandit-results.sarif || true
        semgrep --config=auto --sarif --output=semgrep-results.sarif . || true
        safety check --json --output safety-report.json || true
        pip-audit --format=sarif --output=pip-audit-results.sarif || true
        cyclonedx-py -o ado-sbom.json
    - name: Upload SARIF results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: |
          bandit-results.sarif
          semgrep-results.sarif
          pip-audit-results.sarif
```

## Repository Configuration

### Step 5: Configure Repository Secrets

Add these secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

```
PYPI_API_TOKEN          # For PyPI package publishing
CODECOV_TOKEN          # For coverage reporting (optional)
```

### Step 6: Enable Branch Protection

Configure branch protection for `main` branch (Settings → Branches):
- ✅ Require a pull request before merging
- ✅ Require approvals (minimum 1)
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Include administrators

### Step 7: Enable Security Features

In repository Settings → Security:
- ✅ Enable Dependabot alerts
- ✅ Enable Dependabot security updates
- ✅ Enable Secret scanning
- ✅ Enable Push protection for secrets

## Dependabot Configuration

### Step 8: Create Dependabot Config

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "deps"
      include: "scope"
    labels:
      - "dependencies"
      - "automated"
    allow:
      - dependency-type: "all"
        update-type: "security"
      - dependency-type: "direct:production"
        update-type: "version-update:semver-minor"
      - dependency-type: "direct:production"
        update-type: "version-update:semver-patch"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "github-actions"
      - "ci"
      - "automated"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "docker"
      - "infrastructure"
      - "automated"
```

## Local Development Setup

### Step 9: Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run hooks on all files (optional)
pre-commit run --all-files
```

### Step 10: Set up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run health check
make health-check

# Run tests
make test

# Run quality checks
make quality
```

## Monitoring Setup (Optional)

### Step 11: Deploy Observability Stack

```bash
# Start monitoring stack
docker-compose -f docker-compose.observability.yml --profile development up -d

# Access Grafana (admin/admin_change_me)
open http://localhost:3000

# Access Prometheus
open http://localhost:9090

# Access Jaeger
open http://localhost:16686
```

## Verification Steps

### Step 12: Verify Setup

1. **Push a commit** to trigger CI/CD pipeline
2. **Check Actions tab** for workflow execution
3. **Review security alerts** in Security tab
4. **Verify branch protection** by creating a PR

### Step 13: Monitor and Optimize

1. **Review workflow results** and optimize as needed
2. **Monitor Dependabot PRs** and merge approved updates
3. **Check security scan results** regularly
4. **Update documentation** based on learnings

## Troubleshooting

### Common Issues

1. **Workflow permission errors**: Ensure repository has Actions enabled
2. **Secret access issues**: Verify secrets are properly configured
3. **Branch protection bypass**: Check that rules apply to administrators
4. **Pre-commit failures**: Run `pre-commit autoupdate` and retry

### Getting Help

- **Documentation**: Review workflow logs in Actions tab
- **Community**: GitHub Discussions for questions
- **Issues**: Create issues for bugs or feature requests

## Success Criteria

✅ **All workflows run successfully**  
✅ **Security scans complete without critical issues**  
✅ **Dependabot creates update PRs**  
✅ **Branch protection prevents direct pushes**  
✅ **Pre-commit hooks enforce code quality**  

## Next Steps

After completing this setup:

1. **Team Training**: Ensure team understands new workflows
2. **Process Documentation**: Update team procedures
3. **Monitoring**: Set up monitoring and alerting
4. **Continuous Improvement**: Regularly review and optimize

This manual setup provides the same advanced SDLC capabilities as the automated enhancement, ensuring your repository achieves enterprise-grade security, automation, and operational excellence.