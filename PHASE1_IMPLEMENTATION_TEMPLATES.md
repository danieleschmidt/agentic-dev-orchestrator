# PHASE 1 IMPLEMENTATION TEMPLATES
## Ready-to-Deploy CI/CD Infrastructure

**Repository**: Agentic Development Orchestrator (ADO)  
**Phase**: Critical Infrastructure Gap Closure  
**Templates**: Production-ready GitHub Actions workflows  
**Compatibility**: Builds on existing pyproject.toml and pre-commit configuration  

---

## DEPLOYMENT CHECKLIST

### Prerequisites Verification
- [ ] Repository has existing `pyproject.toml` with tool configurations
- [ ] Pre-commit hooks are configured in `.pre-commit-config.yaml`
- [ ] Test directory structure exists (`tests/`)
- [ ] Security policy exists (`SECURITY.md`)

### Required Repository Secrets
```bash
# Essential secrets to configure in repository settings
PYPI_API_TOKEN=pypi-xxx...      # For package publishing
GITHUB_TOKEN=ghp_xxx...         # Auto-generated, verify permissions
```

### Optional Secrets for Enhanced Features
```bash
CODECOV_TOKEN=xxx...            # For coverage reporting
SLACK_WEBHOOK_URL=https://...   # For notifications
```

---

## TEMPLATE 1: COMPREHENSIVE CI PIPELINE

**File**: `/.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

env:
  PYTHON_DEFAULT_VERSION: "3.11"

jobs:
  # =============================================================================
  # QUALITY CHECKS JOB
  # =============================================================================
  quality-checks:
    name: Quality Checks
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better analysis
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run Black formatting check
        run: black --check --config pyproject.toml .
      
      - name: Run Ruff linting
        run: ruff check --config pyproject.toml .
      
      - name: Run MyPy type checking
        run: mypy --config-file pyproject.toml .
        continue-on-error: true  # Don't fail CI on type errors initially
      
      - name: Check import sorting
        run: isort --check --profile black .

  # =============================================================================
  # SECURITY SCANNING JOB
  # =============================================================================
  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      security-events: write
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[security]"
      
      - name: Run Bandit security scan
        run: |
          bandit -r . -f json -o bandit-report.json -x tests/
        continue-on-error: true
      
      - name: Run Safety dependency scan
        run: |
          safety check --json --output safety-report.json
        continue-on-error: true
      
      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-report.json
        continue-on-error: true
      
      - name: Upload security scan results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
            pip-audit-report.json
          retention-days: 30

  # =============================================================================
  # MULTI-PLATFORM TESTING JOB
  # =============================================================================
  test:
    name: Test Suite
    needs: [quality-checks]
    runs-on: ${{ matrix.os }}
    timeout-minutes: 30
    
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
        exclude:
          # Reduce matrix size for efficiency
          - os: windows-latest
            python-version: "3.8"
          - os: macos-latest
            python-version: "3.8"
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --tb=short --durations=10
      
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v --tb=short
        continue-on-error: ${{ matrix.python-version != env.PYTHON_DEFAULT_VERSION }}
      
      - name: Run E2E tests
        if: matrix.os == 'ubuntu-latest' && matrix.python-version == env.PYTHON_DEFAULT_VERSION
        run: |
          pytest tests/e2e/ -v --tb=short
        continue-on-error: true

  # =============================================================================
  # COVERAGE REPORTING JOB
  # =============================================================================
  coverage:
    name: Coverage Analysis
    needs: [test]
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run tests with coverage
        run: |
          pytest --cov=. --cov-report=xml --cov-report=html --cov-report=term
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        if: env.CODECOV_TOKEN != ''
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          file: ./coverage.xml
          fail_ci_if_error: false
        env:
          CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
      
      - name: Upload coverage artifacts
        uses: actions/upload-artifact@v4
        with:
          name: coverage-reports
          path: |
            coverage.xml
            htmlcov/
          retention-days: 30

  # =============================================================================
  # BUILD VERIFICATION JOB
  # =============================================================================
  build-verification:
    name: Build Verification
    needs: [quality-checks, security-scan]
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT_VERSION }}
          cache: 'pip'
      
      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Verify package
        run: |
          twine check dist/*
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist-packages
          path: dist/
          retention-days: 7

  # =============================================================================
  # STATUS REPORTING JOB
  # =============================================================================
  ci-status:
    name: CI Status
    if: always()
    needs: [quality-checks, security-scan, test, coverage, build-verification]
    runs-on: ubuntu-latest
    
    steps:
      - name: Check CI Status
        run: |
          echo "Quality Checks: ${{ needs.quality-checks.result }}"
          echo "Security Scan: ${{ needs.security-scan.result }}"
          echo "Tests: ${{ needs.test.result }}"
          echo "Coverage: ${{ needs.coverage.result }}"
          echo "Build: ${{ needs.build-verification.result }}"
          
          if [[ "${{ needs.quality-checks.result }}" != "success" || 
                "${{ needs.test.result }}" != "success" || 
                "${{ needs.build-verification.result }}" != "success" ]]; then
            echo "CI pipeline failed!"
            exit 1
          fi
          
          echo "CI pipeline completed successfully!"
```

---

## TEMPLATE 2: SECURITY AUTOMATION PIPELINE

**File**: `/.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  # =============================================================================
  # CODEQL ANALYSIS
  # =============================================================================
  codeql-analysis:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    timeout-minutes: 360
    
    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          queries: security-extended,security-and-quality
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v3
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:${{matrix.language}}"

  # =============================================================================
  # DEPENDENCY SCANNING
  # =============================================================================
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[security]"
      
      - name: Run Safety scan
        run: |
          safety check --json --output safety-scan.json
        continue-on-error: true
      
      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-scan.json
        continue-on-error: true
      
      - name: Upload scan results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: dependency-scan-results
          path: |
            safety-scan.json
            pip-audit-scan.json
          retention-days: 30

  # =============================================================================
  # CONTAINER SECURITY SCANNING
  # =============================================================================
  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 20
    if: github.event_name != 'schedule'  # Skip for scheduled runs
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Build Docker image
        run: |
          docker build -t ado:scan .
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'ado:scan'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  # =============================================================================
  # SECRET DETECTION
  # =============================================================================
  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for comprehensive scan
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install detect-secrets
        run: |
          pip install detect-secrets
      
      - name: Run detect-secrets scan
        run: |
          detect-secrets scan --all-files --force-use-all-plugins \
            --baseline .secrets.baseline
        continue-on-error: true
      
      - name: Verify secrets baseline
        run: |
          detect-secrets audit .secrets.baseline
        continue-on-error: true

  # =============================================================================
  # SECURITY SUMMARY
  # =============================================================================
  security-summary:
    name: Security Summary
    if: always()
    needs: [codeql-analysis, dependency-scan, container-scan, secret-scan]
    runs-on: ubuntu-latest
    
    steps:
      - name: Security scan summary
        run: |
          echo "## Security Scan Results" >> $GITHUB_STEP_SUMMARY
          echo "- CodeQL Analysis: ${{ needs.codeql-analysis.result }}" >> $GITHUB_STEP_SUMMARY
          echo "- Dependency Scan: ${{ needs.dependency-scan.result }}" >> $GITHUB_STEP_SUMMARY
          echo "- Container Scan: ${{ needs.container-scan.result }}" >> $GITHUB_STEP_SUMMARY
          echo "- Secret Detection: ${{ needs.secret-scan.result }}" >> $GITHUB_STEP_SUMMARY
```

---

## TEMPLATE 3: RELEASE AUTOMATION

**File**: `/.github/workflows/release.yml`

```yaml
name: Release Management

on:
  push:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      version_type:
        description: 'Version bump type'
        required: true
        default: 'patch'
        type: choice
        options:
          - patch
          - minor
          - major

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  # =============================================================================
  # RELEASE PREPARATION
  # =============================================================================
  prepare-release:
    name: Prepare Release
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    outputs:
      should_release: ${{ steps.check.outputs.should_release }}
      new_version: ${{ steps.version.outputs.new_version }}
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install semantic release
        run: |
          pip install python-semantic-release
      
      - name: Check if release needed
        id: check
        run: |
          if semantic-release version --print; then
            echo "should_release=true" >> $GITHUB_OUTPUT
          else
            echo "should_release=false" >> $GITHUB_OUTPUT
          fi
      
      - name: Get new version
        id: version
        if: steps.check.outputs.should_release == 'true'
        run: |
          VERSION=$(semantic-release version --print)
          echo "new_version=$VERSION" >> $GITHUB_OUTPUT

  # =============================================================================
  # BUILD AND TEST
  # =============================================================================
  build-and-test:
    name: Build and Test for Release
    needs: prepare-release
    if: needs.prepare-release.outputs.should_release == 'true'
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run comprehensive tests
        run: |
          pytest --cov=. --cov-report=xml
      
      - name: Build package
        run: |
          python -m build
      
      - name: Verify package
        run: |
          pip install twine
          twine check dist/*
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: release-artifacts
          path: dist/
          retention-days: 7

  # =============================================================================
  # CREATE RELEASE
  # =============================================================================
  create-release:
    name: Create Release
    needs: [prepare-release, build-and-test]
    if: needs.prepare-release.outputs.should_release == 'true'
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install semantic release
        run: |
          pip install python-semantic-release
      
      - name: Create release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          semantic-release version
          semantic-release publish

  # =============================================================================
  # PUBLISH TO PYPI
  # =============================================================================
  publish-pypi:
    name: Publish to PyPI
    needs: [create-release]
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: release-artifacts
          path: dist/
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
          verify_metadata: true

  # =============================================================================
  # CONTAINER PUBLISHING
  # =============================================================================
  publish-container:
    name: Publish Container
    needs: [create-release]
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
      
      - name: Build and push container
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## TEMPLATE 4: ISSUE AND PR TEMPLATES

### GitHub Issue Templates

**File**: `/.github/ISSUE_TEMPLATE/config.yml`

```yaml
blank_issues_enabled: false
contact_links:
  - name: Security Vulnerability
    url: https://github.com/terragon-labs/agentic-dev-orchestrator/security/advisories/new
    about: Report security vulnerabilities privately
  - name: Documentation
    url: https://agentic-dev-orchestrator.readthedocs.io/
    about: Check our documentation first
  - name: Discussions
    url: https://github.com/terragon-labs/agentic-dev-orchestrator/discussions
    about: Ask questions and discuss features
```

**File**: `/.github/ISSUE_TEMPLATE/bug_report.md`

```markdown
---
name: Bug Report
about: Create a report to help us improve
title: "[BUG] "
labels: ["bug", "needs-triage"]
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Environment
- OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12.0]
- Python Version: [e.g. 3.11.0]
- ADO Version: [e.g. 0.1.0]
- Installation Method: [e.g. pip, git clone]

## Additional Context
Add any other context about the problem here.

## Logs
```
Paste relevant logs here
```

## Screenshots
If applicable, add screenshots to help explain your problem.

## Checklist
- [ ] I have searched for similar issues
- [ ] I have read the documentation
- [ ] This is not a security vulnerability (use private reporting for those)
```

**File**: `/.github/ISSUE_TEMPLATE/feature_request.md`

```markdown
---
name: Feature Request
about: Suggest an idea for this project
title: "[FEATURE] "
labels: ["enhancement", "needs-discussion"]
assignees: ''
---

## Feature Description
A clear and concise description of what you want to happen.

## Problem Statement
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

## Proposed Solution
A clear and concise description of what you want to happen.

## Alternative Solutions
A clear and concise description of any alternative solutions or features you've considered.

## Use Cases
Describe specific use cases for this feature:
1. Use case 1
2. Use case 2
3. Use case 3

## Implementation Considerations
Any thoughts on how this might be implemented?

## Additional Context
Add any other context or screenshots about the feature request here.

## Checklist
- [ ] I have searched for similar feature requests
- [ ] I have read the roadmap
- [ ] This feature aligns with the project goals
- [ ] I would be willing to help implement this feature
```

### Pull Request Template

**File**: `/.github/pull_request_template.md`

```markdown
## Description
Brief description of changes made in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement
- [ ] Other (please describe):

## Related Issues
Fixes #(issue number)
Closes #(issue number)
Relates to #(issue number)

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass locally

## Documentation
- [ ] Code is self-documenting
- [ ] Docstrings added/updated
- [ ] README updated (if needed)
- [ ] Changelog updated (if needed)
- [ ] API documentation updated (if needed)

## Security
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented
- [ ] Error handling doesn't leak sensitive information
- [ ] Security implications considered

## Performance
- [ ] Performance impact assessed
- [ ] No significant performance degradation
- [ ] Benchmarks run (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Breaking changes documented
- [ ] Commit messages follow conventional format
- [ ] PR title follows conventional format
- [ ] All CI checks pass
- [ ] Ready for review

## Screenshots (if applicable)
Add screenshots to help reviewers understand the changes.

## Additional Notes
Any additional information that reviewers should know.
```

---

## DEPLOYMENT INSTRUCTIONS

### Step 1: Create GitHub Workflows Directory
```bash
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
```

### Step 2: Deploy Templates
Copy each template to its respective file location as specified above.

### Step 3: Configure Repository Secrets
Navigate to repository Settings > Secrets and variables > Actions:
- Add `PYPI_API_TOKEN` for package publishing
- Optionally add `CODECOV_TOKEN` for coverage reporting

### Step 4: Enable Branch Protection
Navigate to repository Settings > Branches:
- Add rule for `main` branch
- Require PR reviews
- Require status checks to pass
- Include administrators

### Step 5: Test Deployment
1. Create a feature branch
2. Make a small change
3. Open a pull request
4. Verify all workflows execute successfully

---

## VALIDATION CHECKLIST

### Pre-deployment
- [ ] All template syntax is valid
- [ ] Required secrets are configured
- [ ] Branch protection rules are set
- [ ] Workflows reference existing configurations

### Post-deployment
- [ ] CI pipeline executes successfully
- [ ] Security scanning completes without errors
- [ ] Build artifacts are generated correctly
- [ ] Issue and PR templates render properly

### Performance Verification
- [ ] CI pipeline completes within 30 minutes
- [ ] Security scans complete within 15 minutes
- [ ] Build verification completes within 10 minutes
- [ ] No excessive resource usage

---

This completes the Phase 1 implementation templates. These are production-ready workflows that build upon the repository's existing excellent foundations while filling the critical CI/CD automation gap.