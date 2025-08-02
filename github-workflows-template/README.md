# GitHub Actions Workflows Template

This directory contains GitHub Actions workflow templates that need to be manually added to `.github/workflows/` due to GitHub's security restrictions on automated workflow creation.

## Installation Instructions

1. **Create the workflows directory:**
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy the workflow files:**
   ```bash
   cp github-workflows-template/*.yml .github/workflows/
   ```

3. **Required Secrets Setup:**

   ### For CI/CD:
   - `CODECOV_TOKEN` (optional): For code coverage reporting
   
   ### For Releases:
   - `PYPI_API_TOKEN`: PyPI API token for package publishing
   - `GITHUB_TOKEN`: Automatically provided by GitHub

4. **Repository Settings:**
   - Enable "Actions" in repository settings
   - For releases, create a "release" environment in repository settings
   - Configure branch protection rules as needed

## Workflow Overview

### `ci.yml` - Continuous Integration
- **Triggers**: Push to main/develop, Pull requests to main
- **Jobs**:
  - **Test**: Multi-platform testing (Python 3.8-3.12)
  - **Build**: Package building and installation testing
  - **Docker**: Docker image building and testing
- **Features**:
  - Code quality checks (ruff, black, mypy)
  - Security scanning (bandit, safety, pip-audit)
  - Test coverage reporting
  - Artifact uploads

### `release.yml` - Release Automation
- **Triggers**: Git tags starting with 'v'
- **Jobs**:
  - **Build-binaries**: Cross-platform binary creation
  - **Release-PyPI**: Automated PyPI publishing
  - **Release-Docker**: Docker image publishing to GHCR
  - **GitHub-Release**: GitHub release with binaries
- **Features**:
  - Multi-platform binary support (Linux, macOS, Windows)
  - Semantic versioning
  - Automated changelog generation

### `security.yml` - Security Scanning
- **Triggers**: Push to main, PRs, weekly schedule
- **Jobs**:
  - **Security-scan**: Multiple security tools
  - **CodeQL**: GitHub's code analysis
  - **Dependency-review**: Dependency vulnerability checks
- **Features**:
  - Comprehensive security scanning
  - Automated dependency reviews
  - Security report artifacts

## Configuration Notes

1. **Python Version**: Default is 3.11, configured in `env.PYTHON_VERSION`
2. **Platforms**: CI tests on multiple Python versions, releases support Linux/macOS/Windows
3. **Security**: All workflows include security best practices and scanning
4. **Caching**: pip dependencies are cached for faster builds
5. **Artifacts**: Build artifacts and security reports are uploaded for review

## Environment Setup

After copying workflows, ensure your development environment has all required tools:

```bash
# Install development dependencies
pip install -e ".[dev,security]"

# Run local quality checks (same as CI)
ruff check .
black --check .
mypy .
pytest --cov=.

# Test security scanning
bandit -r .
safety check
pip-audit
```

## Troubleshooting

- **Permission Errors**: Ensure repository has Actions enabled
- **Secret Errors**: Verify all required secrets are configured
- **Build Failures**: Test locally first with the commands in the workflows
- **Docker Issues**: Ensure Dockerfile builds correctly locally

## Next Steps

After setting up workflows:
1. Push changes to trigger CI
2. Create a release tag (e.g., `v0.1.1`) to test release workflow
3. Monitor Actions tab for execution status
4. Configure notifications for workflow failures

This automation provides production-grade CI/CD for your CLI tool with comprehensive testing, security scanning, and multi-platform distribution.