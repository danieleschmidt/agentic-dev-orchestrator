# GitHub Actions Workflow Examples

This directory contains example GitHub Actions workflow files that should be manually copied to `.github/workflows/` directory by repository maintainers.

## 🚨 Important: Manual Setup Required

Due to GitHub App permission limitations, these workflow files **cannot be automatically created**. Repository maintainers must manually copy these files to the `.github/workflows/` directory.

## 📁 Available Workflows

### Core Workflows

#### 1. `ci.yml` - Continuous Integration
**Purpose**: Validates code quality, runs tests, and performs security checks on pull requests and pushes.

**Features**:
- ✅ Multi-Python version testing (3.9, 3.10, 3.11, 3.12)
- 🔍 Code linting with ruff and formatting checks
- 🎯 Type checking with mypy
- 🧪 Test execution with pytest and coverage reporting
- 🔐 Security scanning with Bandit, Safety, and Semgrep
- 📦 Package building and validation
- 🐳 Docker image building and testing
- 📊 Coverage reporting to Codecov

**Triggers**:
- Pull requests to `main` and `develop` branches
- Pushes to `main` and `develop` branches
- Manual workflow dispatch

#### 2. `cd.yml` - Continuous Deployment
**Purpose**: Automates package building, publishing, and Docker image deployment.

**Features**:
- ✅ Full validation pipeline before deployment
- 📦 PyPI package publishing (production and test)
- 🐳 Docker image building and publishing to GitHub Container Registry
- 📋 SBOM (Software Bill of Materials) generation
- 🔒 Environment protection for production deployments
- 📢 Deployment notifications

**Triggers**:
- GitHub releases
- Git tags starting with 'v'
- Manual workflow dispatch

#### 3. `security.yml` - Comprehensive Security Scanning
**Purpose**: Performs comprehensive security analysis across multiple dimensions.

**Features**:
- 🔍 Dependency vulnerability scanning (Safety, pip-audit)
- 📝 Static code analysis (Bandit, Semgrep)
- 🔐 Secret detection (TruffleHog, GitLeaks)
- 🐳 Container security scanning (Trivy, Snyk)
- 📋 Compliance and license checking
- 📊 Automated security summary generation
- 🚨 SARIF upload to GitHub Security tab

**Triggers**:
- Daily scheduled scan (6 AM UTC)
- Pushes to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

#### 4. `dependency-update.yml` - Automated Dependency Management
**Purpose**: Automates dependency updates with security and compatibility checks.

**Features**:
- 🔄 Automated Python dependency updates (patch/minor/major)
- 🎯 GitHub Actions version updates
- 🔐 Security vulnerability detection
- 📋 Dependency review for pull requests
- 🚨 Automatic security issue creation
- 📊 Comprehensive change reporting

**Triggers**:
- Weekly scheduled updates (Monday 9 AM UTC)
- Manual workflow dispatch with update type selection

## 🛠️ Setup Instructions

### 1. Copy Workflow Files
```bash
# Create workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy all example workflows
cp docs/workflows/examples/*.yml .github/workflows/
```

### 2. Configure Repository Secrets

#### Required Secrets:
```bash
# PyPI Publishing
PYPI_API_TOKEN=your_pypi_token
TEST_PYPI_API_TOKEN=your_test_pypi_token

# Security Scanning (Optional)
SEMGREP_APP_TOKEN=your_semgrep_token
SNYK_TOKEN=your_snyk_token
GITLEAKS_LICENSE=your_gitleaks_license

# Additional tokens are automatically provided by GitHub
GITHUB_TOKEN  # Automatically provided
```

### 3. Configure Repository Settings

#### Branch Protection Rules:
1. Navigate to Settings > Branches
2. Add rule for `main` branch:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (minimum 1)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

#### Required Status Checks:
- `Run Tests`
- `Security Scanning`
- `Build Package`
- `Build Docker Image`

### 4. Environment Configuration

#### Production Environment:
1. Navigate to Settings > Environments
2. Create "production" environment
3. Configure protection rules:
   - ✅ Required reviewers
   - ✅ Wait timer (optional)
   - ✅ Deployment branches (tags and main only)

## 🔧 Customization Guide

### Modifying Python Versions
Edit the matrix strategy in `ci.yml`:
```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]  # Modify as needed
```

### Adding Additional Security Tools
Extend the security workflow by adding new steps:
```yaml
- name: Custom Security Tool
  run: |
    custom-security-tool --scan .
```

### Customizing Deployment Targets
Modify the publishing steps in `cd.yml`:
```yaml
- name: Deploy to Custom Registry
  run: |
    # Custom deployment logic
```

### Adjusting Update Schedules
Modify the cron expressions:
```yaml
schedule:
  - cron: '0 9 * * 1'  # Weekly Monday 9 AM UTC
  - cron: '0 6 * * *'  # Daily 6 AM UTC
```

## 📊 Monitoring and Observability

### Workflow Artifacts
Each workflow generates artifacts for analysis:
- **CI**: Test results, coverage reports, build artifacts
- **CD**: Distribution packages, SBOMs, deployment logs
- **Security**: Vulnerability reports, SARIF files, compliance reports
- **Dependencies**: Audit reports, update summaries

### GitHub Security Integration
- SARIF files are automatically uploaded to GitHub Security tab
- Dependency vulnerabilities appear in the Insights > Dependency graph
- Security advisories trigger automated issue creation

### Notifications
- Deployment status notifications in workflow summaries
- Security issue creation for critical vulnerabilities
- Pull request comments for dependency review failures

## 🚀 Advanced Features

### Parallel Execution
Workflows are optimized for parallel execution:
- CI jobs run in parallel where possible
- Security scans execute concurrently
- Multi-platform builds are parallelized

### Caching Strategy
Efficient caching reduces execution time:
- pip dependency caching
- Docker layer caching
- GitHub Actions cache optimization

### Error Handling
Robust error handling and recovery:
- Conditional execution based on previous job results
- Graceful handling of optional security tools
- Comprehensive failure reporting

## 📋 Maintenance

### Regular Updates
- Review and update workflow versions monthly
- Monitor GitHub Actions marketplace for new tools
- Update security scanning tools and configurations
- Review and adjust branch protection rules

### Performance Optimization
- Monitor workflow execution times
- Optimize caching strategies
- Review artifact retention policies
- Consider workflow concurrency limits

## 🔍 Troubleshooting

### Common Issues
1. **Permission Errors**: Ensure GITHUB_TOKEN has sufficient permissions
2. **Secret Errors**: Verify all required secrets are configured
3. **Test Failures**: Check test dependencies and Python version compatibility
4. **Security Scan Failures**: Review security tool configurations

### Debug Mode
Enable debug logging by setting repository secret:
```
ACTIONS_STEP_DEBUG=true
```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening Guide](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Python Package Publishing](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)