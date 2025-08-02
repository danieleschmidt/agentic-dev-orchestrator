# Contributing to Agentic Dev Orchestrator

Thank you for your interest in contributing to the Agentic Dev Orchestrator! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Release Process](#release-process)
- [Community](#community)

## 📜 Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ 
- Git
- GitHub account
- Basic understanding of AI/ML workflows (helpful but not required)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/agentic-dev-orchestrator.git
   cd agentic-dev-orchestrator
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e .[dev,all]
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run basic health check
   npm run health-check
   
   # Run tests
   npm run test
   
   # Run linting
   npm run lint
   ```

4. **Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## 🔄 Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Emergency fixes
- `release/*`: Release preparation

### Workflow Steps

1. **Create a Feature Branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   npm run validate  # Runs lint, test, and security checks
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` adding or updating tests
   - `refactor:` code refactoring
   - `perf:` performance improvements
   - `chore:` maintenance tasks

## 🎯 Coding Standards

### Python Style Guide

We follow PEP 8 with some project-specific conventions:

- **Line Length**: 88 characters (Black formatter default)
- **Imports**: Use `isort` for import organization
- **Type Hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs

### Code Quality Tools

- **Formatter**: Black
- **Linter**: Ruff (replaces flake8, isort, and others)
- **Type Checker**: mypy
- **Security**: Bandit, Safety

### Configuration Files

All tools are configured in `pyproject.toml`. Run formatting and linting:

```bash
npm run format      # Format code
npm run lint        # Check code quality
npm run type-check  # Run type checking
npm run security    # Run security scans
```

### Architecture Guidelines

- Follow the existing agent pipeline pattern
- Use dependency injection for testability
- Implement proper error handling and logging
- Follow SOLID principles
- Document architectural decisions in ADRs

## 🧪 Testing Guidelines

### Test Categories

- **Unit Tests**: Test individual functions/classes (`tests/unit/`)
- **Integration Tests**: Test component interactions (`tests/integration/`)
- **End-to-End Tests**: Test complete workflows (`tests/e2e/`)
- **Performance Tests**: Test performance characteristics (`tests/performance/`)

### Testing Standards

- **Coverage**: Maintain >80% code coverage
- **Test Structure**: Use AAA pattern (Arrange, Act, Assert)
- **Fixtures**: Use pytest fixtures for test data
- **Mocking**: Mock external dependencies

### Running Tests

```bash
npm run test                    # Run all tests
npm run test:coverage          # Run tests with coverage
npm run test:unit              # Run unit tests only
npm run test:integration       # Run integration tests
npm run test:e2e              # Run e2e tests
```

### Test Writing Guidelines

```python
def test_wsjf_calculation_with_valid_input():
    \"\"\"Test WSJF calculation with valid input values.\"\"\"
    # Arrange
    task = create_test_task(
        user_business_value=8,
        time_criticality=6,
        risk_reduction=4,
        job_size=5
    )
    
    # Act
    result = calculate_wsjf(task)
    
    # Assert
    expected = (8 + 6 + 4) / 5
    assert result == expected
```

## 📖 Documentation

### Types of Documentation

- **API Documentation**: Docstrings in code
- **User Guides**: Markdown files in `docs/`
- **Architecture**: ADRs in `docs/adr/`
- **Operations**: Runbooks in `docs/operations/`

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up-to-date with code changes
- Follow markdown best practices

### Building Documentation

```bash
npm run docs:build    # Build documentation
npm run docs:serve    # Serve documentation locally
```

## 📤 Submitting Changes

### Pull Request Process

1. **Ensure Quality**
   ```bash
   npm run validate  # Must pass before submitting
   ```

2. **Create Pull Request**
   - Use the provided [PR template](.github/PULL_REQUEST_TEMPLATE.md)
   - Write a clear title and description
   - Link related issues
   - Add appropriate labels

3. **PR Requirements**
   - [ ] All tests pass
   - [ ] Code coverage maintained
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated (for significant changes)
   - [ ] No merge conflicts

### Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Examples:
```
feat(wsjf): add priority calculation for backlog items
fix(cli): resolve issue with argument parsing
docs(readme): update installation instructions
```

## 👀 Review Process

### What We Look For

- **Functionality**: Does the code work as intended?
- **Quality**: Is the code well-written and maintainable?
- **Tests**: Are there adequate tests?
- **Documentation**: Is documentation updated?
- **Security**: Are there security implications?
- **Performance**: Any performance impacts?

### Review Timeline

- Initial response: Within 2 business days
- Full review: Within 5 business days
- Maintainer review required for breaking changes

### Addressing Feedback

- Respond to all review comments
- Make requested changes in new commits
- Update PR description if scope changes
- Request re-review when ready

## 🚢 Release Process

We use semantic versioning and automated releases:

- **Patch** (0.0.X): Bug fixes
- **Minor** (0.X.0): New features (backward compatible)
- **Major** (X.0.0): Breaking changes

Releases are automated via semantic-release based on conventional commits.

## 🤝 Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussions
- **Documentation**: Check our comprehensive docs first

### Issue Templates

Use the appropriate issue template:
- [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml)
- [Feature Request](.github/ISSUE_TEMPLATE/feature_request.yml)
- [Documentation Issue](.github/ISSUE_TEMPLATE/documentation.yml)

### Recognition

Contributors are recognized in:
- CHANGELOG.md for significant contributions
- GitHub's contributor graph
- Release notes for major contributions

## 🙏 Thank You

Thank you for contributing to Agentic Dev Orchestrator! Your contributions help make AI-powered development workflows accessible to everyone.

For questions about contributing, please:
1. Check existing documentation and issues
2. Create a new issue with the question label
3. Reach out to maintainers in GitHub discussions

---

**Happy Contributing! 🎉**