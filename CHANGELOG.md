# Changelog

All notable changes to the Agentic Development Orchestrator (ADO) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive GitHub Actions workflows for CI/CD, security scanning, and dependency management
- Enhanced pre-commit configuration with security and quality checks
- Expanded GitHub issue templates for better bug reporting and feature requests
- Comprehensive pull request template with detailed checklist
- Enhanced security documentation with vulnerability reporting guidelines
- Automated changelog generation with semantic-release

### Changed
- Enhanced pyproject.toml with semantic-release configuration
- Improved documentation structure and content

### Security
- Added comprehensive security scanning automation
- Enhanced secret detection and vulnerability reporting processes

## [0.1.0] - 2025-01-15

### Added
- Initial release of Agentic Development Orchestrator
- CLI interface for autonomous backlog management
- WSJF (Weighted Shortest Job First) prioritization algorithm
- Multi-agent execution pipeline with AutoGen integration
- Planner, Coder, Reviewer, and Merger agent framework
- Human-in-the-loop pattern for edge case resolution
- GitHub integration for automated PR creation
- Docker containerization with observability stack
- Comprehensive test suite (unit, integration, e2e, performance)
- Documentation and architecture decision records (ADRs)
- Security scanning and SBOM generation
- Monitoring setup with Prometheus and Grafana

### Features
- **Backlog Management**: JSON-based backlog items with WSJF scoring
- **Agent Pipeline**: Automated task execution through specialized AI agents
- **Safety Gates**: Policy validation and test coverage checks
- **GitHub Integration**: Automated PR creation and status reporting
- **Observability**: Comprehensive monitoring and alerting setup
- **Security**: Vulnerability scanning and supply chain security

### Documentation
- README with quick start guide
- Architecture documentation
- Contributing guidelines
- Code of conduct
- Security policy
- Development setup guide

### Infrastructure
- Docker multi-stage build configuration
- Docker Compose for local development
- Prometheus monitoring configuration
- GitHub issue and PR templates
- Pre-commit hooks for code quality
- Comprehensive .gitignore for Python projects

---

## Release Guidelines

This project uses [Semantic Versioning](https://semver.org/) and [Conventional Commits](https://www.conventionalcommits.org/) to automate releases.

### Commit Message Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types
- **feat**: A new feature (triggers minor version bump)
- **fix**: A bug fix (triggers patch version bump)
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance (triggers patch version bump)
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files

### Breaking Changes
To trigger a major version bump, include `BREAKING CHANGE:` in the commit footer or use `!` after the type/scope:
```
feat!: remove deprecated API endpoints

BREAKING CHANGE: The /v1/legacy endpoint has been removed. Use /v2/tasks instead.
```

### Version History
- **v0.1.0**: Initial release with core functionality
- **Future versions**: Automated releases based on conventional commits

For detailed release notes and upgrade guides, see individual release pages on GitHub.

---

*This changelog is automatically generated and maintained. Manual edits may be overwritten during automated releases.*