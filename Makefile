# Makefile for Agentic Development Orchestrator (ADO)
# Provides standardized commands for development, testing, and deployment

# =============================================================================
# Configuration
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help

# Project information
PROJECT_NAME := agentic-dev-orchestrator
VERSION := $(shell python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])" 2>/dev/null || echo "0.1.0")
PYTHON_VERSION := $(shell python --version 2>&1 | cut -d' ' -f2)
PYTHON := python
PIP := pip

# Directories
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist
COVERAGE_DIR := htmlcov

# Docker configuration
DOCKER_IMAGE := ado
DOCKER_TAG := latest
DOCKER_REGISTRY := 

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
RESET := \033[0m

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(CYAN)Agentic Development Orchestrator (ADO) - Make Commands$(RESET)"
	@echo "$(CYAN)======================================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)Project: $(PROJECT_NAME) v$(VERSION)$(RESET)"
	@echo "$(YELLOW)Python:  $(PYTHON_VERSION)$(RESET)"
	@echo ""
	@echo "$(GREEN)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Command categories:$(RESET)"
	@echo "  $(MAGENTA)Development:$(RESET) install, dev-install, clean"
	@echo "  $(MAGENTA)Code Quality:$(RESET) format, lint, type-check, security"
	@echo "  $(MAGENTA)Testing:$(RESET) test, test-unit, test-integration, test-e2e, coverage"
	@echo "  $(MAGENTA)Build:$(RESET) build, package, docker-build"
	@echo "  $(MAGENTA)Deployment:$(RESET) docker-run, docker-compose-up"
	@echo "  $(MAGENTA)Utilities:$(RESET) docs, clean-all, requirements"

# =============================================================================
# Development Setup
# =============================================================================

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(RESET)"
	$(PIP) install -e .

dev-install: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(RESET)"
	$(PIP) install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type commit-msg

install-all: ## Install all optional dependencies
	@echo "$(GREEN)Installing all dependencies...$(RESET)"
	$(PIP) install -e ".[all]"

requirements: ## Generate requirements files
	@echo "$(GREEN)Generating requirements files...$(RESET)"
	pip-compile --resolver=backtracking requirements.in
	pip-compile --resolver=backtracking --extra=dev requirements-dev.in

venv: ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(RESET)"
	python -m venv venv
	@echo "$(YELLOW)Activate with: source venv/bin/activate$(RESET)"

# =============================================================================
# Code Quality
# =============================================================================

format: ## Format code with black and ruff
	@echo "$(GREEN)Formatting code...$(RESET)"
	black .
	ruff check --fix .
	isort .

lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(RESET)"
	ruff check .
	black --check .
	isort --check-only .

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checking...$(RESET)"
	mypy .

security: ## Run security scans
	@echo "$(GREEN)Running security scans...$(RESET)"
	bandit -r . -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true
	pip-audit --output-format=json --output-file=pip-audit-report.json || true

quality: lint type-check security ## Run all quality checks

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(GREEN)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(RESET)"
	pytest -v

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(RESET)"
	pytest tests/unit -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(RESET)"
	pytest tests/integration -v

test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running e2e tests...$(RESET)"
	pytest tests/e2e -v

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(RESET)"
	pytest tests/performance -v --benchmark-only

test-parallel: ## Run tests in parallel
	@echo "$(GREEN)Running tests in parallel...$(RESET)"
	pytest -n auto

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(RESET)"
	pytest-watch

coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(RESET)"
	pytest --cov=. --cov-report=html --cov-report=term --cov-report=xml
	@echo "$(YELLOW)Coverage report available at: $(COVERAGE_DIR)/index.html$(RESET)"

coverage-open: coverage ## Open coverage report in browser
	@echo "$(GREEN)Opening coverage report...$(RESET)"
	@if command -v open > /dev/null; then \
		open $(COVERAGE_DIR)/index.html; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open $(COVERAGE_DIR)/index.html; \
	else \
		echo "$(YELLOW)Please open $(COVERAGE_DIR)/index.html manually$(RESET)"; \
	fi

# =============================================================================
# ADO Commands
# =============================================================================

ado-init: ## Initialize ADO in current directory
	@echo "$(GREEN)Initializing ADO...$(RESET)"
	$(PYTHON) ado.py init

ado-status: ## Show ADO backlog status
	@echo "$(GREEN)Checking ADO status...$(RESET)"
	$(PYTHON) ado.py status

ado-run: ## Run ADO autonomous execution
	@echo "$(GREEN)Running ADO...$(RESET)"
	$(PYTHON) ado.py run

ado-discover: ## Run ADO backlog discovery
	@echo "$(GREEN)Running ADO discovery...$(RESET)"
	$(PYTHON) ado.py discover

# =============================================================================
# Build and Package
# =============================================================================

build: clean ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(RESET)"
	$(PYTHON) -m build

package: build ## Create package (alias for build)

check-package: build ## Check package integrity
	@echo "$(GREEN)Checking package integrity...$(RESET)"
	twine check $(DIST_DIR)/*

upload-test: check-package ## Upload to test PyPI
	@echo "$(GREEN)Uploading to test PyPI...$(RESET)"
	twine upload --repository testpypi $(DIST_DIR)/*

upload: check-package ## Upload to PyPI
	@echo "$(RED)Uploading to PyPI...$(RESET)"
	@echo "$(YELLOW)Are you sure? This will publish to production PyPI.$(RESET)"
	@read -p "Press Enter to continue or Ctrl+C to cancel: "
	twine upload $(DIST_DIR)/*

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(RESET)"
	@if [ -d "docs" ]; then \
		sphinx-build -b html docs docs/_build/html; \
		echo "$(YELLOW)Sphinx docs available at: docs/_build/html/index.html$(RESET)"; \
	fi
	mkdocs build
	@echo "$(YELLOW)MkDocs available at: site/index.html$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation locally...$(RESET)"
	mkdocs serve

docs-open: docs ## Build and open documentation
	@echo "$(GREEN)Opening documentation...$(RESET)"
	@if command -v open > /dev/null; then \
		open site/index.html; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open site/index.html; \
	else \
		echo "$(YELLOW)Please open site/index.html manually$(RESET)"; \
	fi

# =============================================================================
# Docker Commands
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(RESET)"
	docker build \
		--build-arg BUILD_DATE=$$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
		--build-arg VERSION=$(VERSION) \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		.

docker-build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(RESET)"
	docker build --target development -t $(DOCKER_IMAGE):dev .

docker-build-prod: ## Build production Docker image
	@echo "$(GREEN)Building production Docker image...$(RESET)"
	docker build --target production -t $(DOCKER_IMAGE):prod .

docker-build-minimal: ## Build minimal Docker image
	@echo "$(GREEN)Building minimal Docker image...$(RESET)"
	docker build --target minimal -t $(DOCKER_IMAGE):minimal .

docker-run: docker-build ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(RESET)"
	docker run --rm -it \
		-v $$(pwd):/app \
		-p 8080:8080 \
		-p 9090:9090 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: docker-build-dev ## Run development Docker container
	@echo "$(GREEN)Running development Docker container...$(RESET)"
	docker run --rm -it \
		-v $$(pwd):/app \
		-p 8080:8080 \
		$(DOCKER_IMAGE):dev

docker-test: ## Run tests in Docker
	@echo "$(GREEN)Running tests in Docker...$(RESET)"
	docker build --target testing -t $(DOCKER_IMAGE):test .
	docker run --rm $(DOCKER_IMAGE):test

docker-push: ## Push Docker image to registry
	@echo "$(GREEN)Pushing Docker image...$(RESET)"
	@if [ -n "$(DOCKER_REGISTRY)" ]; then \
		docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG); \
		docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG); \
	else \
		echo "$(RED)DOCKER_REGISTRY not set$(RESET)"; \
		exit 1; \
	fi

# =============================================================================
# Docker Compose Commands
# =============================================================================

docker-compose-up: ## Start all services with docker-compose
	@echo "$(GREEN)Starting services with docker-compose...$(RESET)"
	docker-compose up -d

docker-compose-dev: ## Start development services
	@echo "$(GREEN)Starting development services...$(RESET)"
	docker-compose --profile development up -d

docker-compose-prod: ## Start production services
	@echo "$(GREEN)Starting production services...$(RESET)"
	docker-compose --profile production up -d

docker-compose-test: ## Run tests with docker-compose
	@echo "$(GREEN)Running tests with docker-compose...$(RESET)"
	docker-compose --profile testing up --abort-on-container-exit

docker-compose-down: ## Stop all services
	@echo "$(GREEN)Stopping services...$(RESET)"
	docker-compose down

docker-compose-logs: ## Show service logs
	@echo "$(GREEN)Showing service logs...$(RESET)"
	docker-compose logs -f

docker-compose-clean: ## Clean up docker-compose resources
	@echo "$(GREEN)Cleaning up docker-compose resources...$(RESET)"
	docker-compose down -v --remove-orphans
	docker-compose rm -f

# =============================================================================
# Environment Management
# =============================================================================

tox: ## Run tests across multiple Python versions
	@echo "$(GREEN)Running tox tests...$(RESET)"
	tox

tox-parallel: ## Run tox tests in parallel
	@echo "$(GREEN)Running tox tests in parallel...$(RESET)"
	tox -p auto

tox-env: ## Run specific tox environment (usage: make tox-env ENV=py311)
	@echo "$(GREEN)Running tox environment: $(ENV)$(RESET)"
	tox -e $(ENV)

# =============================================================================
# Maintenance
# =============================================================================

clean: ## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_DIR)/ $(DIST_DIR)/ *.egg-info/
	rm -rf .pytest_cache/ .coverage $(COVERAGE_DIR)/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-docker: ## Clean Docker images and containers
	@echo "$(GREEN)Cleaning Docker resources...$(RESET)"
	docker system prune -f
	docker image prune -f

clean-all: clean clean-docker ## Clean everything
	@echo "$(GREEN)Cleaning all artifacts...$(RESET)"
	rm -rf .tox/ .nox/
	rm -rf site/ docs/_build/
	rm -rf venv/

update-deps: ## Update all dependencies
	@echo "$(GREEN)Updating dependencies...$(RESET)"
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e ".[dev]"
	pip freeze > requirements-freeze.txt

check-security: ## Check for security vulnerabilities
	@echo "$(GREEN)Checking for security vulnerabilities...$(RESET)"
	safety check
	pip-audit

check-outdated: ## Check for outdated packages
	@echo "$(GREEN)Checking for outdated packages...$(RESET)"
	pip list --outdated

# =============================================================================
# CI/CD Helpers
# =============================================================================

ci-install: ## Install dependencies for CI
	@echo "$(GREEN)Installing CI dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

ci-test: ## Run CI test suite
	@echo "$(GREEN)Running CI test suite...$(RESET)"
	pytest --cov=. --cov-report=xml --cov-report=term-missing

ci-quality: ## Run CI quality checks
	@echo "$(GREEN)Running CI quality checks...$(RESET)"
	black --check .
	ruff check .
	mypy .
	bandit -r . || true

ci-build: ## Build for CI
	@echo "$(GREEN)Building for CI...$(RESET)"
	$(PYTHON) -m build
	twine check $(DIST_DIR)/*

ci-all: ci-install ci-quality ci-test ci-build ## Run full CI pipeline

# =============================================================================
# Development Workflow
# =============================================================================

dev-setup: venv dev-install ## Complete development setup
	@echo "$(GREEN)Development setup complete!$(RESET)"
	@echo "$(YELLOW)Don't forget to activate your virtual environment:$(RESET)"
	@echo "$(CYAN)source venv/bin/activate$(RESET)"

dev-check: format lint type-check test ## Run development checks
	@echo "$(GREEN)All development checks passed!$(RESET)"

dev-reset: clean-all dev-setup ## Reset development environment

release-check: clean ci-all ## Check if ready for release
	@echo "$(GREEN)Release checks complete!$(RESET)"

# =============================================================================
# Monitoring and Health
# =============================================================================

health-check: ## Check system health
	@echo "$(GREEN)Checking system health...$(RESET)"
	@echo "Python: $(PYTHON_VERSION)"
	@echo "Project: $(PROJECT_NAME) v$(VERSION)"
	@$(PYTHON) -c "import ado; print('ADO CLI: OK')"
	@$(PYTHON) -c "import backlog_manager; print('Backlog Manager: OK')"
	@$(PYTHON) -c "import autonomous_executor; print('Autonomous Executor: OK')"

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(RESET)"
	pytest tests/performance --benchmark-only --benchmark-sort=mean

profile: ## Run performance profiling
	@echo "$(GREEN)Running performance profiling...$(RESET)"
	python -m cProfile -o profile.stats ado.py status
	@echo "$(YELLOW)Profile saved to profile.stats$(RESET)"

# =============================================================================
# Information
# =============================================================================

info: ## Show project information
	@echo "$(CYAN)Project Information$(RESET)"
	@echo "$(CYAN)==================$(RESET)"
	@echo "Name:           $(PROJECT_NAME)"
	@echo "Version:        $(VERSION)"
	@echo "Python:         $(PYTHON_VERSION)"
	@echo "Working Dir:    $$(pwd)"
	@echo "Git Branch:     $$(git branch --show-current 2>/dev/null || echo 'N/A')"
	@echo "Git Commit:     $$(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
	@echo "Docker Image:   $(DOCKER_IMAGE):$(DOCKER_TAG)"

env: ## Show environment information
	@echo "$(CYAN)Environment Information$(RESET)"
	@echo "$(CYAN)=====================$(RESET)"
	@env | grep -E '^(PYTHON|PIP|PATH|VIRTUAL_ENV|ADO_|GITHUB_|OPENAI_)' | sort

status: info health-check ## Show complete status

# =============================================================================
# Aliases
# =============================================================================

install-dev: dev-install ## Alias for dev-install
fmt: format ## Alias for format
check: dev-check ## Alias for dev-check
test-all: test ## Alias for test