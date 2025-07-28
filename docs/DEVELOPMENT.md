# Development Guide

## Prerequisites

• Python 3.8+ installed
• Git configured for development
• Code editor with Python support

## Setup Instructions

1. Clone repository: `git clone [repository-url]`
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements-dev.txt`

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_ado_cli.py
```

## Development Commands

• Linting: `ruff check .`
• Formatting: `black .`
• Type checking: `mypy src/`

## Project Structure

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed architecture information.