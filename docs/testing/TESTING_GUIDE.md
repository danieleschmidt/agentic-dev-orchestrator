# Testing Guide for Agentic Dev Orchestrator

This guide covers testing practices, patterns, and infrastructure for the ADO project.

## 📋 Testing Philosophy

Our testing strategy follows the **Test Pyramid** principle:

```
    /\
   /  \  E2E Tests (Few, High-Level)
  /____\
 /      \  Integration Tests (Some, Component-Level)
/________\
Unit Tests (Many, Fast, Isolated)
```

### Testing Principles

1. **Fast Feedback**: Unit tests run quickly for immediate feedback
2. **Reliability**: Tests are deterministic and not flaky
3. **Maintainability**: Tests are easy to read, write, and maintain
4. **Coverage**: Aim for >80% code coverage with meaningful tests
5. **Isolation**: Tests don't depend on external services or state

## 🏗️ Test Structure

### Directory Organization

```
tests/
├── conftest.py                 # Global fixtures and configuration
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_ado_cli.py
│   ├── test_backlog_manager.py
│   └── test_wsjf_calculator.py
├── integration/                # Component integration tests
│   ├── test_github_integration.py
│   └── test_agent_pipeline.py
├── e2e/                        # End-to-end workflow tests
│   ├── test_full_workflow.py
│   └── test_error_scenarios.py
├── performance/                # Performance and load tests
│   ├── test_performance.py
│   └── test_load_testing.py
└── security/                   # Security-focused tests
    └── test_security_scanner.py
```

### Test Categories

| Category | Purpose | Speed | Dependencies | Coverage Target |
|----------|---------|-------|--------------|----------------|
| **Unit** | Test individual functions/classes | Very Fast | None | 90%+ |
| **Integration** | Test component interactions | Fast | Limited | 70%+ |
| **E2E** | Test complete workflows | Slow | Full Stack | 50%+ |
| **Performance** | Test performance characteristics | Variable | Full Stack | N/A |
| **Security** | Test security aspects | Fast | None | N/A |

## 🧪 Test Categories in Detail

### Unit Tests

**Purpose**: Test individual functions, methods, and classes in isolation.

**Characteristics**:
- No external dependencies (database, network, file system)
- Use mocks and stubs for dependencies
- Run in milliseconds
- Deterministic and repeatable

**Example**:
```python
@pytest.mark.unit
def test_wsjf_calculation():
    """Test WSJF calculation with valid inputs."""
    # Arrange
    item = BacklogItem(
        value=8,
        time_criticality=6,
        risk_reduction=4,
        effort=5
    )
    
    # Act
    wsjf_score = calculate_wsjf(item)
    
    # Assert
    expected = (8 + 6 + 4) / 5  # 3.6
    assert wsjf_score == expected
```

### Integration Tests

**Purpose**: Test how components work together.

**Characteristics**:
- Test component boundaries and interactions
- May use test databases or mock services
- Focus on data flow and integration points

**Example**:
```python
@pytest.mark.integration
def test_backlog_manager_with_file_system(temp_workspace):
    """Test BacklogManager reading from actual files."""
    # Arrange
    create_sample_backlog_files(temp_workspace)
    manager = BacklogManager(temp_workspace)
    
    # Act
    manager.load_backlog()
    prioritized_items = manager.get_prioritized_items()
    
    # Assert
    assert len(prioritized_items) > 0
    assert prioritized_items[0].wsjf_score > prioritized_items[-1].wsjf_score
```

### End-to-End Tests

**Purpose**: Test complete user workflows from start to finish.

**Characteristics**:
- Test the entire system as a black box
- Use real or realistic test data
- Verify business outcomes

**Example**:
```python
@pytest.mark.e2e
@pytest.mark.slow
def test_complete_ado_workflow(temp_workspace, mock_github_api, mock_openai_api):
    """Test complete ADO workflow: load backlog → execute → create PR."""
    # Arrange
    setup_complete_test_environment(temp_workspace)
    
    # Act
    result = run_ado_command(["run", "--max-items", "1"])
    
    # Assert
    assert result.returncode == 0
    assert "Pull request created" in result.stdout
    mock_github_api.assert_called()
```

### Performance Tests

**Purpose**: Verify performance characteristics and identify bottlenecks.

**Example**:
```python
@pytest.mark.slow
@pytest.mark.performance
def test_backlog_processing_performance(large_backlog_dataset):
    """Test performance with large backlog."""
    manager = BacklogManager()
    manager.items = large_backlog_dataset  # 1000+ items
    
    start_time = time.time()
    prioritized = manager.get_prioritized_items()
    elapsed = time.time() - start_time
    
    # Should process 1000 items in under 1 second
    assert elapsed < 1.0
    assert len(prioritized) == len(large_backlog_dataset)
```

## 🛠️ Testing Tools and Frameworks

### Core Testing Stack

- **pytest**: Primary testing framework
- **pytest-cov**: Code coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-xdist**: Parallel test execution
- **pytest-benchmark**: Performance testing
- **factory-boy**: Test data generation

### Configuration

Testing is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "--strict-markers",
    "--cov=.",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-fail-under=80"
]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests", 
    "unit: marks tests as unit tests",
    "e2e: marks tests as end-to-end tests",
    "github: marks tests that require GitHub API",
    "llm: marks tests that require LLM API"
]
```

## 🏃‍♂️ Running Tests

### Basic Commands

```bash
# Run all tests
npm run test

# Run with coverage
npm run test:coverage

# Run specific test categories
pytest -m unit                    # Only unit tests
pytest -m "integration or e2e"    # Integration and E2E tests
pytest -m "not slow"              # Skip slow tests

# Run specific test files
pytest tests/unit/test_backlog_manager.py
pytest tests/integration/

# Run tests in parallel
pytest -n auto                    # Auto-detect CPU cores
pytest -n 4                       # Use 4 processes
```

### Debugging Tests

```bash
# Run with detailed output
pytest -v

# Stop on first failure
pytest -x

# Enter debugger on failure
pytest --pdb

# Show local variables in traceback
pytest -l

# Run specific test with debugging
pytest -v -s tests/unit/test_backlog_manager.py::TestBacklogManager::test_wsjf_calculation
```

## 📊 Code Coverage

### Coverage Goals

- **Overall**: >80% line coverage
- **Unit Tests**: >90% of business logic
- **Critical Paths**: 100% coverage for security, data integrity

### Coverage Reports

```bash
# Generate coverage report
npm run test:coverage

# View HTML report
open htmlcov/index.html

# Coverage by file
coverage report --show-missing

# Check specific files
coverage report --include="backlog_manager.py"
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["."]
omit = [
    "*/tests/*",
    "*/venv/*",
    "setup.py"
]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
```

## 🔧 Test Fixtures and Utilities

### Global Fixtures (conftest.py)

Common fixtures available to all tests:

```python
# Environment setup
@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""

# Temporary workspace
@pytest.fixture
def temp_workspace() -> Path:
    """Create isolated test workspace."""

# Sample data
@pytest.fixture
def sample_backlog_item() -> BacklogItem:
    """Generate test backlog item."""

# Mock services
@pytest.fixture
def mock_github_api():
    """Mock GitHub API calls."""
```

### Custom Assertions

```python
# File system assertions
@pytest.fixture
def assert_file_exists():
    def _assert(path: Path, should_exist: bool = True):
        if should_exist:
            assert path.exists(), f"File {path} should exist"
        else:
            assert not path.exists(), f"File {path} should not exist"
    return _assert

# Performance assertions
@pytest.fixture
def assert_performance():
    def _assert(elapsed: float, max_time: float):
        assert elapsed < max_time, f"Operation took {elapsed}s, expected <{max_time}s"
    return _assert
```

## 🎭 Mocking and Test Doubles

### When to Mock

- External API calls (GitHub, OpenAI, Anthropic)
- File system operations (when testing logic, not I/O)
- Network requests
- Time-dependent operations
- Expensive computations

### Mocking Examples

```python
# Mock external API
@patch("github.Github")
def test_create_pull_request(mock_github):
    mock_repo = mock_github.return_value.get_repo.return_value
    mock_repo.create_pull.return_value = Mock(number=123)
    
    # Test code here
    
    mock_repo.create_pull.assert_called_once()

# Mock with pytest-mock
def test_with_mocker(mocker):
    mocker.patch("openai.ChatCompletion.create", return_value={"content": "test"})
    # Test code here

# Context manager mocking
def test_with_context_manager():
    with patch("builtins.open", mock_open(read_data="test data")):
        # Test file operations
        pass
```

### Mock Guidelines

1. **Mock at the right level**: Mock external boundaries, not internal logic
2. **Verify interactions**: Use `assert_called_with()` to verify mock usage
3. **Reset mocks**: Use `mock.reset_mock()` between tests if needed
4. **Patch close to usage**: Patch where the function is used, not where it's defined

## 📈 Test Data Management

### Test Data Strategies

1. **Inline Data**: Small, simple test data defined in tests
2. **Fixtures**: Reusable test data defined in conftest.py
3. **Factories**: Dynamic test data generation with factory-boy
4. **Files**: Complex test data stored in `tests/fixtures/`

### Factory Pattern

```python
# Using factory-boy
class BacklogItemFactory(factory.Factory):
    class Meta:
        model = BacklogItem
    
    id = factory.Sequence(lambda n: f"item-{n:03d}")
    title = factory.Faker("sentence", nb_words=4)
    type = factory.Iterator(["feature", "bug", "chore"])
    effort = factory.Faker("random_int", min=1, max=10)
    value = factory.Faker("random_int", min=1, max=10)

# Usage in tests
def test_with_factory():
    item = BacklogItemFactory(effort=3, value=8)
    assert item.effort == 3
    assert item.value == 8
```

## 🚀 Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled runs (daily)

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      
      - name: Run tests
        run: |
          pytest --cov --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Quality Gates

- **Coverage Threshold**: >80% overall coverage
- **Performance Regression**: >20% performance degradation fails CI
- **Security Tests**: All security tests must pass
- **Flaky Test Detection**: Tests failing >5% of time are investigated

## 🐛 Debugging Failed Tests

### Common Test Failures

#### 1. Assertion Errors
```bash
# Run with detailed output
pytest -vv

# Show local variables
pytest -l

# Compare with expected output
pytest --tb=long
```

#### 2. Mock-Related Issues
```python
# Verify mock calls
assert mock_function.called
assert mock_function.call_count == 2
assert mock_function.called_with("expected_arg")

# Debug mock calls
print(mock_function.call_args_list)
```

#### 3. Fixture Issues
```python
# Debug fixture values
@pytest.fixture
def debug_fixture(sample_data):
    print(f"Fixture data: {sample_data}")
    return sample_data
```

#### 4. Test Environment Issues
```bash
# Check environment variables
pytest -s  # Don't capture output

# Run single test with debugging
pytest -v -s --no-cov tests/path/to/test.py::test_function
```

## 📝 Writing Good Tests

### Test Structure (AAA Pattern)

```python
def test_example():
    # Arrange - Set up test data and conditions
    item = BacklogItem(title="Test", effort=5)
    manager = BacklogManager()
    
    # Act - Execute the behavior being tested
    result = manager.calculate_priority(item)
    
    # Assert - Verify the expected outcome
    assert result > 0
    assert isinstance(result, float)
```

### Test Naming Conventions

```python
# Good test names (describe what's being tested)
def test_wsjf_calculation_with_valid_inputs()
def test_backlog_loading_handles_missing_file()
def test_github_integration_retries_on_rate_limit()

# Poor test names (too generic)
def test_manager()  # What about the manager?
def test_failure()  # What kind of failure?
def test_api()      # Which API? What behavior?
```

### Test Documentation

```python
def test_priority_calculation_with_aging():
    """
    Test that WSJF priority calculation correctly applies aging factor.
    
    The aging factor should increase priority for older items to prevent
    them from being perpetually deprioritized by newer, higher-value items.
    
    Given:
    - An older item (created 30 days ago) with moderate WSJF score
    - A newer item (created today) with same WSJF score
    
    When:
    - Priority calculation is performed with aging enabled
    
    Then:
    - Older item should have higher effective priority
    - Aging multiplier should be applied correctly
    """
    # Test implementation here
```

## 🔬 Test-Driven Development (TDD)

### TDD Cycle

1. **Red**: Write a failing test
2. **Green**: Write minimal code to make it pass  
3. **Refactor**: Improve code while keeping tests green

### Example TDD Session

```python
# Step 1: Write failing test
def test_wsjf_calculation():
    item = BacklogItem(value=8, effort=4)
    result = calculate_wsjf(item)  # Function doesn't exist yet
    assert result == 2.0

# Step 2: Make it pass (minimal implementation)
def calculate_wsjf(item):
    return item.value / item.effort

# Step 3: Refactor and add more tests
def test_wsjf_with_time_criticality():
    item = BacklogItem(value=8, time_criticality=6, effort=4)
    result = calculate_wsjf(item)
    assert result == 3.5  # (8 + 6) / 4
```

## 🎯 Best Practices

### Do's ✅

- **Write tests first** for new features (TDD)
- **Test behavior, not implementation** details
- **Use descriptive test names** that explain the scenario
- **Keep tests independent** - no test should depend on another
- **Mock external dependencies** to ensure isolation
- **Use appropriate test categories** (unit/integration/e2e)
- **Maintain test documentation** and examples
- **Review test code** as carefully as production code

### Don'ts ❌

- **Don't test private methods** directly - test through public interface
- **Don't make tests dependent** on execution order
- **Don't use production data** in tests
- **Don't ignore flaky tests** - fix or remove them
- **Don't over-mock** - mock at boundaries, not everywhere
- **Don't write tests just for coverage** - write meaningful tests
- **Don't skip test maintenance** when refactoring code

## 📚 Additional Resources

### Documentation
- [pytest Documentation](https://docs.pytest.org/)
- [Python Testing 101](https://realpython.com/python-testing/)
- [Test-Driven Development Guide](https://testdriven.io/)

### Tools and Libraries
- [pytest-cov](https://pytest-cov.readthedocs.io/) - Coverage reporting
- [factory-boy](https://factoryboy.readthedocs.io/) - Test data generation
- [responses](https://github.com/getsentry/responses) - HTTP request mocking
- [freezegun](https://github.com/spulec/freezegun) - Time mocking

### Internal Resources
- [Contributing Guide](../../CONTRIBUTING.md) - Development workflow
- [Architecture Documentation](../../ARCHITECTURE.md) - System design
- [API Documentation](../api/) - API reference

---

**Happy Testing! 🧪** Remember: Good tests are an investment in code quality, maintainability, and developer confidence.