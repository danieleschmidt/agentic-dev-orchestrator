#!/usr/bin/env python3
"""
Test Template for ADO

Copy this file to create new test modules. Replace placeholders with actual test logic.

Usage:
    cp tests/test_template.py tests/unit/test_new_module.py
    # Edit the new file with your test cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Any, Dict, List

# Import the module you're testing
# from your_module import YourClass, your_function


class TestYourClass:
    """Test cases for YourClass."""
    
    def test_initialization(self):
        """Test class initialization."""
        # Arrange
        # Act
        # Assert
        pass
    
    def test_method_with_valid_input(self):
        """Test method behavior with valid input."""
        # Arrange
        # Act
        # Assert
        pass
    
    def test_method_with_invalid_input(self):
        """Test method behavior with invalid input."""
        # Arrange
        # Act & Assert (for exceptions)
        with pytest.raises(ValueError):
            pass  # Code that should raise ValueError
    
    def test_method_with_mock_dependency(self, mocker):
        """Test method that depends on external service."""
        # Arrange
        mock_service = mocker.patch('your_module.external_service')
        mock_service.return_value = "mocked_response"
        
        # Act
        # Assert
        mock_service.assert_called_once()
    
    @pytest.mark.parametrize("input_value,expected_output", [
        (1, "one"),
        (2, "two"),
        (3, "three"),
    ])
    def test_method_with_multiple_inputs(self, input_value, expected_output):
        """Test method with various input/output combinations."""
        # Arrange
        # Act
        # Assert
        pass


class TestYourFunction:
    """Test cases for standalone functions."""
    
    def test_function_with_simple_input(self):
        """Test function with simple input."""
        # Arrange
        input_data = "test_input"
        expected_output = "expected_result"
        
        # Act
        # result = your_function(input_data)
        
        # Assert
        # assert result == expected_output
        pass
    
    def test_function_with_complex_input(self, sample_data_fixture):
        """Test function with complex input using fixture."""
        # Arrange
        # Use fixture data
        
        # Act
        # result = your_function(sample_data_fixture)
        
        # Assert
        # assert result is not None
        pass


@pytest.mark.integration
class TestIntegration:
    """Integration tests for component interactions."""
    
    def test_component_integration(self, temp_workspace):
        """Test integration between multiple components."""
        # Arrange
        # Set up multiple components
        
        # Act
        # Execute workflow that uses multiple components
        
        # Assert
        # Verify end-to-end behavior
        pass


@pytest.mark.slow
class TestPerformance:
    """Performance tests for the module."""
    
    def test_performance_with_large_dataset(self, performance_timer, large_dataset):
        """Test performance with large dataset."""
        # Arrange
        performance_timer.start()
        
        # Act
        # Execute operation with large dataset
        
        performance_timer.stop()
        
        # Assert
        assert performance_timer.elapsed < 1.0  # Should complete in under 1 second


@pytest.mark.github
class TestGitHubIntegration:
    """Tests that require GitHub API (use mocks in CI)."""
    
    def test_github_api_interaction(self, mock_github_api):
        """Test GitHub API interaction with mocked responses."""
        # Arrange
        # Configure mock responses
        
        # Act
        # Execute code that calls GitHub API
        
        # Assert
        # Verify API was called correctly
        pass


@pytest.mark.llm
class TestLLMIntegration:
    """Tests that require LLM API (use mocks in CI)."""
    
    def test_llm_api_interaction(self, mock_openai_api):
        """Test LLM API interaction with mocked responses."""
        # Arrange
        # Configure mock LLM responses
        
        # Act
        # Execute code that calls LLM API
        
        # Assert
        # Verify LLM was called correctly and response processed
        pass


# =============================================================================
# Test Fixtures (if specific to this module)
# =============================================================================

@pytest.fixture
def sample_data_fixture():
    """Provide sample data for testing."""
    return {
        "key1": "value1",
        "key2": [1, 2, 3],
        "key3": {"nested": "data"}
    }


@pytest.fixture
def large_dataset():
    """Provide large dataset for performance testing."""
    return list(range(10000))  # Adjust size as needed


@pytest.fixture
def mock_external_service():
    """Mock external service for testing."""
    with patch('your_module.external_service') as mock:
        mock.return_value = "mocked_response"
        yield mock


# =============================================================================
# Helper Functions (if needed)
# =============================================================================

def create_test_data(**kwargs) -> Dict[str, Any]:
    """Create test data with optional overrides."""
    defaults = {
        "id": "test-001",
        "name": "Test Item",
        "status": "active"
    }
    defaults.update(kwargs)
    return defaults


def assert_valid_response(response: Dict[str, Any]) -> None:
    """Assert that response has expected structure."""
    assert "status" in response
    assert "data" in response
    assert response["status"] in ["success", "error"]


# =============================================================================
# Test Data Classes (if needed)
# =============================================================================

class TestDataBuilder:
    """Builder pattern for creating test data."""
    
    def __init__(self):
        self.data = {}
    
    def with_id(self, id_value: str) -> 'TestDataBuilder':
        self.data["id"] = id_value
        return self
    
    def with_name(self, name: str) -> 'TestDataBuilder':
        self.data["name"] = name
        return self
    
    def build(self) -> Dict[str, Any]:
        return self.data.copy()


# =============================================================================
# Usage Examples
# =============================================================================

"""
Example usage patterns:

1. Basic unit test:
```python
def test_add_function():
    result = add(2, 3)
    assert result == 5
```

2. Test with fixture:
```python
def test_with_fixture(sample_data_fixture):
    result = process_data(sample_data_fixture)
    assert result is not None
```

3. Test with mock:
```python
def test_with_mock(mocker):
    mock_api = mocker.patch('module.api_call')
    mock_api.return_value = {"status": "success"}
    
    result = function_that_calls_api()
    
    assert result["status"] == "success"
    mock_api.assert_called_once()
```

4. Parametrized test:
```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_uppercase(input, expected):
    assert uppercase(input) == expected
```

5. Exception test:
```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError, match="Invalid input"):
        function_with_validation("invalid")
```

6. Async test:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == "expected"
```
"""


# =============================================================================
# Test Checklist
# =============================================================================

"""
Before submitting tests, ensure:

✅ Tests follow AAA pattern (Arrange, Act, Assert)
✅ Test names clearly describe what's being tested
✅ Tests are independent and can run in any order
✅ External dependencies are mocked appropriately
✅ Edge cases and error conditions are tested
✅ Tests are categorized with appropriate markers
✅ Performance-sensitive code has performance tests
✅ Integration points are covered by integration tests
✅ Code coverage is maintained (aim for >80%)
✅ Tests are documented with clear docstrings
"""