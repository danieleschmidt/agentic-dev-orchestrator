#!/usr/bin/env python3
"""
Global pytest configuration and shared fixtures for ADO tests
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch

import pytest
import yaml

from backlog_manager import BacklogManager, BacklogItem
from autonomous_executor import AutonomousExecutor


# =============================================================================
# Test Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", 
        "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", 
        "github: marks tests that require GitHub API"
    )
    config.addinivalue_line(
        "markers", 
        "llm: marks tests that require LLM API"
    )
    config.addinivalue_line(
        "markers", 
        "requires_network: marks tests that require network access"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Environment and Configuration Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    env_vars = {
        "ADO_LOG_LEVEL": "DEBUG",
        "TESTING": "true",
        "GITHUB_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-key",
    }
    
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env_vars
    
    # Restore original environment
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create basic ADO structure
        (workspace / "backlog").mkdir()
        (workspace / "docs" / "status").mkdir(parents=True)
        (workspace / "escalations").mkdir()
        (workspace / ".ado" / "cache").mkdir(parents=True)
        (workspace / ".ado" / "locks").mkdir(parents=True)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(workspace)
        
        yield workspace
        
        # Restore original directory
        os.chdir(original_cwd)


@pytest.fixture
def sample_backlog_config() -> Dict[str, Any]:
    """Sample backlog configuration for testing."""
    return {
        "name": "Test Project",
        "version": "1.0.0",
        "description": "Test project for ADO",
        "settings": {
            "wsjf": {
                "enable_aging": True,
                "aging_multiplier": 1.05,
                "max_aging_multiplier": 2.0
            },
            "execution": {
                "max_items_per_run": 5,
                "timeout": 300,
                "retry_attempts": 3
            }
        },
        "backlog": []
    }


@pytest.fixture
def sample_backlog_item() -> BacklogItem:
    """Sample backlog item for testing."""
    return BacklogItem(
        id="test-001",
        title="Test Feature Implementation",
        type="feature",
        description="Implement a test feature for validation",
        acceptance_criteria=[
            "Feature should work correctly",
            "Tests should pass",
            "Documentation should be updated"
        ],
        effort=5,
        value=8,
        time_criticality=6,
        risk_reduction=4,
        status="READY",
        risk_tier="medium",
        created_at="2025-01-27T00:00:00Z",
        links=["https://github.com/test/repo/issues/123"]
    )


@pytest.fixture
def sample_backlog_items() -> list[BacklogItem]:
    """Multiple sample backlog items for testing."""
    return [
        BacklogItem(
            id="item-001",
            title="High Value Feature",
            type="feature",
            description="A high-value feature",
            acceptance_criteria=["Should work"],
            effort=3,
            value=9,
            time_criticality=8,
            risk_reduction=7,
            status="READY",
            risk_tier="low",
            created_at="2025-01-27T00:00:00Z",
            links=[]
        ),
        BacklogItem(
            id="item-002", 
            title="Bug Fix",
            type="bug",
            description="Critical bug fix",
            acceptance_criteria=["Bug should be fixed"],
            effort=2,
            value=7,
            time_criticality=9,
            risk_reduction=8,
            status="READY",
            risk_tier="high",
            created_at="2025-01-27T01:00:00Z",
            links=[]
        ),
        BacklogItem(
            id="item-003",
            title="Technical Debt",
            type="chore",
            description="Refactor old code",
            acceptance_criteria=["Code should be cleaner"],
            effort=8,
            value=4,
            time_criticality=2,
            risk_reduction=6,
            status="NEW",
            risk_tier="low",
            created_at="2025-01-27T02:00:00Z",
            links=[]
        )
    ]


# =============================================================================
# Component Fixtures
# =============================================================================

@pytest.fixture
def backlog_manager(temp_workspace: Path) -> BacklogManager:
    """BacklogManager instance for testing."""
    return BacklogManager(str(temp_workspace))


@pytest.fixture
def autonomous_executor(temp_workspace: Path) -> AutonomousExecutor:
    """AutonomousExecutor instance for testing."""
    return AutonomousExecutor(str(temp_workspace))


@pytest.fixture
def mock_github_api():
    """Mock GitHub API responses."""
    with patch('github.Github') as mock_github:
        mock_repo = Mock()
        mock_repo.create_pull.return_value = Mock(number=123, html_url="https://github.com/test/repo/pull/123")
        mock_repo.get_contents.return_value = Mock(decoded_content=b"test content")
        
        mock_github.return_value.get_repo.return_value = mock_repo
        yield mock_github


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses."""
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the AI agent."
                }
            }
        ]
    }
    
    with patch('openai.ChatCompletion.create', return_value=Mock(**mock_response)):
        yield mock_response


@pytest.fixture
def mock_anthropic_api():
    """Mock Anthropic API responses."""
    mock_response = Mock()
    mock_response.completion = "This is a test response from Claude."
    
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_anthropic.return_value.completions.create.return_value = mock_response
        yield mock_response


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def create_test_backlog_file(temp_workspace: Path, sample_backlog_config: Dict[str, Any]):
    """Create a test backlog.yml file."""
    backlog_file = temp_workspace / "backlog.yml"
    with open(backlog_file, 'w') as f:
        yaml.dump(sample_backlog_config, f)
    return backlog_file


@pytest.fixture
def create_test_backlog_items(temp_workspace: Path, sample_backlog_items: list[BacklogItem]):
    """Create test backlog item files."""
    backlog_dir = temp_workspace / "backlog"
    created_files = []
    
    for item in sample_backlog_items:
        item_file = backlog_dir / f"{item.id}.json"
        with open(item_file, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            item_dict = {
                "id": item.id,
                "title": item.title,
                "type": item.type,
                "description": item.description,
                "acceptance_criteria": item.acceptance_criteria,
                "effort": item.effort,
                "value": item.value,
                "time_criticality": item.time_criticality,
                "risk_reduction": item.risk_reduction,
                "status": item.status,
                "risk_tier": item.risk_tier,
                "created_at": item.created_at,
                "links": item.links
            }
            json.dump(item_dict, f, indent=2)
        created_files.append(item_file)
    
    return created_files


# =============================================================================
# Assertion Helpers
# =============================================================================

@pytest.fixture
def assert_file_exists():
    """Helper to assert file existence."""
    def _assert_file_exists(file_path: Path, should_exist: bool = True):
        if should_exist:
            assert file_path.exists(), f"File {file_path} should exist"
            assert file_path.is_file(), f"{file_path} should be a file"
        else:
            assert not file_path.exists(), f"File {file_path} should not exist"
    
    return _assert_file_exists


@pytest.fixture
def assert_directory_structure():
    """Helper to assert directory structure."""
    def _assert_directory_structure(base_path: Path, expected_dirs: list[str]):
        for dir_name in expected_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"Directory {dir_path} should exist"
            assert dir_path.is_dir(), f"{dir_path} should be a directory"
    
    return _assert_directory_structure


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically clean up test artifacts after each test."""
    yield
    
    # Clean up any test files that might have been created
    test_files = [
        "test_backlog.yml",
        "test_config.json",
        "test_output.json"
    ]
    
    for file_name in test_files:
        if os.path.exists(file_name):
            os.remove(file_name)


# =============================================================================
# Skip Conditions
# =============================================================================

def pytest_runtest_setup(item):
    """Skip tests based on conditions."""
    # Skip GitHub tests if no token is available (and not mocked)
    if item.get_closest_marker("github"):
        if not os.environ.get("GITHUB_TOKEN") and "mock" not in item.name:
            pytest.skip("GitHub token not available")
    
    # Skip LLM tests if no API key is available (and not mocked)
    if item.get_closest_marker("llm"):
        if (not os.environ.get("OPENAI_API_KEY") and 
            not os.environ.get("ANTHROPIC_API_KEY") and 
            "mock" not in item.name):
            pytest.skip("LLM API key not available")
    
    # Skip network tests if explicitly disabled
    if item.get_closest_marker("requires_network"):
        if os.environ.get("SKIP_NETWORK_TESTS", "false").lower() == "true":
            pytest.skip("Network tests disabled")


# =============================================================================
# Test Data Generators
# =============================================================================

@pytest.fixture
def backlog_item_factory():
    """Factory for creating test backlog items."""
    def _create_item(
        id_suffix: str = "001",
        item_type: str = "feature",
        status: str = "READY",
        effort: int = 5,
        value: int = 5,
        **kwargs
    ) -> BacklogItem:
        defaults = {
            "id": f"test-{id_suffix}",
            "title": f"Test {item_type.title()}",
            "type": item_type,
            "description": f"Test {item_type} description",
            "acceptance_criteria": ["Should work correctly"],
            "effort": effort,
            "value": value,
            "time_criticality": 5,
            "risk_reduction": 5,
            "status": status,
            "risk_tier": "medium",
            "created_at": "2025-01-27T00:00:00Z",
            "links": []
        }
        defaults.update(kwargs)
        return BacklogItem(**defaults)
    
    return _create_item