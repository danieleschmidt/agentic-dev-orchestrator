#!/usr/bin/env python3
"""
Comprehensive unit tests for autonomous_executor.py
"""

import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from autonomous_executor import AutonomousExecutor
from backlog_manager import BacklogManager, BacklogItem


class TestAutonomousExecutor:
    """Test main AutonomousExecutor functionality"""
    
    def test_executor_initialization(self):
        """Test executor initializes properly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = AutonomousExecutor(temp_dir)
            assert executor.repo_root == Path(temp_dir)
            assert isinstance(executor.backlog_manager, BacklogManager)
    
    @patch.object(BacklogManager, 'is_git_clean')
    @patch.object(BacklogManager, 'load_backlog')
    def test_sync_repo_and_ci_clean(self, mock_load, mock_clean):
        """Test sync when repo is clean"""
        mock_clean.return_value = True
        
        executor = AutonomousExecutor()
        result = executor.sync_repo_and_ci()
        assert result
        mock_load.assert_called_once()
    
    @patch.object(BacklogManager, 'is_git_clean')
    def test_sync_repo_and_ci_dirty(self, mock_clean):
        """Test sync when repo is dirty"""
        mock_clean.return_value = False
        
        executor = AutonomousExecutor()
        result = executor.sync_repo_and_ci()
        assert not result
    
    @patch.object(BacklogManager, 'continuous_discovery')
    @patch.object(BacklogManager, 'load_backlog')
    def test_discover_new_tasks(self, mock_load, mock_discover):
        """Test task discovery"""
        mock_discover.return_value = 3
        
        executor = AutonomousExecutor()
        count = executor.discover_new_tasks()
        assert count == 3
        mock_discover.assert_called_once()
    
    @patch.object(BacklogManager, 'get_next_ready_item')
    @patch.object(BacklogManager, 'load_backlog')
    def test_get_next_task_available(self, mock_load, mock_next):
        """Test getting next task when available"""
        test_item = BacklogItem(
            id="test-1", title="Test Task", type="feature", description="Test",
            acceptance_criteria=["Must work"], effort=3, value=5, 
            time_criticality=2, risk_reduction=3, status="READY", 
            risk_tier="low", created_at="2025-01-01T00:00:00Z", links=[]
        )
        mock_next.return_value = test_item
        
        executor = AutonomousExecutor()
        task = executor.get_next_task()
        assert task == test_item
    
    @patch.object(BacklogManager, 'get_next_ready_item')
    @patch.object(BacklogManager, 'load_backlog')
    def test_get_next_task_none_available(self, mock_load, mock_next):
        """Test getting next task when none available"""
        mock_next.return_value = None
        
        executor = AutonomousExecutor()
        task = executor.get_next_task()
        assert task is None
    
    def test_is_high_risk_task_high_risk_tier(self):
        """Test high risk detection for high risk tier"""
        item = BacklogItem(
            id="test-1", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=3, value=5, time_criticality=2, 
            risk_reduction=3, status="READY", risk_tier="high", 
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        executor = AutonomousExecutor()
        assert executor.is_high_risk_task(item)
    
    def test_is_high_risk_task_api_changes(self):
        """Test high risk detection for API changes"""
        item = BacklogItem(
            id="test-1", title="Change public API endpoints", type="feature", 
            description="Modify public API", acceptance_criteria=[], effort=3, 
            value=5, time_criticality=2, risk_reduction=3, status="READY", 
            risk_tier="low", created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        executor = AutonomousExecutor()
        assert executor.is_high_risk_task(item)
    
    def test_is_high_risk_task_safe(self):
        """Test high risk detection for safe tasks"""
        item = BacklogItem(
            id="test-1", title="Add unit tests", type="feature", 
            description="Add more tests", acceptance_criteria=[], effort=3, 
            value=5, time_criticality=2, risk_reduction=3, status="READY", 
            risk_tier="low", created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        executor = AutonomousExecutor()
        assert not executor.is_high_risk_task(item)
    
    @patch('subprocess.run')
    def test_run_tests_success(self, mock_run):
        """Test successful test run"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "All tests passed"
        
        executor = AutonomousExecutor()
        result = executor.run_tests()
        assert result
    
    @patch('subprocess.run')
    def test_run_tests_failure(self, mock_run):
        """Test failed test run"""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = "Tests failed"
        
        executor = AutonomousExecutor()
        result = executor.run_tests()
        assert not result
    
    @patch('subprocess.run')
    def test_run_linting_success(self, mock_run):
        """Test successful linting"""
        mock_run.return_value.returncode = 0
        
        executor = AutonomousExecutor()
        result = executor.run_linting()
        assert result
    
    @patch('subprocess.run')
    def test_run_linting_failure(self, mock_run):
        """Test failed linting"""
        mock_run.return_value.returncode = 1
        
        executor = AutonomousExecutor()
        result = executor.run_linting()
        assert not result
    
    @patch.object(AutonomousExecutor, 'run_tests')
    @patch.object(AutonomousExecutor, 'run_linting')
    def test_ci_gate_pass(self, mock_lint, mock_test):
        """Test CI gate when all checks pass"""
        mock_lint.return_value = True
        mock_test.return_value = True
        
        executor = AutonomousExecutor()
        result = executor.ci_gate()
        assert result
    
    @patch.object(AutonomousExecutor, 'run_tests')
    @patch.object(AutonomousExecutor, 'run_linting')
    def test_ci_gate_fail_tests(self, mock_lint, mock_test):
        """Test CI gate when tests fail"""
        mock_lint.return_value = True
        mock_test.return_value = False
        
        executor = AutonomousExecutor()
        result = executor.ci_gate()
        assert not result
    
    @patch.object(AutonomousExecutor, 'run_tests')
    @patch.object(AutonomousExecutor, 'run_linting')
    def test_ci_gate_fail_linting(self, mock_lint, mock_test):
        """Test CI gate when linting fails"""
        mock_lint.return_value = False
        mock_test.return_value = True
        
        executor = AutonomousExecutor()
        result = executor.ci_gate()
        assert not result
    
    def test_escalate_task(self):
        """Test task escalation"""
        item = BacklogItem(
            id="test-1", title="Complex Task", type="feature", 
            description="Very complex", acceptance_criteria=[], effort=8, 
            value=5, time_criticality=2, risk_reduction=3, status="READY", 
            risk_tier="high", created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = AutonomousExecutor(temp_dir)
            escalation_file = executor.escalate_task(item, "Too complex for automation")
            
            # Check escalation file was created
            assert escalation_file.exists()
            
            # Check content
            with open(escalation_file, 'r') as f:
                data = json.load(f)
            assert data['item_id'] == 'test-1'
            assert data['reason'] == 'Too complex for automation'
    
    def test_execute_micro_cycle_success(self):
        """Test successful micro cycle execution"""
        item = BacklogItem(
            id="test-1", title="Simple Task", type="feature", 
            description="Simple task", acceptance_criteria=["Must work"], 
            effort=2, value=5, time_criticality=2, risk_reduction=3, 
            status="READY", risk_tier="low", created_at="2025-01-01T00:00:00Z", 
            links=[]
        )
        
        executor = AutonomousExecutor()
        
        with patch.object(executor, 'ci_gate', return_value=True), \
             patch.object(executor.backlog_manager, 'update_item_status', return_value=True) as mock_update, \
             patch.object(executor.backlog_manager, 'update_item_status_by_id', return_value=True) as mock_update_by_id, \
             patch.object(executor.backlog_manager, 'save_backlog') as mock_save:
            result = executor.execute_micro_cycle(item)
            assert result
            # Either method might be called depending on implementation
            assert mock_update.called or mock_update_by_id.called
            mock_save.assert_called()
    
    def test_execute_micro_cycle_ci_failure(self):
        """Test micro cycle with CI failure"""
        item = BacklogItem(
            id="test-1", title="Task", type="feature", description="Task", 
            acceptance_criteria=[], effort=2, value=5, time_criticality=2, 
            risk_reduction=3, status="READY", risk_tier="low", 
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        executor = AutonomousExecutor()
        
        with patch.object(executor, 'ci_gate', return_value=False), \
             patch.object(executor.backlog_manager, 'update_item_status', return_value=True):
            result = executor.execute_micro_cycle(item)
            assert not result


if __name__ == '__main__':
    # Run with pytest
    import subprocess
    import sys
    
    print("🔴 RED PHASE: Running autonomous_executor tests...")
    result = subprocess.run([sys.executable, '-m', 'pytest', __file__, '-v'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print("❌ Tests failed as expected (RED phase)")
    else:
        print("✅ All tests passed!")