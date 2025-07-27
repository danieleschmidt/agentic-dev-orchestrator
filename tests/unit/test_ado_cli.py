#!/usr/bin/env python3
"""
Unit tests for ADO CLI module
"""

import pytest
import sys
from io import StringIO
from unittest.mock import Mock, patch, call

import ado


class TestADOCLI:
    """Test cases for ADO CLI functionality."""
    
    def test_cmd_help(self, capsys):
        """Test help command output."""
        ado.cmd_help()
        
        captured = capsys.readouterr()
        assert "Autonomous Development Orchestrator" in captured.out
        assert "Commands:" in captured.out
        assert "init" in captured.out
        assert "run" in captured.out
        assert "status" in captured.out
        assert "discover" in captured.out
    
    @patch('ado.BacklogManager')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    def test_cmd_init(self, mock_exists, mock_mkdir, mock_backlog_manager, capsys):
        """Test initialization command."""
        mock_exists.return_value = False
        mock_manager = Mock()
        mock_backlog_manager.return_value = mock_manager
        
        ado.cmd_init()
        
        captured = capsys.readouterr()
        assert "Initializing Autonomous Development Orchestrator" in captured.out
        assert "initialization complete" in captured.out
        
        # Verify directories were created
        assert mock_mkdir.called
        
        # Verify backlog manager was used
        mock_manager.save_backlog.assert_called_once()
    
    @patch('ado.AutonomousExecutor')
    def test_cmd_run(self, mock_executor, capsys):
        """Test run command."""
        mock_executor_instance = Mock()
        mock_executor.return_value = mock_executor_instance
        
        # Mock execution results
        mock_results = {
            'completed_items': ['item1', 'item2'],
            'blocked_items': ['item3'],
            'escalated_items': ['item4']
        }
        mock_executor_instance.macro_execution_loop.return_value = mock_results
        
        ado.cmd_run()
        
        captured = capsys.readouterr()
        assert "Starting autonomous backlog execution" in captured.out
        assert "Execution Summary" in captured.out
        assert "Completed items: 2" in captured.out
        assert "Blocked items: 1" in captured.out
        assert "Escalated items: 1" in captured.out
    
    @patch('ado.BacklogManager')
    def test_cmd_status(self, mock_backlog_manager, capsys):
        """Test status command."""
        mock_manager = Mock()
        mock_backlog_manager.return_value = mock_manager
        
        # Mock status report
        mock_report = {
            'total_items': 5,
            'ready_items': 3,
            'backlog_size_by_status': {
                'NEW': 1,
                'READY': 3,
                'DOING': 1
            },
            'wsjf_snapshot': {
                'top_3_ready': [
                    {'id': 'item1', 'title': 'Feature 1', 'wsjf_score': 8.5},
                    {'id': 'item2', 'title': 'Feature 2', 'wsjf_score': 7.2},
                    {'id': 'item3', 'title': 'Feature 3', 'wsjf_score': 6.1}
                ]
            }
        }
        mock_manager.generate_status_report.return_value = mock_report
        
        ado.cmd_status()
        
        captured = capsys.readouterr()
        assert "Backlog Status" in captured.out
        assert "Total items: 5" in captured.out
        assert "Ready items: 3" in captured.out
        assert "NEW: 1" in captured.out
        assert "READY: 3" in captured.out
        assert "Top 3 ready items" in captured.out
        assert "item1: Feature 1 (WSJF: 8.50)" in captured.out
    
    @patch('ado.BacklogManager')
    def test_cmd_discover(self, mock_backlog_manager, capsys):
        """Test discover command."""
        mock_manager = Mock()
        mock_backlog_manager.return_value = mock_manager
        mock_manager.continuous_discovery.return_value = 3
        
        ado.cmd_discover()
        
        captured = capsys.readouterr()
        assert "Running backlog discovery" in captured.out
        assert "Discovered 3 new items" in captured.out
        
        # Verify manager methods were called
        mock_manager.load_backlog.assert_called_once()
        mock_manager.continuous_discovery.assert_called_once()
        mock_manager.save_backlog.assert_called_once()
    
    def test_main_with_no_args(self, capsys):
        """Test main function with no arguments."""
        with patch.object(sys, 'argv', ['ado.py']):
            ado.main()
        
        captured = capsys.readouterr()
        assert "Autonomous Development Orchestrator" in captured.out
    
    @patch('ado.cmd_init')
    def test_main_with_init_command(self, mock_cmd_init):
        """Test main function with init command."""
        with patch.object(sys, 'argv', ['ado.py', 'init']):
            ado.main()
        
        mock_cmd_init.assert_called_once()
    
    @patch('ado.cmd_run')
    def test_main_with_run_command(self, mock_cmd_run):
        """Test main function with run command."""
        with patch.object(sys, 'argv', ['ado.py', 'run']):
            ado.main()
        
        mock_cmd_run.assert_called_once()
    
    @patch('ado.cmd_status')
    def test_main_with_status_command(self, mock_cmd_status):
        """Test main function with status command."""
        with patch.object(sys, 'argv', ['ado.py', 'status']):
            ado.main()
        
        mock_cmd_status.assert_called_once()
    
    @patch('ado.cmd_discover')
    def test_main_with_discover_command(self, mock_cmd_discover):
        """Test main function with discover command."""
        with patch.object(sys, 'argv', ['ado.py', 'discover']):
            ado.main()
        
        mock_cmd_discover.assert_called_once()
    
    @patch('ado.cmd_help')
    def test_main_with_help_command(self, mock_cmd_help):
        """Test main function with help command."""
        with patch.object(sys, 'argv', ['ado.py', 'help']):
            ado.main()
        
        mock_cmd_help.assert_called_once()
    
    @patch('ado.cmd_help')
    def test_main_with_help_flag(self, mock_cmd_help):
        """Test main function with --help flag."""
        with patch.object(sys, 'argv', ['ado.py', '--help']):
            ado.main()
        
        mock_cmd_help.assert_called_once()
    
    def test_main_with_unknown_command(self, capsys):
        """Test main function with unknown command."""
        with patch.object(sys, 'argv', ['ado.py', 'unknown']):
            ado.main()
        
        captured = capsys.readouterr()
        assert "Unknown command: unknown" in captured.out
        assert "Run 'python ado.py help'" in captured.out
    
    @patch('ado.cmd_run')
    def test_main_keyboard_interrupt(self, mock_cmd_run, capsys):
        """Test main function handling keyboard interrupt."""
        mock_cmd_run.side_effect = KeyboardInterrupt()
        
        with patch.object(sys, 'argv', ['ado.py', 'run']):
            ado.main()
        
        captured = capsys.readouterr()
        assert "Operation cancelled" in captured.out
    
    @patch('ado.cmd_run')
    def test_main_exception_handling(self, mock_cmd_run, capsys):
        """Test main function handling general exceptions."""
        mock_cmd_run.side_effect = Exception("Test error")
        
        with patch.object(sys, 'argv', ['ado.py', 'run']):
            ado.main()
        
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.out


class TestCLIHelpers:
    """Test helper functions and utilities for CLI."""
    
    def test_command_mapping(self):
        """Test that all expected commands are mapped."""
        # This would test a commands dictionary if it existed
        # For now, we verify the commands exist as functions
        assert hasattr(ado, 'cmd_init')
        assert hasattr(ado, 'cmd_run')
        assert hasattr(ado, 'cmd_status')
        assert hasattr(ado, 'cmd_discover')
        assert hasattr(ado, 'cmd_help')
        assert hasattr(ado, 'main')
    
    def test_cli_functions_are_callable(self):
        """Test that CLI functions are callable."""
        assert callable(ado.cmd_init)
        assert callable(ado.cmd_run)
        assert callable(ado.cmd_status)
        assert callable(ado.cmd_discover)
        assert callable(ado.cmd_help)
        assert callable(ado.main)


@pytest.mark.integration 
class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    @patch('ado.BacklogManager')
    @patch('ado.AutonomousExecutor')
    def test_full_cli_workflow(
        self, 
        mock_executor, 
        mock_backlog_manager,
        temp_workspace
    ):
        """Test complete CLI workflow."""
        # Setup mocks
        mock_manager = Mock()
        mock_backlog_manager.return_value = mock_manager
        
        mock_executor_instance = Mock()
        mock_executor.return_value = mock_executor_instance
        mock_executor_instance.macro_execution_loop.return_value = {
            'completed_items': [],
            'blocked_items': [],
            'escalated_items': []
        }
        
        # Test init -> status -> run workflow
        with patch.object(sys, 'argv', ['ado.py', 'init']):
            ado.main()
        
        with patch.object(sys, 'argv', ['ado.py', 'status']):
            ado.main()
        
        with patch.object(sys, 'argv', ['ado.py', 'run']):
            ado.main()
        
        # Verify interactions
        assert mock_manager.save_backlog.called
        assert mock_manager.generate_status_report.called
        assert mock_executor_instance.macro_execution_loop.called