#!/usr/bin/env python3
"""
Comprehensive unit tests for backlog_manager.py
Acceptance Criteria:
- Test coverage >80% for core modules
- Tests for WSJF calculation edge cases
- Tests for discovery deduplication
- Tests for status transitions
- Mock tests for git operations
"""

import pytest
import tempfile
import yaml
import json
import os
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from backlog_manager import BacklogManager, BacklogItem


class TestBacklogItem:
    """Test BacklogItem dataclass"""
    
    def test_backlog_item_creation(self):
        """Test basic BacklogItem creation"""
        item = BacklogItem(
            id="test-1",
            title="Test Item", 
            type="feature",
            description="A test item",
            acceptance_criteria=["Must work"],
            effort=3,
            value=5,
            time_criticality=2,
            risk_reduction=3,
            status="NEW",
            risk_tier="low",
            created_at="2025-01-01T00:00:00Z",
            links=[]
        )
        assert item.id == "test-1"
        assert item.wsjf_score is None
        assert item.aging_multiplier == 1.0


class TestWsjfCalculation:
    """Test WSJF scoring edge cases"""
    
    def test_calculate_wsjf_normal_case(self):
        """Test normal WSJF calculation"""
        # This will fail initially - RED phase
        manager = BacklogManager()
        item = BacklogItem(
            id="test-1", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=5, value=8, time_criticality=3, 
            risk_reduction=2, status="NEW", risk_tier="low", 
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        # Expected: (8 + 3 + 2) / 5 = 2.6
        wsjf = manager.calculate_wsjf(item)
        assert wsjf == 2.6
    
    def test_calculate_wsjf_zero_effort(self):
        """Test WSJF with zero effort (edge case)"""
        manager = BacklogManager()
        item = BacklogItem(
            id="test-1", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=0, value=8, time_criticality=3, 
            risk_reduction=2, status="NEW", risk_tier="low", 
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        # Should handle division by zero
        wsjf = manager.calculate_wsjf(item)
        assert wsjf == float('inf') or wsjf == 0  # Implementation decision
    
    def test_calculate_wsjf_with_aging(self):
        """Test WSJF with aging multiplier"""
        manager = BacklogManager()
        item = BacklogItem(
            id="test-1", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=5, value=8, time_criticality=3, 
            risk_reduction=2, status="NEW", risk_tier="low", 
            created_at="2025-01-01T00:00:00Z", links=[], aging_multiplier=1.5
        )
        wsjf = manager.calculate_wsjf(item)
        # Expected: ((8 + 3 + 2) / 5) * 1.5 = 3.9
        assert abs(wsjf - 3.9) < 0.001  # Handle floating point precision


class TestDiscoveryDeduplication:
    """Test discovery and deduplication logic"""
    
    def test_discover_from_code_comments_no_duplicates(self):
        """Test TODO/FIXME discovery without duplicates"""
        manager = BacklogManager()
        # This will fail initially - need to implement properly
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "file.py:10:# TODO: Fix this bug\nfile.py:10:# TODO: Fix this bug\n"
            mock_run.return_value.returncode = 0
            
            items = manager.discover_from_code_comments()
            # Should deduplicate identical TODOs
            assert len(items) == 1
    
    def test_deduplicate_items(self):
        """Test item deduplication by title and description"""
        manager = BacklogManager()
        items = [
            {"id": "1", "title": "Same Title", "description": "Same desc"},
            {"id": "2", "title": "Same Title", "description": "Same desc"},
            {"id": "3", "title": "Different", "description": "Different"}
        ]
        deduplicated = manager.deduplicate_items(items)
        assert len(deduplicated) == 2


class TestStatusTransitions:
    """Test backlog item status transitions"""
    
    def test_valid_status_transitions(self):
        """Test allowed status transitions"""
        manager = BacklogManager()
        
        # NEW → REFINED
        assert manager.is_valid_transition("NEW", "REFINED")
        # REFINED → READY  
        assert manager.is_valid_transition("REFINED", "READY")
        # READY → DOING
        assert manager.is_valid_transition("READY", "DOING")
        # DOING → PR
        assert manager.is_valid_transition("DOING", "PR")
        # PR → DONE
        assert manager.is_valid_transition("PR", "DONE")
    
    def test_invalid_status_transitions(self):
        """Test disallowed status transitions"""
        manager = BacklogManager()
        
        # Can't skip states
        assert not manager.is_valid_transition("NEW", "DOING")
        # Can't go backwards
        assert not manager.is_valid_transition("DONE", "NEW")
    
    def test_update_item_status(self):
        """Test updating item status"""
        manager = BacklogManager()
        item = BacklogItem(
            id="test-1", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=5, value=8, time_criticality=3, 
            risk_reduction=2, status="NEW", risk_tier="low", 
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        # Valid transition
        success = manager.update_item_status(item, "REFINED")
        assert success
        assert item.status == "REFINED"
        
        # Invalid transition
        success = manager.update_item_status(item, "DONE")
        assert not success
        assert item.status == "REFINED"  # Unchanged


class TestGitOperations:
    """Test git operations with mocks"""
    
    @patch('subprocess.run')
    def test_git_status_clean(self, mock_run):
        """Test git status when clean"""
        mock_run.return_value.stdout = ""
        mock_run.return_value.returncode = 0
        
        manager = BacklogManager()
        is_clean = manager.is_git_clean()
        assert is_clean
    
    @patch('subprocess.run')
    def test_git_status_dirty(self, mock_run):
        """Test git status when dirty"""
        mock_run.return_value.stdout = "M file.py\n"
        mock_run.return_value.returncode = 0
        
        manager = BacklogManager()
        is_clean = manager.is_git_clean()
        assert not is_clean
    
    @patch('subprocess.run')
    def test_create_commit(self, mock_run):
        """Test commit creation"""
        mock_run.return_value.returncode = 0
        
        manager = BacklogManager()
        success = manager.create_commit("Test commit message")
        assert success
        
        # Verify git commands were called
        assert mock_run.call_count >= 2  # add + commit


class TestBacklogManager:
    """Test main BacklogManager functionality"""
    
    def test_load_backlog_file_missing(self):
        """Test loading when backlog file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BacklogManager(temp_dir)
            manager.load_backlog()
            assert len(manager.items) == 0
    
    def test_load_backlog_valid_file(self):
        """Test loading valid backlog file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            backlog_data = {
                'version': '1.0',
                'items': [{
                    'id': 'test-1',
                    'title': 'Test Item',
                    'type': 'feature',
                    'description': 'Test description',
                    'acceptance_criteria': ['Must work'],
                    'effort': 3,
                    'value': 5,
                    'time_criticality': 2,
                    'risk_reduction': 3,
                    'status': 'NEW',
                    'risk_tier': 'low',
                    'created_at': '2025-01-01T00:00:00Z',
                    'links': [],
                    'wsjf_score': 3.33,
                    'aging_multiplier': 1.0
                }]
            }
            
            backlog_file = Path(temp_dir) / "backlog.yml"
            with open(backlog_file, 'w') as f:
                yaml.dump(backlog_data, f)
            
            manager = BacklogManager(temp_dir)
            manager.load_backlog()
            assert len(manager.items) == 1
            assert manager.items[0].id == 'test-1'
    
    def test_save_backlog(self):
        """Test saving backlog to file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BacklogManager(temp_dir)
            item = BacklogItem(
                id="test-1", title="Test", type="feature", description="Test",
                acceptance_criteria=["Must work"], effort=3, value=5, 
                time_criticality=2, risk_reduction=3, status="NEW", 
                risk_tier="low", created_at="2025-01-01T00:00:00Z", 
                links=[], wsjf_score=3.33, aging_multiplier=1.0
            )
            manager.items = [item]
            manager.save_backlog()
            
            # Verify file was created
            backlog_file = Path(temp_dir) / "backlog.yml"
            assert backlog_file.exists()
            
            # Verify content
            with open(backlog_file, 'r') as f:
                data = yaml.safe_load(f)
            assert len(data['items']) == 1
            assert data['items'][0]['id'] == 'test-1'


if __name__ == '__main__':
    # Run with pytest for better output
    import subprocess
    import sys
    
    # First run tests to see failures (RED phase)
    print("🔴 RED PHASE: Running tests expecting failures...")
    result = subprocess.run([sys.executable, '-m', 'pytest', __file__, '-v'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print("❌ Tests failed as expected (RED phase)")
    else:
        print("✅ All tests passed!")