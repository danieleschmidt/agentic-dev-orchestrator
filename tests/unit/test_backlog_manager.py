#!/usr/bin/env python3
"""
Unit tests for BacklogManager
"""

import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from backlog_manager import BacklogManager, BacklogItem


class TestBacklogManager:
    """Test cases for BacklogManager class."""
    
    def test_initialization(self, temp_workspace):
        """Test BacklogManager initialization."""
        manager = BacklogManager(str(temp_workspace))
        
        assert manager.repo_root == temp_workspace
        assert manager.backlog_file == temp_workspace / "backlog.yml"
        assert manager.backlog_dir == temp_workspace / "backlog"
        assert manager.status_dir == temp_workspace / "docs" / "status"
        assert isinstance(manager.items, list)
        assert isinstance(manager.config, dict)
    
    def test_load_backlog_with_existing_file(
        self, 
        backlog_manager, 
        create_test_backlog_file,
        sample_backlog_config
    ):
        """Test loading backlog from existing file."""
        backlog_manager.load_backlog()
        
        assert backlog_manager.config["name"] == sample_backlog_config["name"]
        assert backlog_manager.config["version"] == sample_backlog_config["version"]
    
    def test_load_backlog_items(
        self, 
        backlog_manager, 
        create_test_backlog_items,
        sample_backlog_items
    ):
        """Test loading individual backlog items."""
        backlog_manager.load_backlog()
        
        assert len(backlog_manager.items) == len(sample_backlog_items)
        
        # Check that items are loaded correctly
        loaded_ids = {item.id for item in backlog_manager.items}
        expected_ids = {item.id for item in sample_backlog_items}
        assert loaded_ids == expected_ids
    
    def test_calculate_wsjf_score(self, sample_backlog_item):
        """Test WSJF score calculation."""
        # Calculate manually: (8 + 6 + 4) / 5 = 3.6
        expected_score = (8 + 6 + 4) / 5
        
        calculated_score = BacklogManager._calculate_wsjf_score(
            sample_backlog_item.value,
            sample_backlog_item.time_criticality, 
            sample_backlog_item.risk_reduction,
            sample_backlog_item.effort
        )
        
        assert calculated_score == expected_score
    
    def test_wsjf_score_with_zero_effort(self):
        """Test WSJF calculation with zero effort (should handle gracefully)."""
        score = BacklogManager._calculate_wsjf_score(10, 10, 10, 0)
        assert score == float('inf')
    
    def test_prioritize_items(self, backlog_manager, sample_backlog_items):
        """Test item prioritization by WSJF score."""
        backlog_manager.items = sample_backlog_items
        
        # Calculate WSJF scores manually
        for item in backlog_manager.items:
            item.wsjf_score = BacklogManager._calculate_wsjf_score(
                item.value, item.time_criticality, item.risk_reduction, item.effort
            )
        
        prioritized = backlog_manager.get_prioritized_items()
        
        # Should be sorted by WSJF score descending
        assert len(prioritized) > 0
        for i in range(len(prioritized) - 1):
            assert prioritized[i].wsjf_score >= prioritized[i + 1].wsjf_score
    
    def test_get_ready_items(self, backlog_manager, sample_backlog_items):
        """Test filtering for ready items."""
        backlog_manager.items = sample_backlog_items
        
        ready_items = backlog_manager.get_ready_items()
        
        # Only items with status "READY" should be returned
        for item in ready_items:
            assert item.status == "READY"
        
        # Should include the specific ready items from our sample
        ready_ids = {item.id for item in ready_items}
        assert "item-001" in ready_ids
        assert "item-002" in ready_ids
        assert "item-003" not in ready_ids  # This one is "NEW"
    
    def test_save_backlog(self, temp_workspace, sample_backlog_config):
        """Test saving backlog configuration."""
        manager = BacklogManager(str(temp_workspace))
        manager.config = sample_backlog_config
        
        manager.save_backlog()
        
        # Check that file was created
        backlog_file = temp_workspace / "backlog.yml"
        assert backlog_file.exists()
        
        # Check content
        with open(backlog_file) as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config["name"] == sample_backlog_config["name"]
        assert saved_config["version"] == sample_backlog_config["version"]
    
    def test_generate_status_report(self, backlog_manager, sample_backlog_items):
        """Test status report generation."""
        backlog_manager.items = sample_backlog_items
        
        report = backlog_manager.generate_status_report()
        
        assert "total_items" in report
        assert "ready_items" in report
        assert "backlog_size_by_status" in report
        assert "wsjf_snapshot" in report
        
        assert report["total_items"] == len(sample_backlog_items)
        assert report["ready_items"] == 2  # item-001 and item-002
        
        # Check status breakdown
        status_breakdown = report["backlog_size_by_status"]
        assert status_breakdown["READY"] == 2
        assert status_breakdown["NEW"] == 1
    
    def test_continuous_discovery(self, backlog_manager):
        """Test continuous discovery functionality."""
        # Mock the discovery process
        with patch.object(backlog_manager, '_discover_from_code') as mock_discover:
            mock_discover.return_value = [
                {
                    "id": "discovered-001",
                    "title": "Discovered Issue",
                    "type": "bug",
                    "description": "Found in code analysis"
                }
            ]
            
            new_count = backlog_manager.continuous_discovery()
            
            assert new_count == 1
            assert len(backlog_manager.items) == 1
            assert backlog_manager.items[0].id == "discovered-001"
    
    def test_aging_multiplier_application(self, backlog_manager):
        """Test aging multiplier application to WSJF scores."""
        from datetime import datetime, timedelta
        
        # Create an old item
        old_item = BacklogItem(
            id="old-001",
            title="Old Item",
            type="feature",
            description="An old item",
            acceptance_criteria=["Should work"],
            effort=5,
            value=5,
            time_criticality=5,
            risk_reduction=5,
            status="READY",
            risk_tier="medium",
            created_at=(datetime.now() - timedelta(days=30)).isoformat(),
            links=[]
        )
        
        backlog_manager.items = [old_item]
        backlog_manager.config = {
            "settings": {
                "wsjf": {
                    "enable_aging": True,
                    "aging_multiplier": 1.05,
                    "max_aging_multiplier": 2.0
                }
            }
        }
        
        prioritized = backlog_manager.get_prioritized_items()
        
        # Aging should have been applied
        assert prioritized[0].aging_multiplier > 1.0
        assert prioritized[0].wsjf_score > BacklogManager._calculate_wsjf_score(5, 5, 5, 5)


class TestBacklogItem:
    """Test cases for BacklogItem dataclass."""
    
    def test_backlog_item_creation(self, sample_backlog_item):
        """Test BacklogItem creation and properties."""
        item = sample_backlog_item
        
        assert item.id == "test-001"
        assert item.title == "Test Feature Implementation"
        assert item.type == "feature"
        assert item.status == "READY"
        assert item.effort == 5
        assert item.value == 8
        assert len(item.acceptance_criteria) == 3
    
    def test_backlog_item_validation(self):
        """Test BacklogItem validation."""
        # Test with valid data
        item = BacklogItem(
            id="valid-001",
            title="Valid Item",
            type="feature",
            description="Valid description",
            acceptance_criteria=["Criteria 1"],
            effort=5,
            value=8,
            time_criticality=6,
            risk_reduction=4,
            status="NEW",
            risk_tier="low",
            created_at="2025-01-27T00:00:00Z",
            links=[]
        )
        
        assert item.id == "valid-001"
        assert item.wsjf_score is None  # Not calculated yet
        assert item.aging_multiplier == 1.0  # Default value


@pytest.mark.integration
class TestBacklogManagerIntegration:
    """Integration tests for BacklogManager."""
    
    def test_full_workflow(
        self, 
        temp_workspace, 
        sample_backlog_config,
        sample_backlog_items
    ):
        """Test complete BacklogManager workflow."""
        # Initialize manager
        manager = BacklogManager(str(temp_workspace))
        
        # Set configuration
        manager.config = sample_backlog_config
        manager.items = sample_backlog_items
        
        # Save and reload
        manager.save_backlog()
        
        # Create a new manager instance
        new_manager = BacklogManager(str(temp_workspace))
        new_manager.load_backlog()
        
        # Verify data persistence
        assert new_manager.config["name"] == sample_backlog_config["name"]
        
        # Test prioritization
        prioritized = new_manager.get_prioritized_items()
        assert len(prioritized) > 0
        
        # Test status report
        report = new_manager.generate_status_report()
        assert report["total_items"] > 0
        assert "wsjf_snapshot" in report