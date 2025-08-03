#!/usr/bin/env python3
"""
Backlog Repository Implementation
Handles persistence and querying of backlog items
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .base_repository import QueryableRepository
from ..database.connection import ConnectionManager

# Import BacklogItem from the main module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backlog_manager import BacklogItem


class BacklogRepository(QueryableRepository[BacklogItem]):
    """Repository for backlog items with WSJF-specific operations"""
    
    def __init__(self, connection_manager: ConnectionManager):
        super().__init__(connection_manager)
        self.ensure_storage_exists()
    
    def get_storage_path(self) -> Path:
        """Get storage path for backlog items"""
        return self.filesystem.backlog_dir
    
    def serialize_item(self, item: BacklogItem) -> Dict[str, Any]:
        """Convert BacklogItem to dictionary"""
        return asdict(item)
    
    def deserialize_item(self, data: Dict[str, Any]) -> BacklogItem:
        """Convert dictionary to BacklogItem"""
        return BacklogItem(**data)
    
    def get_item_id(self, item: BacklogItem) -> str:
        """Get unique identifier for backlog item"""
        return item.id
    
    def ensure_storage_exists(self):
        """Ensure storage directories exist"""
        self.get_storage_path().mkdir(parents=True, exist_ok=True)
    
    def get_by_status(self, status: str) -> List[BacklogItem]:
        """Get all items with specific status"""
        return self.find_by_field('status', status)
    
    def get_ready_items(self) -> List[BacklogItem]:
        """Get all READY items"""
        return self.get_by_status('READY')
    
    def get_by_type(self, item_type: str) -> List[BacklogItem]:
        """Get all items of specific type"""
        return self.find_by_field('type', item_type)
    
    def get_by_risk_tier(self, risk_tier: str) -> List[BacklogItem]:
        """Get all items with specific risk tier"""
        return self.find_by_field('risk_tier', risk_tier)
    
    def get_high_effort_items(self, effort_threshold: int = 8) -> List[BacklogItem]:
        """Get items with effort above threshold"""
        return self.filter_items(lambda item: item.effort >= effort_threshold)
    
    def get_aged_items(self, days_threshold: int = 30) -> List[BacklogItem]:
        """Get items older than threshold"""
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        def is_aged(item: BacklogItem) -> bool:
            try:
                created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
                return created < cutoff_date
            except:
                return False
        
        return self.filter_items(is_aged)
    
    def get_prioritized_items(self) -> List[BacklogItem]:
        """Get items sorted by WSJF score (highest first)"""
        def get_wsjf_score(item: BacklogItem) -> float:
            if item.wsjf_score is not None:
                return item.wsjf_score
            # Calculate WSJF if not set
            cost_of_delay = item.value + item.time_criticality + item.risk_reduction
            return (cost_of_delay / max(item.effort, 1)) * item.aging_multiplier
        
        return self.sort_items(get_wsjf_score, reverse=True)
    
    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get the highest priority READY item"""
        ready_items = self.get_ready_items()
        if not ready_items:
            return None
        
        # Calculate WSJF scores and return highest
        def get_wsjf_score(item: BacklogItem) -> float:
            cost_of_delay = item.value + item.time_criticality + item.risk_reduction
            return (cost_of_delay / max(item.effort, 1)) * item.aging_multiplier
        
        prioritized = sorted(ready_items, key=get_wsjf_score, reverse=True)
        return prioritized[0]
    
    def update_status(self, item_id: str, new_status: str) -> bool:
        """Update item status"""
        item = self.load(item_id)
        if not item:
            return False
        
        # Validate status transition
        valid_transitions = {
            'NEW': ['REFINED', 'BLOCKED'],
            'REFINED': ['READY', 'NEW', 'BLOCKED'],
            'READY': ['DOING', 'REFINED', 'BLOCKED'],
            'DOING': ['PR', 'READY', 'BLOCKED'],
            'PR': ['DONE', 'DOING', 'BLOCKED'],
            'DONE': [],  # Terminal state
            'BLOCKED': ['NEW', 'REFINED', 'READY', 'DOING', 'PR']
        }
        
        if new_status not in valid_transitions.get(item.status, []):
            return False
        
        item.status = new_status
        return self.save(item)
    
    def update_wsjf_scores(self) -> int:
        """Update WSJF scores for all items and return count updated"""
        items = self.load_all()
        updated_count = 0
        
        for item in items:
            # Calculate base WSJF score
            cost_of_delay = item.value + item.time_criticality + item.risk_reduction
            base_score = cost_of_delay / max(item.effort, 1)
            
            # Apply aging multiplier
            try:
                created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
                age_days = (datetime.now() - created).days
                aging_factor = min(1 + (age_days * 0.01), 2.0)  # Max 2x multiplier
                item.aging_multiplier = aging_factor
            except:
                item.aging_multiplier = 1.0
            
            item.wsjf_score = base_score * item.aging_multiplier
            
            if self.save(item):
                updated_count += 1
        
        return updated_count
    
    def get_status_summary(self) -> Dict[str, int]:
        """Get count of items by status"""
        items = self.load_all()
        summary = {}
        
        for item in items:
            summary[item.status] = summary.get(item.status, 0) + 1
        
        return summary
    
    def get_type_summary(self) -> Dict[str, int]:
        """Get count of items by type"""
        items = self.load_all()
        summary = {}
        
        for item in items:
            summary[item.type] = summary.get(item.type, 0) + 1
        
        return summary
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        items = self.load_all()
        
        if not items:
            return {
                'total_items': 0,
                'status_breakdown': {},
                'type_breakdown': {},
                'average_wsjf': 0,
                'average_effort': 0,
                'average_age_days': 0
            }
        
        # Calculate metrics
        total_wsjf = sum(item.wsjf_score or 0 for item in items)
        total_effort = sum(item.effort for item in items)
        
        # Calculate average age
        total_age_days = 0
        valid_ages = 0
        for item in items:
            try:
                created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
                age_days = (datetime.now() - created).days
                total_age_days += age_days
                valid_ages += 1
            except:
                pass
        
        return {
            'total_items': len(items),
            'status_breakdown': self.get_status_summary(),
            'type_breakdown': self.get_type_summary(),
            'average_wsjf': total_wsjf / len(items) if items else 0,
            'average_effort': total_effort / len(items) if items else 0,
            'average_age_days': total_age_days / valid_ages if valid_ages > 0 else 0,
            'ready_items': len(self.get_ready_items()),
            'high_effort_items': len(self.get_high_effort_items()),
            'aged_items': len(self.get_aged_items())
        }
    
    def _save_to_sqlite(self, item: BacklogItem, data: Dict[str, Any]):
        """Save backlog item to SQLite for indexing"""
        if not self.sqlite:
            return
        
        query = '''
            INSERT OR REPLACE INTO backlog_items 
            (id, title, type, description, status, effort, value, time_criticality, 
             risk_reduction, risk_tier, wsjf_score, aging_multiplier, created_at, 
             updated_at, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            item.id, item.title, item.type, item.description, item.status,
            item.effort, item.value, item.time_criticality, item.risk_reduction,
            item.risk_tier, item.wsjf_score, item.aging_multiplier,
            item.created_at, datetime.now().isoformat(), json.dumps(data)
        )
        
        self.sqlite.execute_update(query, params)
    
    def _delete_from_sqlite(self, item_id: str):
        """Delete backlog item from SQLite"""
        if not self.sqlite:
            return
        
        self.sqlite.execute_update(
            "DELETE FROM backlog_items WHERE id = ?",
            (item_id,)
        )