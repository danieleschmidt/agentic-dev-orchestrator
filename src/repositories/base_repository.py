#!/usr/bin/env python3
"""
Base Repository Pattern Implementation
Provides CRUD operations and common functionality for all repositories
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic
from pathlib import Path
from dataclasses import asdict
import logging

from ..database.connection import ConnectionManager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(Generic[T], ABC):
    """Base repository with common CRUD operations"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection = connection_manager
        self.filesystem = connection_manager.filesystem
        self.sqlite = connection_manager.sqlite
    
    @abstractmethod
    def get_storage_path(self) -> Path:
        """Get the storage path for this repository"""
        pass
    
    @abstractmethod
    def serialize_item(self, item: T) -> Dict[str, Any]:
        """Serialize item to dictionary"""
        pass
    
    @abstractmethod
    def deserialize_item(self, data: Dict[str, Any]) -> T:
        """Deserialize dictionary to item"""  
        pass
    
    @abstractmethod
    def get_item_id(self, item: T) -> str:
        """Get unique identifier for item"""
        pass
    
    def save(self, item: T) -> bool:
        """Save item to storage"""
        try:
            item_id = self.get_item_id(item)
            data = self.serialize_item(item)
            
            # Save to filesystem (primary storage)
            file_path = self.get_storage_path() / f"{item_id}.json"
            success = self.filesystem.save_json(data, file_path)
            
            # Optionally save to SQLite for indexing/querying
            if self.sqlite and success:
                self._save_to_sqlite(item, data)
            
            return success
        except Exception as e:
            logger.error(f"Failed to save item {self.get_item_id(item)}: {e}")
            return False
    
    def load(self, item_id: str) -> Optional[T]:
        """Load item by ID"""
        try:
            file_path = self.get_storage_path() / f"{item_id}.json"
            data = self.filesystem.load_json(file_path)
            
            if data:
                return self.deserialize_item(data)
            return None
        except Exception as e:
            logger.error(f"Failed to load item {item_id}: {e}")
            return None
    
    def load_all(self) -> List[T]:
        """Load all items"""
        items = []
        try:
            storage_path = self.get_storage_path()
            json_files = self.filesystem.list_json_files(storage_path)
            
            for file_path in json_files:
                data = self.filesystem.load_json(file_path)
                if data:
                    try:
                        item = self.deserialize_item(data)
                        items.append(item)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to load all items: {e}")
            
        return items
    
    def delete(self, item_id: str) -> bool:
        """Delete item by ID"""
        try:
            file_path = self.get_storage_path() / f"{item_id}.json"
            success = self.filesystem.delete_file(file_path)
            
            # Remove from SQLite if available
            if self.sqlite and success:
                self._delete_from_sqlite(item_id)
            
            return success
        except Exception as e:
            logger.error(f"Failed to delete item {item_id}: {e}")
            return False
    
    def exists(self, item_id: str) -> bool:
        """Check if item exists"""
        file_path = self.get_storage_path() / f"{item_id}.json"
        return file_path.exists()
    
    def list_ids(self) -> List[str]:
        """List all item IDs"""
        try:
            storage_path = self.get_storage_path()
            json_files = self.filesystem.list_json_files(storage_path)
            return [f.stem for f in json_files]
        except Exception as e:
            logger.error(f"Failed to list IDs: {e}")
            return []
    
    def count(self) -> int:
        """Count total items"""
        return len(self.list_ids())
    
    def backup_all(self, backup_path: Path) -> bool:
        """Backup all items to a directory"""
        try:
            backup_path.mkdir(parents=True, exist_ok=True)
            items = self.load_all()
            
            for item in items:
                item_id = self.get_item_id(item)
                data = self.serialize_item(item)
                backup_file = backup_path / f"{item_id}.json"
                
                if not self.filesystem.save_json(data, backup_file):
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def restore_from_backup(self, backup_path: Path) -> int:
        """Restore items from backup directory"""
        restored_count = 0
        try:
            backup_files = list(backup_path.glob("*.json"))
            
            for backup_file in backup_files:
                data = self.filesystem.load_json(backup_file)
                if data:
                    try:
                        item = self.deserialize_item(data)
                        if self.save(item):
                            restored_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to restore {backup_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            
        return restored_count
    
    def _save_to_sqlite(self, item: T, data: Dict[str, Any]):
        """Save item to SQLite (override in subclasses for specific tables)"""
        pass
    
    def _delete_from_sqlite(self, item_id: str):
        """Delete item from SQLite (override in subclasses)"""
        pass


class QueryableRepository(BaseRepository[T]):
    """Extended repository with query capabilities"""
    
    def find_by_field(self, field_name: str, value: Any) -> List[T]:
        """Find items by field value"""
        items = []
        try:
            all_items = self.load_all()
            for item in all_items:
                data = self.serialize_item(item)
                if data.get(field_name) == value:
                    items.append(item)
        except Exception as e:
            logger.error(f"Query by field failed: {e}")
            
        return items
    
    def find_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """Find items matching multiple criteria"""
        items = []
        try:
            all_items = self.load_all()
            for item in all_items:
                data = self.serialize_item(item)
                
                # Check if all criteria match
                matches = True
                for field, expected_value in criteria.items():
                    if data.get(field) != expected_value:
                        matches = False
                        break
                
                if matches:
                    items.append(item)
                    
        except Exception as e:
            logger.error(f"Query by criteria failed: {e}")
            
        return items
    
    def filter_items(self, filter_func) -> List[T]:
        """Filter items using a custom function"""
        try:
            all_items = self.load_all()
            return [item for item in all_items if filter_func(item)]
        except Exception as e:
            logger.error(f"Filter operation failed: {e}")
            return []
    
    def sort_items(self, key_func, reverse: bool = False) -> List[T]:
        """Sort items using a custom key function"""
        try:
            all_items = self.load_all()
            return sorted(all_items, key=key_func, reverse=reverse)
        except Exception as e:
            logger.error(f"Sort operation failed: {e}")
            return []