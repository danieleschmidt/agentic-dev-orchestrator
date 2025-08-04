#!/usr/bin/env python3
"""
Database Connection Management
Provides file-based and optional database connections for ADO
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class FileSystemConnection:
    """File-based storage connection for ADO - primary storage mechanism"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.backlog_file = self.base_path / "backlog.yml"
        self.backlog_dir = self.base_path / "backlog"
        self.status_dir = self.base_path / "docs" / "status"
        self.escalations_dir = self.base_path / "escalations"
        
        # Ensure directories exist
        self.backlog_dir.mkdir(exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.escalations_dir.mkdir(exist_ok=True)
    
    def save_json(self, data: Dict[Any, Any], file_path: Path) -> bool:
        """Save data to JSON file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    def load_json(self, file_path: Path) -> Optional[Dict[Any, Any]]:
        """Load data from JSON file"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None
    
    def list_json_files(self, directory: Path) -> List[Path]:
        """List all JSON files in directory"""
        if not directory.exists():
            return []
        return list(directory.glob("*.json"))
    
    def delete_file(self, file_path: Path) -> bool:
        """Delete a file"""
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False


class SQLiteConnection:
    """Optional SQLite connection for advanced features and caching"""
    
    def __init__(self, db_path: str = ".ado/ado.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self._initialized = False
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
            if not self._initialized:
                self._create_tables()
                self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Backlog items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backlog_items (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                effort INTEGER,
                value INTEGER,  
                time_criticality INTEGER,
                risk_reduction INTEGER,
                risk_tier TEXT,
                wsjf_score REAL,
                aging_multiplier REAL,
                created_at TEXT,
                updated_at TEXT,
                data_json TEXT
            )
        ''')
        
        # Execution history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                artifacts TEXT,
                FOREIGN KEY (item_id) REFERENCES backlog_items (id)
            )
        ''')
        
        # Metrics table  
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT
            )
        ''')
        
        # Agent performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT,
                item_id TEXT,
                execution_time REAL,
                success BOOLEAN,
                timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        self.connection.commit()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def execute_update(self, query: str, params: tuple = ()) -> bool:
        """Execute an UPDATE/INSERT/DELETE query"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            return False
    
    def get_metrics(self, metric_name: str, limit: int = 100) -> List[Dict]:
        """Get recent metrics"""
        rows = self.execute_query(
            "SELECT * FROM metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT ?",
            (metric_name, limit)
        )
        return [dict(row) for row in rows]
    
    def save_metric(self, metric_name: str, value: float, metadata: Dict = None) -> bool:
        """Save a metric value"""
        from datetime import datetime
        return self.execute_update(
            "INSERT INTO metrics (timestamp, metric_name, metric_value, metadata) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), metric_name, value, json.dumps(metadata or {}))
        )


class ConnectionManager:
    """Manages both file-based and database connections"""
    
    def __init__(self, base_path: str = ".", use_sqlite: bool = False):
        self.base_path = Path(base_path)
        self.filesystem = FileSystemConnection(base_path)
        self.sqlite = SQLiteConnection(f"{base_path}/.ado/ado.db") if use_sqlite else None
        
        if self.sqlite:
            self.sqlite.connect()
    
    def close(self):
        """Close all connections"""
        if self.sqlite:
            self.sqlite.disconnect()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_connection(base_path: str = ".", use_sqlite: bool = False) -> ConnectionManager:
    """Factory function to get connection manager"""
    return ConnectionManager(base_path, use_sqlite)