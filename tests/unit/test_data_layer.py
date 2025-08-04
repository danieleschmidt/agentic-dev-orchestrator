#!/usr/bin/env python3
"""
Tests for data layer components
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.connection import FileSystemConnection, SQLiteConnection, ConnectionManager
from src.repositories.backlog_repository import BacklogRepository
from src.cache.cache_manager import InMemoryCache, FileCache, CacheManager
from backlog_manager import BacklogItem


class TestFileSystemConnection:
    """Test file system connection"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def fs_connection(self, temp_dir):
        return FileSystemConnection(str(temp_dir))
    
    def test_initialization(self, fs_connection, temp_dir):
        """Test filesystem connection initialization"""
        assert fs_connection.base_path == temp_dir
        assert fs_connection.backlog_dir.exists()
        assert fs_connection.status_dir.exists()
        assert fs_connection.escalations_dir.exists()
    
    def test_save_and_load_json(self, fs_connection):
        """Test JSON save and load operations"""
        test_data = {'key': 'value', 'number': 42}
        test_file = fs_connection.base_path / 'test.json'
        
        # Save data
        assert fs_connection.save_json(test_data, test_file)
        assert test_file.exists()
        
        # Load data
        loaded_data = fs_connection.load_json(test_file)
        assert loaded_data == test_data
    
    def test_list_json_files(self, fs_connection):
        """Test listing JSON files"""
        # Create test files
        test_dir = fs_connection.base_path / 'test_dir'
        test_dir.mkdir()
        
        for i in range(3):
            test_file = test_dir / f'test{i}.json'
            fs_connection.save_json({'id': i}, test_file)
        
        # List files
        json_files = fs_connection.list_json_files(test_dir)
        assert len(json_files) == 3
        assert all(f.suffix == '.json' for f in json_files)
    
    def test_delete_file(self, fs_connection):
        """Test file deletion"""
        test_file = fs_connection.base_path / 'test.json'
        fs_connection.save_json({'test': True}, test_file)
        
        assert test_file.exists()
        assert fs_connection.delete_file(test_file)
        assert not test_file.exists()


class TestBacklogRepository:
    """Test backlog repository"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.fixture
    def connection_manager(self, temp_dir):
        return ConnectionManager(str(temp_dir))
    
    @pytest.fixture
    def repository(self, connection_manager):
        return BacklogRepository(connection_manager)
    
    @pytest.fixture
    def sample_item(self):
        return BacklogItem(
            id='test-001',
            title='Test Item',
            type='feature',
            description='A test backlog item',
            acceptance_criteria=['Should work correctly'],
            effort=5,
            value=8,
            time_criticality=6,
            risk_reduction=4,
            status='NEW',
            risk_tier='medium',
            created_at=datetime.now().isoformat() + 'Z',
            links=[],
            aging_multiplier=1.0
        )
    
    def test_save_and_load_item(self, repository, sample_item):
        """Test saving and loading backlog items"""
        # Save item
        assert repository.save(sample_item)
        
        # Load item
        loaded_item = repository.load(sample_item.id)
        assert loaded_item is not None
        assert loaded_item.id == sample_item.id
        assert loaded_item.title == sample_item.title
        assert loaded_item.status == sample_item.status
    
    def test_load_all_items(self, repository):
        """Test loading all items"""
        # Create multiple items
        items = []
        for i in range(3):
            item = BacklogItem(
                id=f'test-{i:03d}',
                title=f'Test Item {i}',
                type='feature',
                description=f'Test item {i}',
                acceptance_criteria=[f'Criteria {i}'],
                effort=i + 1,
                value=5,
                time_criticality=3,
                risk_reduction=2,
                status='NEW',
                risk_tier='low',
                created_at=datetime.now().isoformat() + 'Z',
                links=[],
                aging_multiplier=1.0
            )
            items.append(item)
            repository.save(item)
        
        # Load all
        loaded_items = repository.load_all()
        assert len(loaded_items) == 3
        
        loaded_ids = {item.id for item in loaded_items}
        expected_ids = {item.id for item in items}
        assert loaded_ids == expected_ids
    
    def test_get_by_status(self, repository):
        """Test querying by status"""
        # Create items with different statuses
        statuses = ['NEW', 'READY', 'DOING']
        for i, status in enumerate(statuses):
            item = BacklogItem(
                id=f'test-{status.lower()}',
                title=f'Item {status}',
                type='feature',
                description=f'Item with {status} status',
                acceptance_criteria=['Test criteria'],
                effort=3,
                value=5,
                time_criticality=3,
                risk_reduction=2,
                status=status,
                risk_tier='low',
                created_at=datetime.now().isoformat() + 'Z',
                links=[],
                aging_multiplier=1.0
            )
            repository.save(item)
        
        # Query by status
        ready_items = repository.get_by_status('READY')
        assert len(ready_items) == 1
        assert ready_items[0].status == 'READY'
    
    def test_update_status(self, repository, sample_item):
        """Test status updates with validation"""
        repository.save(sample_item)
        
        # Valid transition: NEW -> REFINED
        assert repository.update_status(sample_item.id, 'REFINED')
        updated_item = repository.load(sample_item.id)
        assert updated_item.status == 'REFINED'
        
        # Invalid transition: REFINED -> DONE (should go through READY -> DOING -> PR -> DONE)
        assert not repository.update_status(sample_item.id, 'DONE')
        # Status should remain REFINED
        updated_item = repository.load(sample_item.id)
        assert updated_item.status == 'REFINED'
    
    def test_get_prioritized_items(self, repository):
        """Test WSJF prioritization"""
        # Create items with different WSJF scores
        items = [
            BacklogItem(
                id='high-priority',
                title='High Priority',
                type='feature',
                description='High value, low effort',
                acceptance_criteria=['High priority criteria'],
                effort=2,  # Low effort
                value=9,   # High value
                time_criticality=8,
                risk_reduction=7,
                status='READY',
                risk_tier='low',
                created_at=datetime.now().isoformat() + 'Z',
                links=[],
                aging_multiplier=1.0
            ),
            BacklogItem(
                id='low-priority',
                title='Low Priority',
                type='feature',
                description='Low value, high effort',
                acceptance_criteria=['Low priority criteria'],
                effort=8,  # High effort
                value=2,   # Low value
                time_criticality=1,
                risk_reduction=1,
                status='READY',
                risk_tier='low',
                created_at=datetime.now().isoformat() + 'Z',
                links=[],
                aging_multiplier=1.0
            )
        ]
        
        for item in items:
            repository.save(item)
        
        # Get prioritized items
        prioritized = repository.get_prioritized_items()
        assert len(prioritized) == 2
        
        # High priority should come first
        assert prioritized[0].id == 'high-priority'
        assert prioritized[1].id == 'low-priority'
    
    def test_get_metrics_summary(self, repository):
        """Test metrics generation"""
        # Create test items
        items = [
            BacklogItem(
                id='item-1',
                title='Item 1',
                type='feature',
                description='Feature item',
                acceptance_criteria=['Criteria 1'],
                effort=3,
                value=5,
                time_criticality=4,
                risk_reduction=3,
                status='READY',
                risk_tier='low',
                created_at=datetime.now().isoformat() + 'Z',
                links=[],
                aging_multiplier=1.0
            ),
            BacklogItem(
                id='item-2',
                title='Item 2',
                type='bug',
                description='Bug fix',
                acceptance_criteria=['Criteria 2'],
                effort=5,
                value=7,
                time_criticality=6,
                risk_reduction=5,
                status='DOING',
                risk_tier='medium',
                created_at=datetime.now().isoformat() + 'Z',
                links=[],
                aging_multiplier=1.0
            )
        ]
        
        for item in items:
            repository.save(item)
        
        # Get metrics
        metrics = repository.get_metrics_summary()
        
        assert metrics['total_items'] == 2
        assert 'READY' in metrics['status_breakdown']
        assert 'DOING' in metrics['status_breakdown']
        assert 'feature' in metrics['type_breakdown']
        assert 'bug' in metrics['type_breakdown']
        assert metrics['ready_items'] == 1


class TestInMemoryCache:
    """Test in-memory cache"""
    
    @pytest.fixture
    def cache(self):
        return InMemoryCache(default_ttl=3600)
    
    def test_set_and_get(self, cache):
        """Test basic cache operations"""
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        cache.set('key2', {'data': 123})
        assert cache.get('key2') == {'data': 123}
    
    def test_ttl_expiration(self, cache):
        """Test TTL expiration"""
        cache.set('short_lived', 'value', ttl=1)
        assert cache.get('short_lived') == 'value'
        
        # Wait for expiration (we'll mock this in real implementation)
        import time
        time.sleep(1.1)
        assert cache.get('short_lived') is None
    
    def test_delete(self, cache):
        """Test cache deletion"""
        cache.set('delete_me', 'value')
        assert cache.exists('delete_me')
        
        assert cache.delete('delete_me')
        assert not cache.exists('delete_me')
    
    def test_clear(self, cache):
        """Test cache clearing"""
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        cache.clear()
        assert not cache.exists('key1')
        assert not cache.exists('key2')
    
    def test_stats(self, cache):
        """Test cache statistics"""
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.get('key1')  # Access to update stats
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 2


class TestFileCache:
    """Test file-based cache"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def file_cache(self, temp_dir):
        return FileCache(temp_dir, default_ttl=3600)
    
    def test_set_and_get(self, file_cache):
        """Test file cache operations"""
        file_cache.set('key1', 'value1')
        assert file_cache.get('key1') == 'value1'
        
        file_cache.set('complex', {'data': [1, 2, 3]})
        assert file_cache.get('complex') == {'data': [1, 2, 3]}
    
    def test_persistence(self, temp_dir):
        """Test cache persistence across instances"""
        cache1 = FileCache(temp_dir)
        cache1.set('persistent', 'value')
        
        cache2 = FileCache(temp_dir)
        assert cache2.get('persistent') == 'value'
    
    def test_clear(self, file_cache):
        """Test cache clearing"""
        file_cache.set('key1', 'value1')
        file_cache.set('key2', 'value2')
        
        cleared_count = file_cache.clear()
        assert cleared_count == 2
        assert not file_cache.exists('key1')
        assert not file_cache.exists('key2')


class TestCacheManager:
    """Test combined cache manager"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def cache_manager(self, temp_dir):
        return CacheManager(
            use_memory_cache=True,
            use_file_cache=True,
            cache_dir=temp_dir
        )
    
    def test_layered_caching(self, cache_manager):
        """Test memory and file cache layers"""
        # Set value (should go to both layers)
        cache_manager.set('key1', 'value1')
        
        # Should retrieve from memory cache
        assert cache_manager.get('key1') == 'value1'
        
        # Clear memory cache only
        cache_manager.memory_cache.clear()
        
        # Should retrieve from file cache and populate memory
        assert cache_manager.get('key1') == 'value1'
        assert cache_manager.memory_cache.get('key1') == 'value1'
    
    def test_memory_only_caching(self, cache_manager):
        """Test memory-only caching"""
        cache_manager.set('memory_only', 'value', memory_only=True)
        
        # Should be in memory cache
        assert cache_manager.memory_cache.get('memory_only') == 'value'
        
        # Should not be in file cache
        assert cache_manager.file_cache.get('memory_only') is None
    
    def test_convenience_methods(self, cache_manager):
        """Test convenience methods"""
        # Test backlog metrics caching
        metrics = {'total': 10, 'ready': 5}
        cache_manager.cache_backlog_metrics(metrics)
        
        retrieved = cache_manager.get_backlog_metrics()
        assert retrieved == metrics
        
        # Test WSJF scores caching
        scores = {'item1': 8.5, 'item2': 6.2}
        cache_manager.cache_wsjf_scores(scores)
        
        retrieved_scores = cache_manager.get_wsjf_scores()
        assert retrieved_scores == scores
    
    def test_stats(self, cache_manager):
        """Test cache statistics"""
        cache_manager.set('key1', 'value1')
        cache_manager.set('key2', 'value2')
        
        stats = cache_manager.get_stats()
        assert 'memory' in stats
        assert 'file' in stats
        assert stats['memory']['total_entries'] == 2