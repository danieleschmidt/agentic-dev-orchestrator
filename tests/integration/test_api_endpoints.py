#!/usr/bin/env python3
"""
Integration tests for ADO REST API endpoints
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.api.server import create_app, ADOAPIServer
    from backlog_manager import BacklogItem
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="Flask not available")
class TestAPIEndpoints:
    """Test REST API endpoints"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def app(self, temp_dir):
        """Create test Flask app"""
        app = create_app(temp_dir)
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def sample_item_data(self):
        """Sample backlog item data"""
        return {
            'id': 'test-api-001',
            'title': 'API Test Item',
            'type': 'feature',
            'description': 'Test item for API testing',
            'acceptance_criteria': ['Should pass API tests'],
            'effort': 3,
            'value': 7,
            'time_criticality': 5,
            'risk_reduction': 4,
            'status': 'NEW',
            'risk_tier': 'low',
            'links': []
        }
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['service'] == 'ado-api'
    
    def test_get_empty_backlog(self, client):
        """Test getting empty backlog"""
        response = client.get('/api/v1/backlog')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['total'] == 0
        assert data['items'] == []
    
    def test_create_backlog_item(self, client, sample_item_data):
        """Test creating backlog item"""
        response = client.post('/api/v1/backlog', 
                              json=sample_item_data,
                              content_type='application/json')
        assert response.status_code == 201
        
        data = json.loads(response.data)
        assert data['id'] == sample_item_data['id']
        assert data['title'] == sample_item_data['title']
        assert data['status'] == 'NEW'
    
    def test_create_item_validation(self, client):
        """Test item creation validation"""
        # Missing required fields
        incomplete_data = {'title': 'Incomplete Item'}
        response = client.post('/api/v1/backlog',
                              json=incomplete_data,
                              content_type='application/json')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'Missing required field' in data['error']
    
    def test_get_specific_item(self, client, sample_item_data):
        """Test getting specific backlog item"""
        # First create the item
        client.post('/api/v1/backlog', 
                   json=sample_item_data,
                   content_type='application/json')
        
        # Then retrieve it
        response = client.get(f'/api/v1/backlog/{sample_item_data["id"]}')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['id'] == sample_item_data['id']
        assert data['title'] == sample_item_data['title']
    
    def test_get_nonexistent_item(self, client):
        """Test getting non-existent item"""
        response = client.get('/api/v1/backlog/nonexistent')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert data['error'] == 'Item not found'
    
    def test_update_item(self, client, sample_item_data):
        """Test updating backlog item"""
        # Create item first
        client.post('/api/v1/backlog',
                   json=sample_item_data,
                   content_type='application/json')
        
        # Update item
        update_data = {
            'title': 'Updated Title',
            'description': 'Updated description'
        }
        response = client.put(f'/api/v1/backlog/{sample_item_data["id"]}',
                             json=update_data,
                             content_type='application/json')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['title'] == 'Updated Title'
        assert data['description'] == 'Updated description'
    
    def test_update_item_status(self, client, sample_item_data):
        """Test updating item status"""
        # Create item first
        client.post('/api/v1/backlog',
                   json=sample_item_data,
                   content_type='application/json')
        
        # Update status: NEW -> REFINED
        status_data = {'status': 'REFINED'}
        response = client.patch(f'/api/v1/backlog/{sample_item_data["id"]}/status',
                               json=status_data,
                               content_type='application/json')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'REFINED'
        
        # Test invalid status transition
        invalid_status_data = {'status': 'DONE'}  # Can't go directly from REFINED to DONE
        response = client.patch(f'/api/v1/backlog/{sample_item_data["id"]}/status',
                               json=invalid_status_data,
                               content_type='application/json')
        assert response.status_code == 400
    
    def test_delete_item(self, client, sample_item_data):
        """Test deleting backlog item"""
        # Create item first
        client.post('/api/v1/backlog',
                   json=sample_item_data,
                   content_type='application/json')
        
        # Delete item
        response = client.delete(f'/api/v1/backlog/{sample_item_data["id"]}')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['message'] == 'Item deleted'
        
        # Verify item is gone
        response = client.get(f'/api/v1/backlog/{sample_item_data["id"]}')
        assert response.status_code == 404
    
    def test_get_ready_items(self, client):
        """Test getting READY items"""
        # Create items with different statuses
        items = [
            {
                'id': 'ready-1',
                'title': 'Ready Item 1',
                'type': 'feature',
                'description': 'Ready item',
                'status': 'READY'
            },
            {
                'id': 'new-1',
                'title': 'New Item 1', 
                'type': 'feature',
                'description': 'New item',
                'status': 'NEW'
            }
        ]
        
        for item in items:
            client.post('/api/v1/backlog',
                       json=item,
                       content_type='application/json')
        
        # Get ready items
        response = client.get('/api/v1/backlog/ready')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['total'] == 1
        assert data['items'][0]['status'] == 'READY'
    
    def test_get_prioritized_backlog(self, client):
        """Test getting prioritized backlog"""
        # Create items with different priorities
        high_priority_item = {
            'id': 'high-pri',
            'title': 'High Priority',
            'type': 'feature',
            'description': 'High priority item',
            'effort': 2,
            'value': 9,
            'time_criticality': 8,
            'risk_reduction': 7,
            'status': 'READY'
        }
        
        low_priority_item = {
            'id': 'low-pri',
            'title': 'Low Priority',
            'type': 'feature',
            'description': 'Low priority item',
            'effort': 8,
            'value': 2,
            'time_criticality': 1,
            'risk_reduction': 1,
            'status': 'READY'
        }
        
        # Create items
        client.post('/api/v1/backlog',
                   json=high_priority_item,
                   content_type='application/json')
        client.post('/api/v1/backlog',
                   json=low_priority_item,
                   content_type='application/json')
        
        # Get prioritized backlog
        response = client.get('/api/v1/backlog/prioritized')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['total'] == 2
        
        # High priority should come first
        assert data['items'][0]['id'] == 'high-pri'
        assert data['items'][1]['id'] == 'low-pri'
    
    def test_get_next_item(self, client):
        """Test getting next highest priority item"""
        # Create a ready item
        ready_item = {
            'id': 'next-item',
            'title': 'Next Item',
            'type': 'feature',
            'description': 'Next item to work on',
            'status': 'READY'
        }
        
        client.post('/api/v1/backlog',
                   json=ready_item,
                   content_type='application/json')
        
        # Get next item
        response = client.get('/api/v1/backlog/next')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['id'] == 'next-item'
        assert data['status'] == 'READY'
    
    def test_get_next_item_empty(self, client):
        """Test getting next item when none available"""
        response = client.get('/api/v1/backlog/next')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'No ready items available' in data['message']
    
    def test_discover_items(self, client):
        """Test item discovery endpoint"""
        response = client.post('/api/v1/backlog/discover')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'discovered_count' in data
        assert 'message' in data
    
    def test_get_metrics(self, client, sample_item_data):
        """Test metrics endpoint"""
        # Create a test item first
        client.post('/api/v1/backlog',
                   json=sample_item_data,
                   content_type='application/json')
        
        response = client.get('/api/v1/metrics')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'total_items' in data
        assert 'status_breakdown' in data
        assert 'type_breakdown' in data
        assert data['total_items'] >= 1
    
    def test_update_wsjf_scores(self, client, sample_item_data):
        """Test WSJF score update endpoint"""
        # Create a test item first
        client.post('/api/v1/backlog',
                   json=sample_item_data,
                   content_type='application/json')
        
        response = client.post('/api/v1/wsjf/update')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'updated_count' in data
        assert data['updated_count'] >= 1
    
    def test_clear_cache(self, client):
        """Test cache clearing endpoint"""
        response = client.post('/api/v1/cache/clear')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['message'] == 'Cache cleared'
        assert 'results' in data
    
    def test_duplicate_item_creation(self, client, sample_item_data):
        """Test creating duplicate items"""
        # Create item first time
        response = client.post('/api/v1/backlog',
                              json=sample_item_data,
                              content_type='application/json')
        assert response.status_code == 201
        
        # Try to create same item again
        response = client.post('/api/v1/backlog',
                              json=sample_item_data,
                              content_type='application/json')
        assert response.status_code == 409
        
        data = json.loads(response.data)
        assert data['error'] == 'Item already exists'
    
    def test_error_handling(self, client):
        """Test API error handling"""
        # Test 404 for non-existent endpoints
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404
        
        # Test 400 for malformed JSON
        response = client.post('/api/v1/backlog',
                              data='invalid json',
                              content_type='application/json')
        assert response.status_code == 400


@pytest.mark.skipif(not API_AVAILABLE, reason="Flask not available")
class TestAPIServerIntegration:
    """Test API server integration with components"""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def server(self, temp_dir):
        """Create API server instance"""
        return ADOAPIServer(temp_dir)
    
    def test_server_initialization(self, server):
        """Test server initialization"""
        assert server.backlog_manager is not None
        assert server.executor is not None
        assert server.repository is not None
        assert server.cache is not None
    
    def test_server_components_integration(self, server):
        """Test integration between server components"""
        # Create item via repository
        from backlog_manager import BacklogItem
        
        item = BacklogItem(
            id='integration-test',
            title='Integration Test Item',
            type='feature',
            description='Test integration',
            acceptance_criteria=['Should integrate'],
            effort=3,
            value=5,
            time_criticality=4,
            risk_reduction=3,
            status='NEW',
            risk_tier='low',
            created_at=datetime.now().isoformat() + 'Z',
            links=[],
            aging_multiplier=1.0
        )
        
        # Save via repository
        assert server.repository.save(item)
        
        # Verify via backlog manager
        server.backlog_manager.load_backlog()
        loaded_items = server.backlog_manager.items
        
        # Should have at least our test item
        assert len(loaded_items) >= 1
        
        # Find our item
        our_item = next((i for i in loaded_items if i.id == 'integration-test'), None)
        assert our_item is not None
        assert our_item.title == 'Integration Test Item'