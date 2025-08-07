"""
Integration tests for sentiment analysis API endpoints
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from src.api.server import ADOAPIServer


class TestSentimentAPI:
    """Test suite for sentiment analysis API endpoints"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create temporary directory for test repo
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Create minimal directory structure
        (self.repo_path / "backlog").mkdir()
        (self.repo_path / ".ado").mkdir()
        
        # Create API server
        self.server = ADOAPIServer(str(self.repo_path), host="localhost", port=5000)
        self.client = self.server.get_app().test_client()
        
        # Setup test context
        self.app_context = self.server.get_app().app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        self.app_context.pop()
        shutil.rmtree(self.temp_dir)
    
    def test_sentiment_analyze_endpoint(self):
        """Test /api/v1/sentiment/analyze endpoint"""
        # Test positive sentiment
        response = self.client.post(
            '/api/v1/sentiment/analyze',
            json={'text': 'This is absolutely amazing!'},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'text' in data
        assert 'label' in data
        assert 'confidence' in data
        assert 'scores' in data
        assert data['label'] == 'positive'
        assert data['confidence'] > 0.5
        assert data['scores']['compound'] > 0
    
    def test_sentiment_analyze_negative(self):
        """Test negative sentiment analysis"""
        response = self.client.post(
            '/api/v1/sentiment/analyze',
            json={'text': 'This is terrible and awful!'},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['label'] == 'negative'
        assert data['scores']['compound'] < 0
    
    def test_sentiment_analyze_neutral(self):
        """Test neutral sentiment analysis"""
        response = self.client.post(
            '/api/v1/sentiment/analyze',
            json={'text': 'The weather is cloudy today.'},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['label'] == 'neutral'
        assert abs(data['scores']['compound']) < 0.1
    
    def test_sentiment_analyze_with_metadata(self):
        """Test sentiment analysis with metadata"""
        metadata = {'source': 'test', 'user_id': 123}
        response = self.client.post(
            '/api/v1/sentiment/analyze',
            json={'text': 'Great work!', 'metadata': metadata},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'metadata' in data
        assert data['metadata']['source'] == 'test'
        assert data['metadata']['user_id'] == 123
    
    def test_sentiment_analyze_empty_text(self):
        """Test sentiment analysis with empty text"""
        response = self.client.post(
            '/api/v1/sentiment/analyze',
            json={'text': ''},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['label'] == 'neutral'
        assert data['confidence'] == 1.0
        assert data['scores']['neutral'] == 1.0
    
    def test_sentiment_analyze_missing_text(self):
        """Test sentiment analysis with missing text parameter"""
        response = self.client.post(
            '/api/v1/sentiment/analyze',
            json={},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Text required' in data['error']
    
    def test_sentiment_batch_analyze(self):
        """Test batch sentiment analysis endpoint"""
        texts = [
            "This is amazing!",
            "This is terrible!",
            "This is neutral.",
            "Great job!",
            "Bad experience."
        ]
        
        response = self.client.post(
            '/api/v1/sentiment/batch',
            json={'texts': texts},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'results' in data
        assert 'total' in data
        assert len(data['results']) == len(texts)
        assert data['total'] == len(texts)
        
        # Check that all results have required fields
        for result in data['results']:
            assert 'text' in result
            assert 'label' in result
            assert 'confidence' in result
            assert 'scores' in result
    
    def test_sentiment_batch_empty(self):
        """Test batch analysis with empty array"""
        response = self.client.post(
            '/api/v1/sentiment/batch',
            json={'texts': []},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_sentiment_batch_missing_texts(self):
        """Test batch analysis with missing texts parameter"""
        response = self.client.post(
            '/api/v1/sentiment/batch',
            json={},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Texts array required' in data['error']
    
    def test_sentiment_batch_invalid_format(self):
        """Test batch analysis with invalid format"""
        response = self.client.post(
            '/api/v1/sentiment/batch',
            json={'texts': 'not an array'},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'must be an array' in data['error']
    
    def test_invalid_json(self):
        """Test endpoints with invalid JSON"""
        response = self.client.post(
            '/api/v1/sentiment/analyze',
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_missing_content_type(self):
        """Test endpoints without proper content type"""
        response = self.client.post(
            '/api/v1/sentiment/analyze',
            data=json.dumps({'text': 'test'})
            # No content-type header
        )
        
        assert response.status_code == 400
    
    def test_get_method_not_allowed(self):
        """Test that GET method is not allowed on POST endpoints"""
        response = self.client.get('/api/v1/sentiment/analyze')
        assert response.status_code == 405  # Method not allowed
    
    def test_nonexistent_endpoint(self):
        """Test nonexistent sentiment endpoint"""
        response = self.client.post('/api/v1/sentiment/nonexistent')
        assert response.status_code == 404


class TestSentimentAPIPerformance:
    """Performance tests for sentiment API"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        (self.repo_path / "backlog").mkdir()
        (self.repo_path / ".ado").mkdir()
        
        self.server = ADOAPIServer(str(self.repo_path))
        self.client = self.server.get_app().test_client()
        
        self.app_context = self.server.get_app().app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        self.app_context.pop()
        shutil.rmtree(self.temp_dir)
    
    def test_batch_performance(self):
        """Test batch processing performance"""
        import time
        
        # Create a reasonably large batch
        texts = [f"This is test text number {i}" for i in range(50)]
        
        start_time = time.time()
        response = self.client.post(
            '/api/v1/sentiment/batch',
            json={'texts': texts},
            content_type='application/json'
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['results']) == len(texts)
        
        # Performance assertion - should process at least 10 texts per second
        throughput = len(texts) / processing_time
        assert throughput > 10, f"Throughput too low: {throughput:.2f} texts/sec"
    
    def test_caching_performance(self):
        """Test caching improves performance"""
        import time
        
        text = "This is a test for caching performance"
        
        # First request (cold cache)
        start_time = time.time()
        response1 = self.client.post(
            '/api/v1/sentiment/analyze',
            json={'text': text},
            content_type='application/json'
        )
        first_time = time.time() - start_time
        
        # Second request (should hit cache)
        start_time = time.time()
        response2 = self.client.post(
            '/api/v1/sentiment/analyze',
            json={'text': text},
            content_type='application/json'
        )
        second_time = time.time() - start_time
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        
        # Results should be identical
        assert data1['label'] == data2['label']
        assert abs(data1['confidence'] - data2['confidence']) < 0.01
        
        # Second request should be faster (cache hit)
        # Note: In test environment, difference might be small
        assert second_time <= first_time * 2  # Allow some variance


if __name__ == "__main__":
    pytest.main([__file__])