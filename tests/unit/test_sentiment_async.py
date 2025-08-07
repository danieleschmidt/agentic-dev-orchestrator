"""
Unit tests for async sentiment analysis
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.sentiment import (
    AsyncSentimentAnalyzer, SentimentAnalyzer, SentimentCache,
    SentimentResult, SentimentScore, SentimentLabel
)


class TestAsyncSentimentAnalyzer:
    """Test suite for AsyncSentimentAnalyzer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = AsyncSentimentAnalyzer(max_workers=2, batch_size=5)
    
    def teardown_method(self):
        """Cleanup resources"""
        self.analyzer.cleanup()
    
    @pytest.mark.asyncio
    async def test_async_analyze_single(self):
        """Test async single text analysis"""
        result = await self.analyzer.analyze_async("This is great!")
        
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_async_analyze_with_cache(self):
        """Test async analysis with caching"""
        text = "This is a test for caching"
        
        # First call - should miss cache
        result1 = await self.analyzer.analyze_async(text)
        
        # Second call - should hit cache
        result2 = await self.analyzer.analyze_async(text)
        
        assert result1.text == result2.text
        assert result1.label == result2.label
        assert abs(result1.confidence - result2.confidence) < 0.01
    
    @pytest.mark.asyncio
    async def test_batch_analysis_async(self):
        """Test async batch analysis"""
        texts = [
            "This is amazing!",
            "This is terrible!",
            "This is neutral.",
            "Great work!",
            "Bad experience."
        ]
        
        progress_calls = []
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        results = await self.analyzer.analyze_batch_async(texts, progress_callback)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, SentimentResult) for r in results)
        
        # Check that progress callback was called
        assert len(progress_calls) >= 1
        assert progress_calls[-1] == (len(texts), len(texts))
    
    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test empty batch handling"""
        results = await self.analyzer.analyze_batch_async([])
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_stream_analysis(self):
        """Test streaming analysis"""
        texts = ["Great!", "Terrible!", "Okay."]
        
        results = []
        async for result in self.analyzer.analyze_stream(texts):
            results.append(result)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, SentimentResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing performance"""
        texts = ["Test text"] * 20  # Many identical texts
        
        start_time = asyncio.get_event_loop().time()
        results = await self.analyzer.analyze_batch_async(texts)
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        
        assert len(results) == len(texts)
        # Should be faster than sequential due to caching and concurrency
        assert processing_time < 5.0  # Reasonable time limit
    
    def test_performance_stats(self):
        """Test performance statistics"""
        stats = self.analyzer.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'cache_stats' in stats
        assert 'max_workers' in stats
        assert 'batch_size' in stats
        assert 'executor_status' in stats
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in async processing"""
        # Mock the analyzer to raise an exception
        with patch.object(self.analyzer.analyzer, 'analyze', side_effect=Exception("Test error")):
            result = await self.analyzer.analyze_async("test text")
            
            # Should return neutral result with error metadata
            assert result.label == SentimentLabel.NEUTRAL
            assert result.confidence == 0.0
            assert 'error' in result.metadata
    
    def test_context_manager(self):
        """Test async analyzer as context manager"""
        with AsyncSentimentAnalyzer() as analyzer:
            assert analyzer.executor is not None
        
        # Executor should be shutdown after context exit
        assert analyzer.executor._shutdown


class TestSentimentCache:
    """Test suite for SentimentCache"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cache = SentimentCache(ttl=60)  # 1 minute TTL for testing
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        key1 = self.cache._generate_cache_key("test text", {"meta": "data"})
        key2 = self.cache._generate_cache_key("test text", {"meta": "data"})
        key3 = self.cache._generate_cache_key("different text", {"meta": "data"})
        
        assert key1 == key2  # Same input should generate same key
        assert key1 != key3  # Different input should generate different key
        assert key1.startswith("sentiment_")
    
    def test_cache_set_get(self):
        """Test cache set and get operations"""
        text = "test text"
        result = SentimentResult(
            text=text,
            scores=SentimentScore(0.6, 0.2, 0.2, 0.4),
            label=SentimentLabel.POSITIVE,
            confidence=0.8,
            metadata={"test": True}
        )
        
        # Set cache
        success = self.cache.set(text, result)
        assert success
        
        # Get from cache
        cached_result = self.cache.get(text)
        assert cached_result is not None
        assert cached_result.text == result.text
        assert cached_result.label == result.label
        assert abs(cached_result.confidence - result.confidence) < 0.01
    
    def test_cache_miss(self):
        """Test cache miss behavior"""
        result = self.cache.get("non-existent text")
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        text = "test text"
        result = SentimentResult(
            text=text,
            scores=SentimentScore(0.5, 0.3, 0.2, 0.2),
            label=SentimentLabel.POSITIVE,
            confidence=0.7
        )
        
        # Cache miss
        self.cache.get("missing text")
        
        # Cache set and hit
        self.cache.set(text, result)
        self.cache.get(text)
        
        stats = self.cache.get_stats()
        
        assert stats['total_requests'] == 2
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
        assert stats['hit_rate_percent'] == 50.0
    
    def test_serialization_deserialization(self):
        """Test result serialization and deserialization"""
        result = SentimentResult(
            text="test",
            scores=SentimentScore(0.6, 0.2, 0.2, 0.4),
            label=SentimentLabel.POSITIVE,
            confidence=0.8,
            metadata={"key": "value"}
        )
        
        # Serialize
        serialized = self.cache._serialize_result(result)
        assert isinstance(serialized, dict)
        assert 'text' in serialized
        assert 'scores' in serialized
        assert 'label' in serialized
        
        # Deserialize
        deserialized = self.cache._deserialize_result(serialized)
        assert isinstance(deserialized, SentimentResult)
        assert deserialized.text == result.text
        assert deserialized.label == result.label
        assert deserialized.confidence == result.confidence


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Integration tests for async sentiment analysis"""
    
    async def test_async_batch_with_cache(self):
        """Test async batch processing with caching"""
        analyzer = AsyncSentimentAnalyzer(max_workers=2, batch_size=3)
        
        try:
            texts = [
                "This is great!",
                "This is terrible!",
                "This is great!",  # Duplicate - should hit cache
                "This is neutral."
            ]
            
            results = await analyzer.analyze_batch_async(texts)
            
            assert len(results) == len(texts)
            
            # Check that duplicates have consistent results
            assert results[0].text == results[2].text
            assert results[0].label == results[2].label
            
            # Check cache performance
            stats = analyzer.get_performance_stats()
            cache_stats = stats['cache_stats']
            assert cache_stats['total_requests'] > 0
            
        finally:
            analyzer.cleanup()
    
    async def test_large_batch_processing(self):
        """Test processing of large batches"""
        analyzer = AsyncSentimentAnalyzer(max_workers=4, batch_size=10)
        
        try:
            # Generate large batch with varied content
            texts = []
            for i in range(50):
                if i % 3 == 0:
                    texts.append(f"This is amazing text number {i}!")
                elif i % 3 == 1:
                    texts.append(f"This is terrible text number {i}.")
                else:
                    texts.append(f"This is neutral text number {i}.")
            
            start_time = asyncio.get_event_loop().time()
            results = await analyzer.analyze_batch_async(texts)
            end_time = asyncio.get_event_loop().time()
            
            processing_time = end_time - start_time
            
            assert len(results) == len(texts)
            
            # Check sentiment distribution
            positive_count = sum(1 for r in results if r.label == SentimentLabel.POSITIVE)
            negative_count = sum(1 for r in results if r.label == SentimentLabel.NEGATIVE)
            neutral_count = sum(1 for r in results if r.label == SentimentLabel.NEUTRAL)
            
            assert positive_count > 0
            assert negative_count > 0
            
            # Performance check - should be reasonable
            throughput = len(texts) / processing_time
            assert throughput > 10  # At least 10 texts per second
            
        finally:
            analyzer.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])