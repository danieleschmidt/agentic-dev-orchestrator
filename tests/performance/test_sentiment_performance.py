"""
Performance and load tests for sentiment analysis
"""
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.sentiment import (
    SentimentAnalyzer, AsyncSentimentAnalyzer, SentimentCache,
    performance_monitor
)


class TestSentimentPerformance:
    """Performance tests for sentiment analysis"""
    
    def test_single_analysis_performance(self):
        """Test single text analysis performance"""
        analyzer = SentimentAnalyzer()
        
        with performance_monitor.monitor_operation("single_analysis", 1) as metrics:
            start_time = time.time()
            result = analyzer.analyze("This is a test text for performance measurement.")
            end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert result is not None
        assert processing_time < 0.1  # Should complete within 100ms
        
        # Check performance metrics were recorded
        assert len(performance_monitor.metrics_history) > 0
        latest_metric = performance_monitor.metrics_history[-1]
        assert latest_metric.operation_name == "single_analysis"
        assert latest_metric.texts_processed == 1
    
    def test_batch_analysis_performance(self):
        """Test batch analysis performance"""
        analyzer = SentimentAnalyzer()
        
        # Generate test texts
        texts = [f"This is test text number {i}" for i in range(100)]
        
        with performance_monitor.monitor_operation("batch_analysis", len(texts)) as metrics:
            start_time = time.time()
            results = analyzer.analyze_batch(texts)
            end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(texts) / processing_time
        
        assert len(results) == len(texts)
        assert throughput > 100  # Should process at least 100 texts per second
        assert processing_time < 5.0  # Should complete within 5 seconds
    
    def test_cache_performance_impact(self):
        """Test caching performance impact"""
        cache = SentimentCache(ttl=300)  # 5 minutes
        analyzer = SentimentAnalyzer()
        
        text = "This is a test text for cache performance measurement"
        
        # First analysis (cache miss)
        start_time = time.time()
        result1 = analyzer.analyze(text)
        first_time = time.time() - start_time
        cache.set(text, result1)
        
        # Second analysis (cache hit simulation)
        start_time = time.time()
        cached_result = cache.get(text)
        second_time = time.time() - start_time
        
        assert cached_result is not None
        assert second_time < first_time  # Cache should be faster
        
        # Check cache statistics
        stats = cache.get_stats()
        assert 'total_requests' in stats
        assert 'hit_rate_percent' in stats
    
    @pytest.mark.asyncio
    async def test_async_performance(self):
        """Test async analyzer performance"""
        analyzer = AsyncSentimentAnalyzer(max_workers=4, batch_size=10)
        
        try:
            texts = [f"Performance test text {i}" for i in range(50)]
            
            start_time = time.time()
            results = await analyzer.analyze_batch_async(texts)
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = len(texts) / processing_time
            
            assert len(results) == len(texts)
            assert throughput > 20  # Should be faster than synchronous
            assert processing_time < 10.0
            
        finally:
            analyzer.cleanup()
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable during processing"""
        import psutil
        import gc
        
        analyzer = SentimentAnalyzer()
        process = psutil.Process()
        
        # Measure initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many texts
        for batch_num in range(10):
            texts = [f"Memory test batch {batch_num} text {i}" for i in range(100)]
            results = analyzer.analyze_batch(texts)
            
            # Force garbage collection
            del results
            gc.collect()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (less than 50MB)
        assert memory_growth < 50, f"Memory grew by {memory_growth:.2f}MB"
    
    def test_concurrent_analysis_performance(self):
        """Test concurrent analysis performance"""
        analyzer = SentimentAnalyzer()
        num_threads = 4
        texts_per_thread = 25
        
        def analyze_batch(thread_id):
            texts = [f"Thread {thread_id} text {i}" for i in range(texts_per_thread)]
            start_time = time.time()
            results = analyzer.analyze_batch(texts)
            end_time = time.time()
            return len(results), end_time - start_time
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            futures = [executor.submit(analyze_batch, i) for i in range(num_threads)]
            
            total_processed = 0
            for future in as_completed(futures):
                processed, _ = future.result()
                total_processed += processed
            
            total_time = time.time() - start_time
        
        overall_throughput = total_processed / total_time
        
        assert total_processed == num_threads * texts_per_thread
        assert overall_throughput > 50  # Should maintain good throughput under concurrency
        assert total_time < 15.0  # Should complete reasonably quickly
    
    def test_large_text_performance(self):
        """Test performance with large texts"""
        analyzer = SentimentAnalyzer()
        
        # Create progressively larger texts
        base_text = "This is a great product with excellent features. "
        test_sizes = [100, 1000, 5000]  # Number of repetitions
        
        for size in test_sizes:
            large_text = base_text * size
            
            start_time = time.time()
            result = analyzer.analyze(large_text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            assert result is not None
            # Processing time should scale reasonably with text size
            assert processing_time < (size / 100)  # Rough scaling expectation
    
    def test_performance_monitoring_overhead(self):
        """Test that performance monitoring doesn't add significant overhead"""
        analyzer = SentimentAnalyzer()
        text = "Test text for performance monitoring overhead"
        
        # Test without monitoring
        start_time = time.time()
        for _ in range(100):
            analyzer.analyze(text)
        time_without_monitoring = time.time() - start_time
        
        # Test with monitoring
        start_time = time.time()
        for i in range(100):
            with performance_monitor.monitor_operation(f"test_op_{i}", 1):
                analyzer.analyze(text)
        time_with_monitoring = time.time() - start_time
        
        # Monitoring overhead should be minimal (less than 20% increase)
        overhead = (time_with_monitoring - time_without_monitoring) / time_without_monitoring
        assert overhead < 0.2, f"Performance monitoring overhead too high: {overhead:.2%}"


class TestLoadTesting:
    """Load testing for sentiment analysis"""
    
    def test_sustained_load(self):
        """Test system under sustained load"""
        analyzer = SentimentAnalyzer()
        cache = SentimentCache()
        
        total_texts = 0
        start_time = time.time()
        duration = 30  # 30 seconds of load testing
        
        while time.time() - start_time < duration:
            # Simulate mixed workload
            texts = [
                "This is an excellent product!",
                "Terrible service, very disappointed",
                "Average quality, nothing special",
                f"Dynamic text {int(time.time())}"  # Some cache misses
            ]
            
            results = analyzer.analyze_batch(texts)
            total_texts += len(results)
            
            # Brief pause to simulate realistic load
            time.sleep(0.1)
        
        actual_duration = time.time() - start_time
        average_throughput = total_texts / actual_duration
        
        assert total_texts > 0
        assert average_throughput > 10  # Minimum sustained throughput
        
        print(f"Sustained load test: {total_texts} texts in {actual_duration:.1f}s "
              f"({average_throughput:.1f} texts/sec)")
    
    @pytest.mark.asyncio
    async def test_async_load_capacity(self):
        """Test async system load capacity"""
        analyzer = AsyncSentimentAnalyzer(max_workers=8, batch_size=20)
        
        try:
            # Create large batch to test capacity
            texts = []
            for i in range(500):  # Large batch
                sentiment_type = i % 3
                if sentiment_type == 0:
                    texts.append(f"This is amazing content number {i}!")
                elif sentiment_type == 1:
                    texts.append(f"This is terrible content number {i}.")
                else:
                    texts.append(f"This is neutral content number {i}.")
            
            start_time = time.time()
            results = await analyzer.analyze_batch_async(texts)
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = len(texts) / processing_time
            
            assert len(results) == len(texts)
            assert throughput > 30  # Should handle high load efficiently
            assert processing_time < 60.0  # Should complete within reasonable time
            
            # Check performance stats
            stats = analyzer.get_performance_stats()
            assert 'cache_stats' in stats
            
            print(f"Async load test: {len(texts)} texts in {processing_time:.1f}s "
                  f"({throughput:.1f} texts/sec)")
            
        finally:
            analyzer.cleanup()
    
    def test_stress_testing(self):
        """Stress test the sentiment analysis system"""
        analyzer = SentimentAnalyzer()
        
        # Gradually increase load to find breaking point
        batch_sizes = [10, 50, 100, 200, 500]
        max_successful_size = 0
        
        for size in batch_sizes:
            texts = [f"Stress test text {i} with varied content" for i in range(size)]
            
            try:
                start_time = time.time()
                results = analyzer.analyze_batch(texts)
                end_time = time.time()
                
                processing_time = end_time - start_time
                throughput = len(results) / processing_time
                
                if len(results) == size and processing_time < 30.0 and throughput > 5:
                    max_successful_size = size
                    print(f"Stress test passed for size {size}: {throughput:.1f} texts/sec")
                else:
                    print(f"Stress test failed for size {size}")
                    break
                    
            except Exception as e:
                print(f"Stress test exception at size {size}: {e}")
                break
        
        assert max_successful_size >= 100, f"System failed at batch size {max_successful_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])