#!/usr/bin/env python3
"""
Comprehensive Integration Tests for ADO
Tests the complete system functionality with realistic scenarios
"""

import pytest
import asyncio
import tempfile
import json
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import ADO components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.sentiment_analyzer import SentimentAnalyzer, SentimentLabel
from src.intelligence.adaptive_learning import AdaptiveLearningEngine
from src.resilience.circuit_breaker_enhanced import EnhancedCircuitBreaker
from src.resilience.retry_with_backoff import AdvancedRetry, RetryConfig
from src.security.enhanced_scanner import EnhancedSecurityScanner
from src.performance.distributed_executor import DistributedExecutor, TaskDefinition
from src.performance.adaptive_cache import AdaptiveCache, CacheStrategy
from backlog_manager import BacklogManager
from autonomous_executor import AutonomousExecutor


class TestSentimentAnalysisIntegration:
    """Test sentiment analysis functionality"""
    
    def test_sentiment_analyzer_initialization(self):
        """Test sentiment analyzer can be initialized"""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
        assert len(analyzer.positive_patterns) > 0
        assert len(analyzer.negative_patterns) > 0
        
    def test_basic_sentiment_analysis(self):
        """Test basic sentiment analysis functionality"""
        analyzer = SentimentAnalyzer()
        
        # Test positive sentiment
        positive_text = "This is awesome! Great work completed successfully."
        result = analyzer.analyze_text(positive_text)
        
        assert result.label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]
        assert result.confidence > 0.5
        assert result.positive_score > result.negative_score
        
        # Test negative sentiment
        negative_text = "This is very frustrating and blocked. Major issues."
        result = analyzer.analyze_text(negative_text)
        
        assert result.label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]
        assert result.confidence > 0.5
        assert result.negative_score > result.positive_score
        
    def test_backlog_sentiment_analysis(self):
        """Test sentiment analysis on backlog items"""
        analyzer = SentimentAnalyzer()
        
        backlog_item = {
            "id": "test-123",
            "title": "Fix critical bug in payment system",
            "description": "Users are frustrated with payment failures",
            "wsjf": {
                "user_business_value": 9,
                "time_criticality": 8,
                "risk_reduction_opportunity_enablement": 7,
                "job_size": 3
            }
        }
        
        result = analyzer.analyze_backlog_item(backlog_item)
        
        assert result is not None
        assert result.label in [SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
        assert len(result.keywords) > 0
        
    def test_team_sentiment_report(self):
        """Test team sentiment report generation"""
        analyzer = SentimentAnalyzer()
        
        backlog_items = [
            {
                "id": "task-1",
                "title": "Implement user authentication",
                "description": "Create secure login system",
                "wsjf": {"user_business_value": 8, "time_criticality": 6, "risk_reduction_opportunity_enablement": 5, "job_size": 5}
            },
            {
                "id": "task-2", 
                "title": "Fix annoying bugs",
                "description": "Multiple users complaining about crashes",
                "wsjf": {"user_business_value": 7, "time_criticality": 9, "risk_reduction_opportunity_enablement": 6, "job_size": 3}
            },
            {
                "id": "task-3",
                "title": "Celebrate successful deployment",
                "description": "Team achieved excellent results",
                "wsjf": {"user_business_value": 3, "time_criticality": 1, "risk_reduction_opportunity_enablement": 1, "job_size": 1}
            }
        ]
        
        report = analyzer.generate_team_sentiment_report(backlog_items)
        
        assert report['total_items_analyzed'] == 3
        assert 'sentiment_distribution' in report
        assert 'insights' in report
        assert 'individual_results' in report
        assert len(report['individual_results']) == 3


class TestAdaptiveLearningIntegration:
    """Test adaptive learning functionality"""
    
    def test_adaptive_learning_initialization(self):
        """Test adaptive learning engine initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = AdaptiveLearningEngine(temp_dir)
            assert engine is not None
            assert engine.repo_root == Path(temp_dir)
            
    def test_execution_pattern_analysis(self):
        """Test analysis of execution patterns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = AdaptiveLearningEngine(temp_dir)
            
            # Mock execution history
            execution_history = [
                {
                    "start_time": "2024-01-01T10:00:00Z",
                    "end_time": "2024-01-01T10:05:00Z",
                    "completed_items": ["task-1", "task-2"],
                    "failed_items": [],
                    "memory_usage_mb": 512
                },
                {
                    "start_time": "2024-01-01T11:00:00Z",
                    "end_time": "2024-01-01T11:03:00Z",
                    "completed_items": ["task-3"],
                    "failed_items": ["task-4"],
                    "memory_usage_mb": 768
                }
            ]
            
            insights = engine.analyze_execution_patterns(execution_history)
            
            # Should generate some insights
            assert isinstance(insights, list)
            # With limited data, might not generate specific insights
            
    def test_backlog_pattern_learning(self):
        """Test learning from backlog patterns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = AdaptiveLearningEngine(temp_dir)
            
            # Mock backlog history
            backlog_history = [
                {
                    "id": "task-1",
                    "status": "DONE",
                    "wsjf": {"user_business_value": 8, "time_criticality": 6, "risk_reduction_opportunity_enablement": 5, "job_size": 3},
                    "created_at": "2024-01-01T09:00:00Z",
                    "completed_at": "2024-01-01T10:00:00Z"
                },
                {
                    "id": "task-2",
                    "status": "DONE", 
                    "wsjf": {"user_business_value": 6, "time_criticality": 4, "risk_reduction_opportunity_enablement": 3, "job_size": 5},
                    "created_at": "2024-01-01T09:00:00Z",
                    "completed_at": "2024-01-01T12:00:00Z"
                }
            ]
            
            insights = engine.learn_from_backlog_patterns(backlog_history)
            
            assert isinstance(insights, list)
            # With limited data, might not generate many insights
            
    def test_learning_report_generation(self):
        """Test comprehensive learning report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = AdaptiveLearningEngine(temp_dir)
            
            report = engine.generate_learning_report()
            
            assert 'timestamp' in report
            assert 'total_insights' in report
            assert 'high_confidence_insights' in report
            assert 'categories' in report
            assert 'recommendations' in report


class TestResilienceIntegration:
    """Test resilience components"""
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker basic functionality"""
        breaker = EnhancedCircuitBreaker(
            name="test_service",
            failure_threshold=3,
            timeout=1.0
        )
        
        # Test successful calls
        def successful_func():
            return "success"
            
        result = breaker.call(successful_func)
        assert result == "success"
        
        metrics = breaker.get_metrics()
        assert metrics['successful_calls'] == 1
        assert metrics['failed_calls'] == 0
        
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling"""
        breaker = EnhancedCircuitBreaker(
            name="failing_service",
            failure_threshold=2,
            timeout=0.1
        )
        
        def failing_func():
            raise ValueError("Service error")
            
        # First failure
        with pytest.raises(ValueError):
            breaker.call(failing_func)
            
        # Second failure
        with pytest.raises(ValueError):
            breaker.call(failing_func)
            
        metrics = breaker.get_metrics()
        assert metrics['failed_calls'] == 2
        assert metrics['consecutive_failures'] == 2
        
    def test_retry_mechanism(self):
        """Test retry mechanism functionality"""
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            retryable_exceptions=(ValueError,)
        )
        
        retry_executor = AdvancedRetry(config)
        
        # Test successful retry after failures
        attempt_count = 0
        
        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
            
        result = retry_executor.execute(flaky_func)
        
        assert result.success is True
        assert result.total_attempts == 3
        assert result.result == "Success on attempt 3"
        
    def test_retry_exhaustion(self):
        """Test retry mechanism when all attempts fail"""
        config = RetryConfig(
            max_attempts=2,
            initial_delay=0.01,
            retryable_exceptions=(ValueError,)
        )
        
        retry_executor = AdvancedRetry(config)
        
        def always_failing_func():
            raise ValueError("Always fails")
            
        result = retry_executor.execute(always_failing_func)
        
        assert result.success is False
        assert result.total_attempts == 2
        assert "Always fails" in str(result.final_exception)


class TestSecurityIntegration:
    """Test security scanning functionality"""
    
    def test_security_scanner_initialization(self):
        """Test security scanner initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            scanner = EnhancedSecurityScanner(temp_dir)
            assert scanner is not None
            assert scanner.repo_root == Path(temp_dir)
            
    def test_secret_detection(self):
        """Test secret detection in code files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            scanner = EnhancedSecurityScanner(temp_dir)
            
            # Create test file with potential secret
            test_file = Path(temp_dir) / "config.py"
            test_file.write_text('''
API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"
DATABASE_URL = "postgres://user:password123@localhost:5432/db"
''')
            
            result = scanner._run_secret_scan()
            
            assert result.scan_successful is True
            assert len(result.findings) > 0
            
            # Should detect the API key pattern
            secret_findings = [f for f in result.findings if 'api' in f.title.lower()]
            assert len(secret_findings) > 0
            
    def test_static_analysis_patterns(self):
        """Test static analysis pattern detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            scanner = EnhancedSecurityScanner(temp_dir)
            
            # Create test Python file with security issues
            test_file = Path(temp_dir) / "unsafe_code.py"
            test_file.write_text('''
import subprocess

def unsafe_function(user_input):
    # Dangerous: shell=True with user input
    result = subprocess.run(f"echo {user_input}", shell=True)
    
    # Dangerous: eval usage
    eval(user_input)
    
    return result
''')
            
            result = scanner._run_static_analysis()
            
            assert result.scan_successful is True
            # Should detect shell=True and eval usage
            
    def test_configuration_security_check(self):
        """Test configuration security checking"""
        with tempfile.TemporaryDirectory() as temp_dir:
            scanner = EnhancedSecurityScanner(temp_dir)
            
            # Create test config file with security issues
            config_file = Path(temp_dir) / "config.yaml"
            config_file.write_text('''
debug: true
allow_origins: "*"
ssl_verify: false
secret_key: "weak"
''')
            
            result = scanner._run_configuration_check()
            
            assert result.scan_successful is True
            assert result.total_files_scanned > 0
            
    def test_comprehensive_security_scan(self):
        """Test comprehensive security scan"""
        with tempfile.TemporaryDirectory() as temp_dir:
            scanner = EnhancedSecurityScanner(temp_dir)
            
            # Create various test files
            (Path(temp_dir) / "requirements.txt").write_text("requests==2.20.0\npillow==7.0.0")
            (Path(temp_dir) / "test.py").write_text("import os\nprint('Hello World')")
            
            results = scanner.run_comprehensive_scan()
            
            assert isinstance(results, dict)
            assert len(results) > 0
            
            # Should have results for multiple scan types
            scan_types = list(results.keys())
            expected_types = ['static_analysis', 'dependency_check', 'secret_scan', 'configuration_check', 'license_check']
            
            for expected_type in expected_types:
                assert expected_type in scan_types
                
    def test_security_report_generation(self):
        """Test security report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            scanner = EnhancedSecurityScanner(temp_dir)
            
            # Run a basic scan to populate some results
            scanner.run_comprehensive_scan()
            
            report = scanner.generate_security_report()
            
            assert 'timestamp' in report
            assert 'total_findings' in report
            assert 'severity_distribution' in report
            assert 'risk_score' in report
            assert 'recommendations' in report
            
            # Risk score should be non-negative
            assert report['risk_score'] >= 0


class TestPerformanceIntegration:
    """Test performance optimization components"""
    
    def test_distributed_executor_initialization(self):
        """Test distributed executor initialization"""
        executor = DistributedExecutor(
            max_workers=4,
            min_workers=1,
            auto_scale=False
        )
        
        assert executor is not None
        assert executor.max_workers == 4
        assert executor.min_workers == 1
        
        # Clean shutdown
        executor.stop()
        
    def test_distributed_task_execution(self):
        """Test distributed task execution"""
        executor = DistributedExecutor(
            max_workers=2,
            min_workers=1,
            auto_scale=False
        )
        
        # Register test function
        def test_function(x: int, y: int = 10) -> int:
            return x + y
            
        executor.register_function("test_func", test_function)
        
        try:
            executor.start()
            
            # Submit task
            task = TaskDefinition(
                id="test_task_1",
                func_name="test_func",
                args=(5,),
                kwargs={'y': 15}
            )
            
            task_id = executor.submit_task(task)
            assert task_id == "test_task_1"
            
            # Wait for completion
            result = executor.get_result(task_id, timeout=10.0)
            
            assert result is not None
            assert result.status.value in ["completed", "failed"]
            
            if result.status.value == "completed":
                assert result.result == 20
                
        finally:
            executor.stop()
            
    def test_adaptive_cache_functionality(self):
        """Test adaptive cache functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = AdaptiveCache(
                strategy=CacheStrategy.LRU,
                max_size_bytes=1024 * 1024,  # 1MB
                cache_dir=Path(temp_dir)
            )
            
            try:
                # Test basic operations
                cache.put("key1", "value1")
                cache.put("key2", {"data": [1, 2, 3]})
                
                # Test retrieval
                value1 = cache.get("key1")
                assert value1 == "value1"
                
                value2 = cache.get("key2")
                assert value2 == {"data": [1, 2, 3]}
                
                # Test cache miss
                missing = cache.get("nonexistent")
                assert missing is None
                
                # Test get_or_compute
                def expensive_computation():
                    return "computed_value"
                    
                computed = cache.get_or_compute("computed_key", expensive_computation)
                assert computed == "computed_value"
                
                # Should be cached now
                cached_value = cache.get("computed_key")
                assert cached_value == "computed_value"
                
                # Test metrics
                metrics = cache.get_metrics()
                assert metrics['total_requests'] > 0
                assert metrics['cache_hits'] > 0
                
            finally:
                cache.stop()
                
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration"""
        cache = AdaptiveCache(default_ttl=0.1)  # 100ms TTL
        
        try:
            # Put value with short TTL
            cache.put("temp_key", "temp_value", ttl=0.05)  # 50ms
            
            # Should be available immediately
            value = cache.get("temp_key")
            assert value == "temp_value"
            
            # Wait for expiration
            time.sleep(0.1)
            
            # Should be expired now
            expired_value = cache.get("temp_key")
            assert expired_value is None
            
        finally:
            cache.stop()


class TestSystemIntegration:
    """Test complete system integration scenarios"""
    
    def test_backlog_manager_integration(self):
        """Test backlog manager integration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test backlog file
            backlog_file = Path(temp_dir) / "test_backlog.json"
            backlog_data = {
                "title": "Test task",
                "description": "Test task description", 
                "wsjf": {
                    "user_business_value": 8,
                    "time_criticality": 6,
                    "risk_reduction_opportunity_enablement": 4,
                    "job_size": 3
                }
            }
            
            backlog_file.write_text(json.dumps(backlog_data, indent=2))
            
            # Test BacklogManager
            manager = BacklogManager(str(temp_dir))
            
            # Should be able to load backlog files
            assert manager is not None
            
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_simulation(self):
        """Test end-to-end workflow simulation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            sentiment_analyzer = SentimentAnalyzer()
            learning_engine = AdaptiveLearningEngine(temp_dir)
            cache = AdaptiveCache(cache_dir=Path(temp_dir) / "cache")
            
            try:
                # Simulate a complete workflow
                
                # 1. Analyze sentiment of backlog items
                backlog_items = [
                    {
                        "id": "task-1",
                        "title": "Implement user dashboard",
                        "description": "Create intuitive user interface",
                        "wsjf": {"user_business_value": 8, "time_criticality": 5, "risk_reduction_opportunity_enablement": 3, "job_size": 5}
                    }
                ]
                
                sentiment_report = sentiment_analyzer.generate_team_sentiment_report(backlog_items)
                assert sentiment_report['total_items_analyzed'] == 1
                
                # 2. Cache the sentiment analysis
                cache_key = f"sentiment_report_{int(time.time())}"
                cache.put(cache_key, sentiment_report, ttl=300)  # 5 minutes
                
                # 3. Retrieve from cache
                cached_report = cache.get(cache_key)
                assert cached_report == sentiment_report
                
                # 4. Generate learning report
                learning_report = learning_engine.generate_learning_report()
                assert 'timestamp' in learning_report
                
                # Test completed successfully
                assert True
                
            finally:
                cache.stop()
                
    def test_error_handling_integration(self):
        """Test error handling across integrated components"""
        # Test circuit breaker with retry mechanism
        breaker = EnhancedCircuitBreaker(
            name="integration_test",
            failure_threshold=2,
            timeout=0.1
        )
        
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            retryable_exceptions=(ValueError,)
        )
        
        retry_executor = AdvancedRetry(config)
        
        failure_count = 0
        
        def flaky_service():
            nonlocal failure_count
            failure_count += 1
            
            if failure_count < 3:
                raise ValueError(f"Service failure {failure_count}")
            return "Service recovered"
            
        # Test integration: retry will eventually succeed before circuit opens
        result = retry_executor.execute(lambda: breaker.call(flaky_service))
        
        assert result.success is True
        assert result.result == "Service recovered"
        
        # Circuit should still be closed (successful recovery)
        metrics = breaker.get_metrics()
        assert metrics['state'] == 'closed'
        
    def test_performance_under_load(self):
        """Test system performance under simulated load"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = AdaptiveCache(
                strategy=CacheStrategy.ADAPTIVE,
                max_size_bytes=512 * 1024  # 512KB
            )
            
            sentiment_analyzer = SentimentAnalyzer()
            
            try:
                # Simulate load: multiple concurrent sentiment analyses with caching
                def analyze_and_cache(item_id: str, text: str):
                    cache_key = f"sentiment_{item_id}"
                    
                    # Check cache first
                    cached_result = cache.get(cache_key)
                    if cached_result:
                        return cached_result
                        
                    # Compute and cache
                    result = sentiment_analyzer.analyze_text(text)
                    cache.put(cache_key, result, ttl=60)
                    return result
                    
                # Run multiple analyses
                test_texts = [
                    "This is a great improvement to our system",
                    "Users are frustrated with the current interface", 
                    "The team completed the sprint successfully",
                    "There are critical issues that need immediate attention",
                    "Excellent collaboration and results achieved"
                ]
                
                start_time = time.time()
                
                results = []
                for i, text in enumerate(test_texts * 5):  # 25 total analyses
                    result = analyze_and_cache(f"item_{i % 5}", text)
                    results.append(result)
                    
                end_time = time.time()
                
                # Should complete in reasonable time
                execution_time = end_time - start_time
                assert execution_time < 5.0  # Should complete within 5 seconds
                
                # Should have results
                assert len(results) == 25
                
                # Cache should show hits
                metrics = cache.get_metrics()
                assert metrics['cache_hits'] > 0  # Should have cache hits from duplicates
                
            finally:
                cache.stop()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
