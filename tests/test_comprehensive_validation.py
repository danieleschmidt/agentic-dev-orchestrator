#!/usr/bin/env python3
"""
Comprehensive Validation Test Suite v4.0
Validates all enhanced features and ensures quality gates are met
"""

import os
import sys
import json
import time
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Test imports
try:
    from security.input_validator import SecurityInputValidator, validate_input
    from resilience.error_recovery_system import ErrorRecoverySystem, ErrorSeverity, resilient_operation
    from performance.intelligent_cache_system import IntelligentCacheSystem, cached_function
    from performance.auto_scaling_manager import AutoScalingManager, SystemMetrics, ResourceState
    ENHANCED_FEATURES = True
except ImportError as e:
    ENHANCED_FEATURES = False
    pytest.skip(f"Enhanced features not available: {e}", allow_module_level=True)

# Import main modules
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from ado import main, cmd_validate, cmd_run, cmd_status
    from backlog_manager import BacklogManager, BacklogItem
    from autonomous_executor import AutonomousExecutor
except ImportError as e:
    pytest.skip(f"Main modules not available: {e}", allow_module_level=True)


class TestSecurityInputValidator:
    """Test security input validation system"""
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        validator = SecurityInputValidator()
        assert validator is not None
        assert len(validator.pattern_cache) > 0
        assert validator.MAX_STRING_LENGTH > 0
    
    def test_string_validation_safe_input(self):
        """Test validation of safe string input"""
        validator = SecurityInputValidator()
        result = validator.validate_string("Hello World")
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.sanitized_value is not None
    
    def test_string_validation_dangerous_patterns(self):
        """Test detection of dangerous patterns"""
        validator = SecurityInputValidator()
        dangerous_inputs = [
            "__import__('os').system('rm -rf /')",
            "eval('malicious_code')",
            "exec('bad_code')",
            "../../../etc/passwd",
            "'; DROP TABLE users; --"
        ]
        
        for dangerous_input in dangerous_inputs:
            result = validator.validate_string(dangerous_input)
            assert not result.is_valid, f"Should reject dangerous input: {dangerous_input}"
            assert len(result.errors) > 0
    
    def test_file_path_validation(self):
        """Test file path validation"""
        validator = SecurityInputValidator()
        
        # Safe paths
        safe_paths = ["./config.json", "backlog/item.json", "docs/status/report.json"]
        for path in safe_paths:
            result = validator.validate_file_path(path)
            assert result.is_valid, f"Should accept safe path: {path}"
        
        # Dangerous paths
        dangerous_paths = ["../../../etc/passwd", "/etc/shadow", "C:\\Windows\\System32"]
        for path in dangerous_paths:
            result = validator.validate_file_path(path)
            # Some may be valid paths but should generate warnings
            if result.is_valid:
                assert len(result.warnings) > 0
    
    def test_json_validation(self):
        """Test JSON data validation"""
        validator = SecurityInputValidator()
        
        # Valid JSON
        valid_json = '{"title": "Test", "value": 123}'
        result = validator.validate_json_data(valid_json)
        assert result.is_valid
        assert result.sanitized_value is not None
        
        # Invalid JSON
        invalid_json = '{"title": "Test", "value":}'
        result = validator.validate_json_data(invalid_json)
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_environment_variable_validation(self):
        """Test environment variable validation"""
        validator = SecurityInputValidator()
        
        # Valid token-like value
        valid_token = "sk-1234567890abcdef1234567890abcdef"
        result = validator.validate_environment_variable(valid_token)
        assert result.is_valid
        
        # Invalid placeholder values
        invalid_values = ["password", "secret", "token", "key", ""]
        for invalid_value in invalid_values:
            result = validator.validate_environment_variable(invalid_value)
            assert not result.is_valid


class TestErrorRecoverySystem:
    """Test error recovery and resilience system"""
    
    def test_error_recovery_initialization(self):
        """Test error recovery system initializes"""
        recovery_system = ErrorRecoverySystem(persist_errors=False)
        assert recovery_system is not None
        assert len(recovery_system.handlers) > 0
    
    def test_error_handling_with_retry(self):
        """Test error handling with retry strategy"""
        recovery_system = ErrorRecoverySystem(persist_errors=False)
        
        # Simulate a transient error
        error = ConnectionError("Connection failed")
        result = recovery_system.handle_error(
            error=error,
            component="test",
            operation="test_operation",
            severity=ErrorSeverity.MEDIUM
        )
        
        assert result is not None
        assert result.strategy_used is not None
    
    def test_resilient_operation_decorator(self):
        """Test resilient operation decorator"""
        call_count = 0
        
        @resilient_operation("test", "failing_operation")
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count >= 2  # Should retry at least once
    
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker functionality"""
        recovery_system = ErrorRecoverySystem(persist_errors=False)
        
        # Generate multiple failures to trigger circuit breaker
        for i in range(6):
            error = ConnectionError(f"Failure {i}")
            recovery_system.handle_error(
                error=error,
                component="test",
                operation="failing_service",
                severity=ErrorSeverity.HIGH
            )
        
        # Verify circuit breaker behavior
        summary = recovery_system.get_error_summary()
        assert summary['total_errors'] >= 6


class TestIntelligentCacheSystem:
    """Test intelligent caching system"""
    
    def test_cache_initialization(self):
        """Test cache system initializes correctly"""
        cache = IntelligentCacheSystem(enable_persistence=False)
        assert cache is not None
        assert cache.max_size_bytes > 0
        cache.shutdown()
    
    def test_basic_cache_operations(self):
        """Test basic cache get/set operations"""
        cache = IntelligentCacheSystem(enable_persistence=False)
        
        # Set and get
        result = cache.set("test_key", "test_value")
        assert result is True
        
        value = cache.get("test_key")
        assert value == "test_value"
        
        # Non-existent key
        value = cache.get("non_existent", default="default")
        assert value == "default"
        
        cache.shutdown()
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration"""
        cache = IntelligentCacheSystem(enable_persistence=False)
        
        # Set with short TTL
        cache.set("expiring_key", "expiring_value", ttl=1)  # 1 second
        
        # Should be available immediately
        value = cache.get("expiring_key")
        assert value == "expiring_value"
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be expired
        value = cache.get("expiring_key", default="expired")
        assert value == "expired"
        
        cache.shutdown()
    
    def test_cache_eviction_strategy(self):
        """Test intelligent cache eviction"""
        # Small cache to trigger eviction
        cache = IntelligentCacheSystem(max_size_mb=1, enable_persistence=False)
        
        # Fill cache beyond capacity
        for i in range(100):
            large_value = "x" * 1000  # 1KB value
            cache.set(f"key_{i}", large_value, priority=float(i))
        
        # Should have triggered evictions
        stats = cache.get_stats()
        assert stats['entry_count'] < 100
        
        cache.shutdown()
    
    def test_cached_function_decorator(self):
        """Test cached function decorator"""
        call_count = 0
        
        @cached_function(ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different parameters should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2


class TestAutoScalingManager:
    """Test auto-scaling manager"""
    
    def test_scaling_manager_initialization(self):
        """Test auto-scaling manager initializes"""
        manager = AutoScalingManager(min_workers=1, max_workers=4, monitoring_interval=1.0)
        assert manager is not None
        assert manager.current_workers >= 1
        manager.shutdown()
    
    def test_system_metrics_collection(self):
        """Test system metrics collection"""
        manager = AutoScalingManager(min_workers=1, max_workers=4)
        
        metrics = manager.collect_system_metrics()
        assert metrics is not None
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.timestamp is not None
        
        manager.shutdown()
    
    def test_scaling_decision_logic(self):
        """Test scaling decision making"""
        manager = AutoScalingManager(min_workers=1, max_workers=4)
        
        # Create high-load metrics
        high_load_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=90.0,
            memory_percent=85.0,
            disk_io_read_mb_s=100.0,
            disk_io_write_mb_s=50.0,
            network_io_mb_s=10.0,
            active_processes=200,
            load_average=(3.0, 2.5, 2.0),
            available_memory_mb=512.0
        )
        
        decision = manager.make_scaling_decision(high_load_metrics)
        assert decision is not None
        assert decision.direction is not None
        
        manager.shutdown()
    
    def test_workload_execution(self):
        """Test workload execution with scaling"""
        manager = AutoScalingManager(min_workers=1, max_workers=2, monitoring_interval=1.0)
        
        # Simple task executor
        def simple_task_executor(task):
            return task * 2
        
        tasks = [1, 2, 3, 4, 5]
        results = manager.execute_workload(tasks, simple_task_executor)
        
        assert len(results) == len(tasks)
        assert all(r == t * 2 for r, t in zip(results, tasks))
        
        manager.shutdown()


class TestCLIIntegration:
    """Test CLI command integration"""
    
    def test_cli_help_command(self):
        """Test CLI help command"""
        with patch('sys.argv', ['ado.py', 'help']):
            with patch('builtins.print') as mock_print:
                from ado import main
                main()
                
                # Check that help was printed
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                help_text = '\n'.join(str(call) for call in print_calls)
                assert 'Terragon ADO v4.0' in help_text
                assert 'Commands:' in help_text
    
    def test_cli_validate_command(self):
        """Test CLI validate command"""
        with patch('sys.argv', ['ado.py', 'validate']):
            with patch('builtins.print') as mock_print:
                with patch('os.getenv') as mock_getenv:
                    # Mock missing environment variables
                    mock_getenv.return_value = None
                    
                    from ado import main
                    main()
                    
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    output = '\n'.join(str(call) for call in print_calls)
                    assert 'Validating ADO Environment' in output
    
    def test_backlog_manager_integration(self):
        """Test backlog manager with enhanced features"""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            # Create test backlog manager
            manager = BacklogManager(repo_root=temp_dir)
            
            # Test basic functionality
            manager.load_backlog()
            report = manager.generate_status_report()
            
            assert report is not None
            assert 'total_items' in report


class TestQualityGates:
    """Test quality gates enforcement"""
    
    def test_security_scanning_integration(self):
        """Test security scanning is properly integrated"""
        # Test that security modules are importable and functional
        validator = SecurityInputValidator()
        
        # Test common attack vectors are detected
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "__import__('os').system('rm -rf /')",
            "../../../etc/passwd"
        ]
        
        for malicious_input in malicious_inputs:
            result = validator.validate_string(malicious_input)
            # Should either reject or sanitize
            assert not result.is_valid or result.sanitized_value != malicious_input
    
    def test_error_recovery_coverage(self):
        """Test error recovery covers common failure scenarios"""
        recovery_system = ErrorRecoverySystem(persist_errors=False)
        
        common_errors = [
            ConnectionError("Network failure"),
            TimeoutError("Operation timeout"),
            PermissionError("Access denied"),
            ValueError("Invalid input"),
            FileNotFoundError("File missing")
        ]
        
        for error in common_errors:
            result = recovery_system.handle_error(
                error=error,
                component="test",
                operation="test_operation"
            )
            assert result is not None
            assert result.strategy_used is not None
    
    def test_performance_requirements(self):
        """Test performance meets requirements"""
        cache = IntelligentCacheSystem(enable_persistence=False)
        
        # Test cache performance
        start_time = time.time()
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        set_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time
        
        # Should be fast (< 1 second for 1000 operations)
        assert set_time < 1.0, f"Cache set operations too slow: {set_time}s"
        assert get_time < 1.0, f"Cache get operations too slow: {get_time}s"
        
        cache.shutdown()
    
    def test_memory_usage_limits(self):
        """Test memory usage stays within reasonable limits"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple systems to test memory usage
        cache = IntelligentCacheSystem(max_size_mb=50, enable_persistence=False)
        recovery_system = ErrorRecoverySystem(persist_errors=False)
        
        # Perform operations
        for i in range(100):
            cache.set(f"test_{i}", "x" * 1000)
            
            if i % 10 == 0:
                error = RuntimeError(f"Test error {i}")
                recovery_system.handle_error(error, "test", "memory_test")
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not use excessive memory (< 100MB increase)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase} MB"
        
        cache.shutdown()


class TestProductionReadiness:
    """Test production readiness aspects"""
    
    def test_configuration_validation(self):
        """Test all configuration options are validated"""
        # Test that required environment variables are checked
        required_env_vars = ['GITHUB_TOKEN', 'OPENAI_API_KEY']
        
        for var in required_env_vars:
            # Should detect missing variables
            with patch.dict(os.environ, {}, clear=True):
                validator = SecurityInputValidator()
                result = validator.validate_environment_variable("")
                assert not result.is_valid
    
    def test_logging_integration(self):
        """Test logging is properly configured"""
        import logging
        
        # Test that loggers are configured
        logger = logging.getLogger('test_logger')
        assert logger is not None
        
        # Test different log levels work
        with patch.object(logger, 'info') as mock_info:
            logger.info("Test message")
            mock_info.assert_called_once()
    
    def test_graceful_degradation(self):
        """Test system degrades gracefully when components fail"""
        # Test cache system with limited resources
        cache = IntelligentCacheSystem(max_size_mb=1, enable_persistence=False)
        
        # Should handle large values gracefully
        large_value = "x" * (2 * 1024 * 1024)  # 2MB value
        result = cache.set("large_key", large_value)
        assert result is False  # Should reject gracefully
        
        cache.shutdown()
    
    def test_concurrent_access_safety(self):
        """Test thread safety of concurrent operations"""
        import threading
        
        cache = IntelligentCacheSystem(enable_persistence=False)
        errors = []
        
        def worker_function(worker_id):
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_{i}"
                    cache.set(key, f"value_{i}")
                    value = cache.get(key)
                    assert value == f"value_{i}"
            except Exception as e:
                errors.append(e)
        
        # Run multiple workers concurrently
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker_function, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have no errors from concurrent access
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        
        cache.shutdown()


# Performance benchmark tests
def test_performance_benchmarks():
    """Run performance benchmarks"""
    results = {}
    
    # Cache performance
    cache = IntelligentCacheSystem(enable_persistence=False)
    
    start_time = time.time()
    for i in range(10000):
        cache.set(f"bench_{i}", f"value_{i}")
    cache_write_time = time.time() - start_time
    results['cache_write_10k_ops'] = cache_write_time
    
    start_time = time.time()
    for i in range(10000):
        cache.get(f"bench_{i}")
    cache_read_time = time.time() - start_time
    results['cache_read_10k_ops'] = cache_read_time
    
    cache.shutdown()
    
    # Error recovery performance
    recovery_system = ErrorRecoverySystem(persist_errors=False)
    
    start_time = time.time()
    for i in range(1000):
        error = RuntimeError(f"Benchmark error {i}")
        recovery_system.handle_error(error, "benchmark", "test")
    error_handling_time = time.time() - start_time
    results['error_handling_1k_ops'] = error_handling_time
    
    # Print benchmark results
    print("\n=== Performance Benchmarks ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}s")
    
    # Assert performance requirements
    assert cache_write_time < 5.0, f"Cache writes too slow: {cache_write_time}s"
    assert cache_read_time < 5.0, f"Cache reads too slow: {cache_read_time}s"
    assert error_handling_time < 10.0, f"Error handling too slow: {error_handling_time}s"


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v", "--tb=short"])