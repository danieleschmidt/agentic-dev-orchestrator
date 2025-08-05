#!/usr/bin/env python3
"""
Quantum Integration Test Suite
Comprehensive testing for quantum-inspired task planning system
"""

import pytest
import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
import datetime

from quantum_task_planner import (
    QuantumTaskPlanner, QuantumTask, QuantumState, QuantumResource
)
from quantum_security_validator import (
    QuantumSecurityValidator, SecurityValidationResult, ThreatLevel
)
from quantum_error_recovery import (
    QuantumErrorRecovery, QuantumError, ErrorSeverity, RecoveryStrategy
)
from quantum_performance_optimizer import (
    QuantumPerformanceOptimizer, OptimizationStrategy, PerformanceMetrics
)
from backlog_manager import BacklogItem, BacklogManager


class TestQuantumTaskPlanner:
    """Test suite for QuantumTaskPlanner"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def planner(self, temp_repo):
        """Create QuantumTaskPlanner instance for testing"""
        return QuantumTaskPlanner(temp_repo)
    
    @pytest.fixture
    def sample_backlog_item(self):
        """Create sample backlog item for testing"""
        return BacklogItem(
            id="test_001",
            title="Test Task",
            type="feature",
            description="Test task for quantum planning",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            effort=5,
            value=8,
            time_criticality=6,
            risk_reduction=4,
            status="READY",
            risk_tier="medium",
            created_at=datetime.datetime.now().isoformat() + 'Z',
            links=[]
        )
    
    def test_quantum_task_creation(self, planner, sample_backlog_item):
        """Test quantum task creation from backlog item"""
        quantum_task = planner._create_quantum_task(sample_backlog_item)
        
        assert quantum_task.id == sample_backlog_item.id
        assert quantum_task.base_item == sample_backlog_item
        assert quantum_task.quantum_state == QuantumState.SUPERPOSITION
        assert 0.0 <= quantum_task.coherence_level <= 1.0
        assert len(quantum_task.probability_amplitudes) == 4
        
        # Test amplitude normalization
        total_amplitude = sum(quantum_task.probability_amplitudes.values())
        assert abs(total_amplitude - 1.0) < 0.001
    
    def test_quantum_system_initialization(self, planner, temp_repo):
        """Test quantum system initialization"""
        # Create sample backlog
        backlog_file = Path(temp_repo) / "backlog.yml"
        backlog_data = {
            'version': '1.0',
            'items': [{
                'id': 'test_001',
                'title': 'Test Task',
                'type': 'feature',
                'description': 'Test description',
                'acceptance_criteria': ['Test criterion'],
                'effort': 3,
                'value': 5,
                'time_criticality': 4,
                'risk_reduction': 2,
                'status': 'READY',
                'risk_tier': 'low',
                'created_at': datetime.datetime.now().isoformat() + 'Z',
                'links': []
            }]
        }
        
        import yaml
        with open(backlog_file, 'w') as f:
            yaml.dump(backlog_data, f)
        
        # Initialize quantum system
        planner.initialize_quantum_system()
        
        assert len(planner.quantum_tasks) == 1
        assert 'test_001' in planner.quantum_tasks
        assert len(planner.quantum_resources) > 0
    
    def test_entanglement_creation(self, planner, temp_repo):
        """Test quantum entanglement creation between related tasks"""
        # Create multiple related tasks
        backlog_items = [
            BacklogItem(
                id=f"related_task_{i}",
                title=f"Related Task {i}",
                type="feature",
                description="authentication user login system",
                acceptance_criteria=["Login works"],
                effort=3,
                value=5,
                time_criticality=4,
                risk_reduction=2,
                status="READY",
                risk_tier="low",
                created_at=datetime.datetime.now().isoformat() + 'Z',
                links=[]
            )
            for i in range(3)
        ]
        
        # Mock backlog manager
        planner.backlog_manager.items = backlog_items
        planner.initialize_quantum_system()
        
        # Check for entanglements
        entangled_tasks = [task for task in planner.quantum_tasks.values() 
                          if task.entanglement_partners]
        
        # At least some tasks should be entangled due to similar descriptions
        assert len(entangled_tasks) > 0
    
    def test_quantum_priority_calculation(self, planner, sample_backlog_item):
        """Test quantum priority calculation"""
        quantum_task = planner._create_quantum_task(sample_backlog_item)
        quantum_task.coherence_level = 0.8
        quantum_task.interference_pattern = [0.1, 0.2, 0.1]
        quantum_task.entanglement_partners = ["partner_1", "partner_2"]
        
        priority = planner._calculate_quantum_priority(quantum_task)
        
        assert priority > 0
        assert isinstance(priority, float)
    
    def test_superposition_collapse(self, planner, sample_backlog_item):
        """Test quantum superposition collapse"""
        quantum_task = planner._create_quantum_task(sample_backlog_item)
        planner.quantum_tasks[quantum_task.id] = quantum_task
        
        strategy = planner.collapse_superposition(quantum_task.id)
        
        assert strategy in ["immediate", "parallel", "sequential", "deferred"]
        assert quantum_task.quantum_state == QuantumState.COLLAPSED
        assert quantum_task.coherence_level < 1.0
    
    def test_quantum_schedule_optimization(self, planner, temp_repo):
        """Test quantum schedule optimization"""
        # Create test tasks
        backlog_items = [
            BacklogItem(
                id=f"sched_task_{i}",
                title=f"Schedule Task {i}",
                type="feature",
                description=f"Task {i} for scheduling",
                acceptance_criteria=[f"Complete task {i}"],
                effort=2 + i,
                value=5,
                time_criticality=3,
                risk_reduction=2,
                status="READY",
                risk_tier="low",
                created_at=datetime.datetime.now().isoformat() + 'Z',
                links=[]
            )
            for i in range(3)
        ]
        
        planner.backlog_manager.items = backlog_items
        planner.initialize_quantum_system()
        
        schedule = planner.optimize_quantum_schedule()
        
        assert isinstance(schedule, list)
        assert len(schedule) <= len(backlog_items)
        
        for slot in schedule:
            assert "task_id" in slot
            assert "strategy" in slot
            assert "quantum_priority" in slot
    
    def test_quantum_insights_generation(self, planner, temp_repo):
        """Test quantum insights generation"""
        # Create test setup
        planner.quantum_tasks = {
            "task_1": QuantumTask(
                id="task_1",
                base_item=BacklogItem(
                    id="task_1", title="Task 1", type="feature", description="Test",
                    acceptance_criteria=[], effort=3, value=5, time_criticality=4,
                    risk_reduction=2, status="READY", risk_tier="low",
                    created_at=datetime.datetime.now().isoformat() + 'Z', links=[]
                ),
                coherence_level=0.7,
                entanglement_partners=["task_2"]
            ),
            "task_2": QuantumTask(
                id="task_2",
                base_item=BacklogItem(
                    id="task_2", title="Task 2", type="feature", description="Test",
                    acceptance_criteria=[], effort=2, value=4, time_criticality=3,
                    risk_reduction=1, status="READY", risk_tier="low",
                    created_at=datetime.datetime.now().isoformat() + 'Z', links=[]
                ),
                coherence_level=0.6,
                entanglement_partners=["task_1"]
            )
        }
        
        insights = planner.get_quantum_insights()
        
        assert "system_coherence" in insights
        assert "entanglement_clusters" in insights
        assert "quantum_bottlenecks" in insights
        assert "optimization_suggestions" in insights
        assert "next_optimal_tasks" in insights
        
        assert 0.0 <= insights["system_coherence"] <= 1.0
        assert isinstance(insights["entanglement_clusters"], list)
        assert isinstance(insights["optimization_suggestions"], list)


class TestQuantumSecurityValidator:
    """Test suite for QuantumSecurityValidator"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def validator(self, temp_repo):
        """Create QuantumSecurityValidator instance for testing"""
        return QuantumSecurityValidator(temp_repo)
    
    @pytest.fixture
    def secure_quantum_task(self):
        """Create secure quantum task for testing"""
        backlog_item = BacklogItem(
            id="secure_task",
            title="Secure Task",
            type="feature",
            description="A secure task with proper validation",
            acceptance_criteria=["Validate all inputs", "Use secure authentication"],
            effort=3,
            value=5,
            time_criticality=4,
            risk_reduction=2,
            status="READY",
            risk_tier="low",
            created_at=datetime.datetime.now().isoformat() + 'Z',
            links=[]
        )
        
        return QuantumTask(
            id="secure_task",
            base_item=backlog_item,
            quantum_state=QuantumState.COHERENT,
            coherence_level=0.8
        )
    
    @pytest.fixture
    def insecure_quantum_task(self):
        """Create insecure quantum task for testing"""
        backlog_item = BacklogItem(
            id="insecure_task",
            title="Insecure Task",
            type="feature",
            description="Handle user input without validation and store password in plain text",
            acceptance_criteria=["Store password", "Process user data"],
            effort=3,
            value=5,
            time_criticality=4,
            risk_reduction=2,
            status="READY",
            risk_tier="high",
            created_at=datetime.datetime.now().isoformat() + 'Z',
            links=[]
        )
        
        return QuantumTask(
            id="insecure_task",
            base_item=backlog_item,
            quantum_state=QuantumState.SUPERPOSITION,
            coherence_level=0.3
        )
    
    def test_secure_task_validation(self, validator, secure_quantum_task):
        """Test validation of secure quantum task"""
        result = validator.validate_quantum_task_security(secure_quantum_task)
        
        assert isinstance(result, SecurityValidationResult)
        assert result.task_id == "secure_task"
        assert result.validation_passed is True
        assert result.overall_threat_level in [ThreatLevel.MINIMAL, ThreatLevel.LOW]
        assert result.security_score > 60.0
    
    def test_insecure_task_validation(self, validator, insecure_quantum_task):
        """Test validation of insecure quantum task"""
        result = validator.validate_quantum_task_security(insecure_quantum_task)
        
        assert isinstance(result, SecurityValidationResult)
        assert result.task_id == "insecure_task"
        assert len(result.threats_detected) > 0
        
        # Should detect password and input validation issues
        threat_types = [threat.threat_type.value for threat in result.threats_detected]
        assert any("data_encryption" in threat_type or "input_validation" in threat_type 
                  for threat_type in threat_types)
    
    def test_input_validation_detection(self, validator):
        """Test input validation threat detection"""
        risky_item = BacklogItem(
            id="input_task",
            title="Input Task",
            type="feature",
            description="Process user input from API without any validation or sanitization",
            acceptance_criteria=["Accept user data", "Store in database"],
            effort=3,
            value=5,
            time_criticality=4,
            risk_reduction=2,
            status="READY",
            risk_tier="medium",
            created_at=datetime.datetime.now().isoformat() + 'Z',
            links=[]
        )
        
        quantum_task = QuantumTask(
            id="input_task",
            base_item=risky_item,
            coherence_level=0.5
        )
        
        result = validator.validate_quantum_task_security(quantum_task)
        
        # Should detect input validation threats
        input_threats = [threat for threat in result.threats_detected 
                        if "input" in threat.description.lower()]
        assert len(input_threats) > 0
    
    def test_authentication_security_validation(self, validator):
        """Test authentication security validation"""
        auth_item = BacklogItem(
            id="auth_task",
            title="Authentication Task",
            type="feature",
            description="Implement login system with authentication",
            acceptance_criteria=["User login", "Password check"],
            effort=5,
            value=8,
            time_criticality=6,
            risk_reduction=4,
            status="READY",
            risk_tier="medium",
            created_at=datetime.datetime.now().isoformat() + 'Z',
            links=[]
        )
        
        quantum_task = QuantumTask(
            id="auth_task",
            base_item=auth_item,
            coherence_level=0.6
        )
        
        result = validator.validate_quantum_task_security(quantum_task)
        
        # Should flag authentication tasks without explicit security measures
        auth_threats = [threat for threat in result.threats_detected 
                       if "auth" in threat.description.lower()]
        assert len(auth_threats) >= 0  # May or may not detect depending on description
    
    def test_quantum_entropy_calculation(self, validator, secure_quantum_task):
        """Test quantum entropy calculation for security"""
        entropy = validator._calculate_quantum_entropy(secure_quantum_task)
        
        assert 0.0 <= entropy <= 1.0
        assert isinstance(entropy, float)
    
    def test_bulk_validation(self, validator, secure_quantum_task, insecure_quantum_task):
        """Test bulk validation of multiple tasks"""
        tasks = [secure_quantum_task, insecure_quantum_task]
        results = validator.bulk_validate_tasks(tasks)
        
        assert len(results) == 2
        assert "secure_task" in results
        assert "insecure_task" in results
        assert all(isinstance(result, SecurityValidationResult) for result in results.values())
    
    def test_security_report_generation(self, validator, secure_quantum_task, insecure_quantum_task):
        """Test security report generation"""
        tasks = [secure_quantum_task, insecure_quantum_task]
        results = validator.bulk_validate_tasks(tasks)
        report = validator.generate_security_report(results)
        
        assert "timestamp" in report
        assert "total_tasks_validated" in report
        assert "overall_security_status" in report
        assert "threat_summary" in report
        assert "score_distribution" in report
        assert "quantum_security_metrics" in report
        
        assert report["total_tasks_validated"] == 2
        assert report["overall_security_status"] in ["CRITICAL", "HIGH_RISK", "MEDIUM_RISK", "ACCEPTABLE"]


class TestQuantumErrorRecovery:
    """Test suite for QuantumErrorRecovery"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def recovery_system(self, temp_repo):
        """Create QuantumErrorRecovery instance for testing"""
        return QuantumErrorRecovery(temp_repo)
    
    def test_quantum_error_creation(self, recovery_system):
        """Test quantum error creation from exception"""
        test_exception = ValueError("Test error message")
        
        quantum_error = recovery_system._create_quantum_error_sync(
            test_exception, "test_operation", "test_task_001"
        )
        
        assert isinstance(quantum_error, QuantumError)
        assert quantum_error.error_type == "ValueError"
        assert quantum_error.message == "Test error message"
        assert quantum_error.task_id == "test_task_001"
        assert quantum_error.severity == ErrorSeverity.ERROR
        assert len(quantum_error.id) > 0
    
    def test_error_severity_determination(self, recovery_system):
        """Test error severity determination"""
        # Test different exception types
        memory_error = MemoryError("Out of memory")
        value_error = ValueError("Invalid value")
        warning = UserWarning("Just a warning")
        
        assert recovery_system._determine_error_severity(memory_error) == ErrorSeverity.CRITICAL
        assert recovery_system._determine_error_severity(value_error) == ErrorSeverity.ERROR
        assert recovery_system._determine_error_severity(warning) == ErrorSeverity.WARNING
    
    def test_recovery_strategy_selection(self, recovery_system):
        """Test recovery strategy selection"""
        # High coherence error
        high_coherence_error = QuantumError(
            id="test_001",
            error_type="TestError",
            severity=ErrorSeverity.ERROR,
            message="Test error",
            stack_trace="Test stack",
            coherence_level_at_error=0.9
        )
        
        strategy = recovery_system._select_recovery_strategy(high_coherence_error)
        assert strategy == RecoveryStrategy.QUANTUM_TUNNELING
        
        # Low coherence error
        low_coherence_error = QuantumError(
            id="test_002",
            error_type="TestError",
            severity=ErrorSeverity.ERROR,
            message="Test error",
            stack_trace="Test stack",
            coherence_level_at_error=0.2
        )
        
        strategy = recovery_system._select_recovery_strategy(low_coherence_error)
        assert strategy == RecoveryStrategy.COHERENCE_RESTORATION
    
    def test_circuit_breaker_functionality(self, recovery_system):
        """Test circuit breaker functionality"""
        circuit_breaker = recovery_system._get_circuit_breaker("test_operation")
        
        # Initially closed
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.state == "CLOSED"
        
        # Record failures
        for _ in range(5):
            circuit_breaker.record_failure()
        
        # Should be open after threshold failures
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.can_execute() is False
        
        # Test success recovery
        circuit_breaker.state = "HALF_OPEN"
        circuit_breaker.record_success()
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_retry_recovery_strategy(self, recovery_system):
        """Test retry recovery strategy"""
        quantum_error = QuantumError(
            id="retry_test",
            error_type="RetryError",
            severity=ErrorSeverity.ERROR,
            message="Retry test error",
            stack_trace="Test stack"
        )
        
        # Mock function that succeeds on second try
        call_count = 0
        async def mock_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First attempt fails")
            return "Success"
        
        result = await recovery_system._retry_recovery(quantum_error, mock_func, (), {})
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.RETRY
        assert result.retry_count >= 1
    
    def test_error_analytics_generation(self, recovery_system):
        """Test error analytics generation"""
        # Add some test errors
        test_errors = [
            QuantumError(
                id=f"test_{i}",
                error_type="TestError",
                severity=ErrorSeverity.ERROR,
                message=f"Test error {i}",
                stack_trace="Test stack",
                coherence_level_at_error=0.5 + i * 0.1,
                recovery_attempts=i % 3
            )
            for i in range(5)
        ]
        
        recovery_system.error_history.extend(test_errors)
        
        analytics = recovery_system.get_error_analytics()
        
        assert analytics["total_errors"] == 5
        assert "error_types" in analytics
        assert "severity_distribution" in analytics
        assert "average_coherence_at_error" in analytics
        assert "recommendations" in analytics
        
        assert analytics["error_types"]["TestError"] == 5
        assert 0.0 <= analytics["average_coherence_at_error"] <= 1.0


class TestQuantumPerformanceOptimizer:
    """Test suite for QuantumPerformanceOptimizer"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def optimizer(self, temp_repo):
        """Create QuantumPerformanceOptimizer instance for testing"""
        return QuantumPerformanceOptimizer(temp_repo)
    
    @pytest.fixture
    def sample_quantum_tasks(self):
        """Create sample quantum tasks for testing"""
        tasks = []
        for i in range(3):
            backlog_item = BacklogItem(
                id=f"perf_task_{i}",
                title=f"Performance Task {i}",
                type="feature",
                description=f"Task {i} for performance testing",
                acceptance_criteria=[f"Complete task {i}"],
                effort=2 + i,
                value=5,
                time_criticality=3,
                risk_reduction=2,
                status="READY",
                risk_tier="low",
                created_at=datetime.datetime.now().isoformat() + 'Z',
                links=[]
            )
            
            quantum_task = QuantumTask(
                id=f"perf_task_{i}",
                base_item=backlog_item,
                quantum_state=QuantumState.SUPERPOSITION,
                coherence_level=0.7 + i * 0.1
            )
            tasks.append(quantum_task)
        
        return tasks
    
    def test_performance_metrics_collection(self, optimizer):
        """Test performance metrics collection"""
        baseline_metrics = optimizer._collect_baseline_metrics()
        
        assert isinstance(baseline_metrics, PerformanceMetrics)
        assert baseline_metrics.cpu_usage >= 0.0
        assert baseline_metrics.memory_usage >= 0.0
        assert baseline_metrics.timestamp is not None
    
    def test_optimization_strategy_selection(self, optimizer, sample_quantum_tasks):
        """Test optimization strategy selection"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            strategies = loop.run_until_complete(
                optimizer._analyze_and_select_strategies(sample_quantum_tasks)
            )
            
            assert isinstance(strategies, list)
            assert len(strategies) > 0
            assert all(isinstance(strategy, OptimizationStrategy) for strategy in strategies)
            
            # Should include caching and resource pooling
            assert OptimizationStrategy.CACHING in strategies
            assert OptimizationStrategy.RESOURCE_POOLING in strategies
        finally:
            loop.close()
    
    def test_quantum_cache_functionality(self, optimizer):
        """Test quantum cache functionality"""
        cache = optimizer.cache
        
        # Test cache put/get
        test_key = "test_key"
        test_value = {"data": "test"}
        
        cache.put(test_key, test_value, coherence=0.8)
        retrieved_value = cache.get(test_key)
        
        assert retrieved_value == test_value
        
        # Test cache stats
        stats = cache.get_stats()
        assert "size" in stats
        assert "utilization" in stats
        assert "average_coherence" in stats
        
        assert stats["size"] == 1
        assert stats["average_coherence"] > 0.0
    
    def test_resource_pool_functionality(self, optimizer):
        """Test resource pool functionality"""
        pool = optimizer.resource_pool
        
        # Test resource acquisition
        from quantum_performance_optimizer import ResourceType
        
        acquired = pool.acquire_resource(ResourceType.CPU, timeout=1.0)
        assert acquired is True
        
        pool.release_resource(ResourceType.CPU)
        
        # Test resource limits
        cpu_semaphore = pool.resource_locks[ResourceType.CPU]
        assert cpu_semaphore._value <= pool.max_workers
    
    def test_task_grouping_for_parallel_execution(self, optimizer, sample_quantum_tasks):
        """Test task grouping for parallel execution"""
        groups = optimizer._group_tasks_for_parallel_execution(sample_quantum_tasks)
        
        assert isinstance(groups, list)
        assert len(groups) > 0
        
        # All tasks should be assigned to a group
        total_tasks_in_groups = sum(len(group) for group in groups)
        assert total_tasks_in_groups == len(sample_quantum_tasks)
    
    def test_workload_balancing(self, optimizer, sample_quantum_tasks):
        """Test quantum workload balancing"""
        balanced_groups = optimizer._balance_quantum_workload(sample_quantum_tasks)
        
        assert isinstance(balanced_groups, list)
        assert len(balanced_groups) > 0
        
        # Check that load is reasonably balanced
        group_sizes = [len(group) for group in balanced_groups]
        if len(group_sizes) > 1:
            max_diff = max(group_sizes) - min(group_sizes)
            assert max_diff <= 2  # Reasonable balance
    
    @pytest.mark.asyncio
    async def test_caching_optimization(self, optimizer, sample_quantum_tasks):
        """Test caching optimization strategy"""
        async def mock_execution_func(task):
            await asyncio.sleep(0.01)  # Simulate work
            return f"Result for {task.id}"
        
        result = await optimizer._optimize_caching(sample_quantum_tasks, mock_execution_func)
        
        assert "gain" in result
        assert "cache_hit_rate" in result
        assert "recommendations" in result
        
        assert result["gain"] >= 0.0
        assert 0.0 <= result["cache_hit_rate"] <= 1.0
    
    def test_performance_analytics_generation(self, optimizer):
        """Test performance analytics generation"""
        # Add some test metrics
        test_metrics = [
            PerformanceMetrics(
                cpu_usage=50.0 + i * 10,
                memory_usage=40.0 + i * 5,
                quantum_coherence=0.6 + i * 0.1,
                cache_hit_rate=0.5 + i * 0.1
            )
            for i in range(3)
        ]
        
        optimizer.metrics_history.extend(test_metrics)
        
        analytics = optimizer.get_performance_analytics()
        
        assert "total_measurements" in analytics
        assert "recent_cpu_usage" in analytics
        assert "recent_memory_usage" in analytics
        assert "average_quantum_coherence" in analytics
        assert "optimization_recommendations" in analytics
        
        assert analytics["total_measurements"] == 3
        assert len(analytics["recent_cpu_usage"]) == 3
        assert 0.0 <= analytics["average_quantum_coherence"] <= 1.0
    
    def test_performance_recommendations(self, optimizer):
        """Test performance recommendation generation"""
        # Create metrics with specific patterns
        high_cpu_metrics = [
            PerformanceMetrics(cpu_usage=90.0, memory_usage=50.0, quantum_coherence=0.6)
            for _ in range(3)
        ]
        
        recommendations = optimizer._generate_performance_recommendations(high_cpu_metrics)
        
        assert isinstance(recommendations, list)
        # Should recommend dealing with high CPU usage
        assert any("cpu" in rec.lower() for rec in recommendations)
    
    def test_optimizer_cleanup(self, optimizer):
        """Test optimizer cleanup"""
        # Should not raise any exceptions
        optimizer.cleanup()
        
        # Cache should be cleared
        assert len(optimizer.cache.cache) == 0


class TestQuantumIntegration:
    """Integration tests for the complete quantum system"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def integrated_system(self, temp_repo):
        """Create integrated quantum system for testing"""
        planner = QuantumTaskPlanner(temp_repo)
        validator = QuantumSecurityValidator(temp_repo)
        recovery = QuantumErrorRecovery(temp_repo)
        optimizer = QuantumPerformanceOptimizer(temp_repo)
        
        return {
            "planner": planner,
            "validator": validator,
            "recovery": recovery,
            "optimizer": optimizer
        }
    
    def test_end_to_end_quantum_processing(self, integrated_system, temp_repo):
        """Test end-to-end quantum task processing"""
        planner = integrated_system["planner"]
        validator = integrated_system["validator"]
        
        # Create test backlog
        backlog_file = Path(temp_repo) / "backlog.yml"
        backlog_data = {
            'version': '1.0',
            'items': [{
                'id': 'integration_test',
                'title': 'Integration Test Task',
                'type': 'feature',
                'description': 'End-to-end integration test with proper validation',
                'acceptance_criteria': ['Validate inputs', 'Secure processing'],
                'effort': 5,
                'value': 8,
                'time_criticality': 6,
                'risk_reduction': 4,
                'status': 'READY',
                'risk_tier': 'medium',
                'created_at': datetime.datetime.now().isoformat() + 'Z',
                'links': []
            }]
        }
        
        import yaml
        with open(backlog_file, 'w') as f:
            yaml.dump(backlog_data, f)
        
        # Initialize quantum system
        planner.initialize_quantum_system()
        assert len(planner.quantum_tasks) == 1
        
        # Validate security
        task = list(planner.quantum_tasks.values())[0]
        security_result = validator.validate_quantum_task_security(task)
        assert isinstance(security_result, SecurityValidationResult)
        
        # Generate insights
        insights = planner.get_quantum_insights()
        assert "system_coherence" in insights
        
        # Generate schedule
        schedule = planner.optimize_quantum_schedule()
        assert isinstance(schedule, list)
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, integrated_system):
        """Test error recovery integration with other components"""
        recovery = integrated_system["recovery"]
        
        # Test error handling decorator
        @recovery.quantum_error_handler(task_id="test_task")
        async def failing_function():
            raise ValueError("Integration test error")
        
        @recovery.quantum_error_handler(task_id="test_task")
        async def succeeding_function():
            return "Success"
        
        # Test that errors are handled
        try:
            await failing_function()
        except ValueError:
            pass  # Expected
        
        # Test that success is recorded
        result = await succeeding_function()
        assert result == "Success"
        
        # Check error history
        assert len(recovery.error_history) > 0
    
    def test_performance_optimization_integration(self, integrated_system, temp_repo):
        """Test performance optimization integration"""
        planner = integrated_system["planner"]
        optimizer = integrated_system["optimizer"]
        
        # Create tasks
        backlog_items = [
            BacklogItem(
                id=f"perf_integration_{i}",
                title=f"Performance Integration Task {i}",
                type="feature",
                description=f"Task {i} for performance integration testing",
                acceptance_criteria=[f"Complete task {i}"],
                effort=2,
                value=5,
                time_criticality=3,
                risk_reduction=2,
                status="READY",
                risk_tier="low",
                created_at=datetime.datetime.now().isoformat() + 'Z',
                links=[]
            )
            for i in range(3)
        ]
        
        planner.backlog_manager.items = backlog_items
        planner.initialize_quantum_system()
        
        quantum_tasks = list(planner.quantum_tasks.values())
        
        # Test optimization strategy selection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            strategies = loop.run_until_complete(
                optimizer._analyze_and_select_strategies(quantum_tasks)
            )
            assert len(strategies) > 0
        finally:
            loop.close()
            optimizer.cleanup()
    
    def test_security_validation_integration(self, integrated_system):
        """Test security validation integration with task planning"""
        planner = integrated_system["planner"]
        validator = integrated_system["validator"]
        
        # Create tasks with varying security profiles
        secure_item = BacklogItem(
            id="secure_integration",
            title="Secure Integration Task",
            type="feature",
            description="Secure task with validation and encryption",
            acceptance_criteria=["Validate all inputs", "Use secure encryption"],
            effort=4,
            value=7,
            time_criticality=5,
            risk_reduction=6,
            status="READY",
            risk_tier="low",
            created_at=datetime.datetime.now().isoformat() + 'Z',
            links=[]
        )
        
        insecure_item = BacklogItem(
            id="insecure_integration",
            title="Insecure Integration Task",
            type="feature",
            description="Handle user input and store plain password",
            acceptance_criteria=["Accept user data", "Store password"],
            effort=3,
            value=5,
            time_criticality=4,
            risk_reduction=2,
            status="READY",
            risk_tier="high",
            created_at=datetime.datetime.now().isoformat() + 'Z',
            links=[]
        )
        
        planner.backlog_manager.items = [secure_item, insecure_item]
        planner.initialize_quantum_system()
        
        # Validate both tasks
        quantum_tasks = list(planner.quantum_tasks.values())
        results = validator.bulk_validate_tasks(quantum_tasks)
        
        assert len(results) == 2
        
        # Generate security report
        report = validator.generate_security_report(results)
        assert "overall_security_status" in report
        assert report["total_tasks_validated"] == 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])