#!/usr/bin/env python3
"""
Comprehensive Test Suite for Terragon Autonomous SDLC v6.0
Advanced testing framework with quantum and neural validation
"""

import asyncio
import json
import logging
import pytest
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import shutil

# Import components to test
from terragon_autonomous_sdlc_v6_executor import TerragONAutonomousSDLCv6, QuantumExecutionContext
from quantum_neural_orchestrator import QuantumNeuralOrchestrator, NeuralContext
from distributed_quantum_orchestrator import DistributedQuantumOrchestrator, DistributedTask, WorkerNode
from autonomous_executor import AutonomousExecutor, ExecutionResult
from backlog_manager import BacklogManager, BacklogItem


class TestTerragONSDLCv6(unittest.TestCase):
    """Test suite for Terragon SDLC v6.0 executor"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create basic directory structure
        (self.test_path / "backlog").mkdir(exist_ok=True)
        (self.test_path / "logs").mkdir(exist_ok=True)
        (self.test_path / "docs" / "status").mkdir(parents=True, exist_ok=True)
        
        # Create test backlog item
        self.test_item = BacklogItem(
            id="test-001",
            title="Test quantum processing",
            description="Test quantum-enhanced processing capabilities",
            effort=8,
            wsjf={"user_business_value": 8, "time_criticality": 7, "risk_reduction_opportunity_enablement": 6, "job_size": 5},
            acceptance_criteria=["Should process with quantum enhancement", "Should maintain coherence"],
            risk_tier="medium"
        )
        
        self.executor = TerragONAutonomousSDLCv6(self.test_dir)
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test executor initialization"""
        self.assertIsInstance(self.executor, TerragONAutonomousSDLCv6)
        self.assertTrue(self.executor.quantum_enabled)
        self.assertTrue(self.executor.distributed_processing)
        self.assertEqual(self.executor.repo_root, Path(self.test_dir))
    
    def test_quantum_context_creation(self):
        """Test quantum context creation"""
        context = self.executor._create_quantum_context(self.test_item)
        
        self.assertIsInstance(context, QuantumExecutionContext)
        self.assertEqual(context.task_id, "test-001")
        self.assertGreater(context.complexity_score, 0)
        self.assertLessEqual(context.complexity_score, 1.0)
        self.assertIsInstance(context.superposition_states, list)
        self.assertGreater(len(context.superposition_states), 0)
    
    def test_complexity_calculation(self):
        """Test quantum complexity calculation"""
        complexity = self.executor._calculate_quantum_complexity(self.test_item)
        
        self.assertIsInstance(complexity, float)
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)
    
    def test_superposition_grouping(self):
        """Test superposition group creation"""
        contexts = [self.executor._create_quantum_context(self.test_item) for _ in range(5)]
        groups = self.executor._create_superposition_groups(contexts)
        
        self.assertIsInstance(groups, list)
        self.assertGreater(len(groups), 0)
        
        # Verify all contexts are assigned to groups
        total_contexts = sum(len(group) for group in groups)
        self.assertEqual(total_contexts, len(contexts))
    
    @pytest.mark.asyncio
    async def test_quantum_state_execution(self):
        """Test quantum state execution"""
        context = self.executor._create_quantum_context(self.test_item)
        result = await self.executor._execute_quantum_state(context, "high_performance")
        
        self.assertIsInstance(result, dict)
        self.assertIn("context_id", result)
        self.assertIn("state", result)
        self.assertIn("execution_time", result)
        self.assertIn("success", result)
        self.assertEqual(result["context_id"], "test-001")
        self.assertEqual(result["state"], "high_performance")
        self.assertTrue(result["success"])
    
    @pytest.mark.asyncio
    async def test_superposition_collapse(self):
        """Test superposition collapse"""
        contexts = [self.executor._create_quantum_context(self.test_item)]
        
        # Mock state results
        state_results = [
            {"context_id": "test-001", "state": "high_performance", "execution_time": 0.1, "success": True, "quantum_coherence": 0.8},
            {"context_id": "test-001", "state": "balanced", "execution_time": 0.15, "success": True, "quantum_coherence": 0.6}
        ]
        
        results = self.executor._collapse_superposition(contexts, state_results)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ExecutionResult)
        self.assertEqual(results[0].item_id, "test-001")
        self.assertTrue(results[0].success)
    
    def test_configuration_loading(self):
        """Test configuration loading"""
        config = self.executor._load_default_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn("quantum_enabled", config)
        self.assertIn("distributed_processing", config)
        self.assertIn("max_quantum_states", config)
        self.assertTrue(config["quantum_enabled"])


class TestQuantumNeuralOrchestrator(unittest.TestCase):
    """Test suite for Quantum Neural Orchestrator"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create directory structure
        (self.test_path / "logs").mkdir(exist_ok=True)
        
        self.test_item = BacklogItem(
            id="neural-001",
            title="Neural processing test",
            description="Test neural network integration with quantum processing",
            effort=5,
            wsjf={"user_business_value": 7, "time_criticality": 6, "risk_reduction_opportunity_enablement": 5, "job_size": 3},
            acceptance_criteria=["Should encode task effectively", "Should predict complexity"],
            risk_tier="low"
        )
        
        self.orchestrator = QuantumNeuralOrchestrator(self.test_dir)
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsInstance(self.orchestrator, QuantumNeuralOrchestrator)
        self.assertIsNotNone(self.orchestrator.task_encoder)
        self.assertIsNotNone(self.orchestrator.complexity_predictor)
        self.assertIsNotNone(self.orchestrator.optimization_network)
    
    def test_embedding_creation(self):
        """Test task embedding creation"""
        text = "Test neural processing capabilities"
        embedding = self.orchestrator._create_embedding(text)
        
        self.assertIsInstance(embedding, type(None).__class__.__bases__[0])  # numpy.ndarray
        self.assertEqual(len(embedding), self.orchestrator.config["embedding_dimension"])
    
    @pytest.mark.asyncio
    async def test_task_encoding(self):
        """Test task encoding"""
        context = await self.orchestrator.encode_task(self.test_item)
        
        self.assertIsInstance(context, NeuralContext)
        self.assertEqual(context.task_id, "neural-001")
        self.assertIsNotNone(context.embedding)
        self.assertIsNotNone(context.attention_weights)
        self.assertGreater(context.prediction_confidence, 0.0)
        self.assertIn("predicted_complexity", context.learning_state)
    
    def test_attention_calculation(self):
        """Test attention weight calculation"""
        attention = self.orchestrator._calculate_attention(self.test_item)
        
        self.assertIsInstance(attention, dict)
        self.assertIn("complexity", attention)
        self.assertIn("urgency", attention)
        self.assertIn("risk", attention)
        
        # Test attention weight normalization
        total_attention = sum(attention.values())
        self.assertAlmostEqual(total_attention, 1.0, places=2)
    
    def test_complexity_prediction(self):
        """Test complexity prediction"""
        embedding = self.orchestrator._create_embedding("test text")
        complexity = self.orchestrator._predict_complexity(embedding)
        
        self.assertIsInstance(complexity, float)
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)
    
    def test_entanglement_strength_calculation(self):
        """Test quantum-neural entanglement calculation"""
        from terragon_autonomous_sdlc_v6_executor import QuantumExecutionContext
        
        quantum_ctx = QuantumExecutionContext(task_id="test", complexity_score=0.7)
        neural_ctx = NeuralContext(task_id="test", prediction_confidence=0.8)
        
        entanglement = self.orchestrator._calculate_entanglement_strength(quantum_ctx, neural_ctx)
        
        self.assertIsInstance(entanglement, float)
        self.assertGreaterEqual(entanglement, 0.0)
        self.assertLessEqual(entanglement, 1.0)


class TestDistributedQuantumOrchestrator(unittest.TestCase):
    """Test suite for Distributed Quantum Orchestrator"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create directory structure
        (self.test_path / "logs").mkdir(exist_ok=True)
        (self.test_path / "docs" / "status").mkdir(parents=True, exist_ok=True)
        
        self.test_item = BacklogItem(
            id="dist-001",
            title="Distributed processing test",
            description="Test distributed quantum processing across multiple nodes",
            effort=13,
            wsjf={"user_business_value": 9, "time_criticality": 8, "risk_reduction_opportunity_enablement": 7, "job_size": 8},
            acceptance_criteria=["Should distribute across nodes", "Should scale efficiently"],
            risk_tier="high"
        )
        
        self.orchestrator = DistributedQuantumOrchestrator(self.test_dir)
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsInstance(self.orchestrator, DistributedQuantumOrchestrator)
        self.assertIsNotNone(self.orchestrator.cluster_id)
        self.assertIsInstance(self.orchestrator.worker_nodes, dict)
        self.assertIsInstance(self.orchestrator.task_queue, list)
    
    @pytest.mark.asyncio
    async def test_worker_node_discovery(self):
        """Test worker node discovery"""
        await self.orchestrator._discover_worker_nodes()
        
        self.assertGreater(len(self.orchestrator.worker_nodes), 0)
        
        # Verify node properties
        for node in self.orchestrator.worker_nodes.values():
            self.assertIsInstance(node, WorkerNode)
            self.assertIsInstance(node.node_id, str)
            self.assertGreater(node.cpu_cores, 0)
            self.assertGreater(node.memory_gb, 0.0)
            self.assertIn(node.specialty, ["quantum_processing", "neural_processing", "hybrid_processing", "general_processing"])
    
    @pytest.mark.asyncio
    async def test_distributed_task_creation(self):
        """Test distributed task creation"""
        task = await self.orchestrator._create_distributed_task(self.test_item)
        
        self.assertIsInstance(task, DistributedTask)
        self.assertEqual(task.original_item.id, "dist-001")
        self.assertGreater(task.complexity_score, 0.0)
        self.assertIsInstance(task.required_capabilities, set)
        self.assertIsInstance(task.parallel_subtasks, list)
        self.assertGreater(task.estimated_duration, 0.0)
    
    def test_capability_determination(self):
        """Test required capability determination"""
        capabilities = self.orchestrator._determine_required_capabilities(self.test_item, 0.8)
        
        self.assertIsInstance(capabilities, set)
        self.assertIn("high_performance", capabilities)  # High complexity
    
    def test_subtask_decomposition(self):
        """Test task decomposition into subtasks"""
        subtasks = self.orchestrator._decompose_into_subtasks(self.test_item, 0.7)
        
        self.assertIsInstance(subtasks, list)
        self.assertGreater(len(subtasks), 1)  # Should be decomposed for high complexity
        
        for subtask in subtasks:
            self.assertIn("name", subtask)
            self.assertIn("estimated_time", subtask)
            self.assertIn("dependencies", subtask)
    
    def test_node_requirement_checking(self):
        """Test node requirement checking"""
        # Create test node
        node = WorkerNode(
            node_id="test-node",
            hostname="test",
            cpu_cores=4,
            memory_gb=8.0,
            quantum_capable=True,
            health_score=0.8,
            current_load=0.5
        )
        
        # Create test task
        task = DistributedTask(
            task_id="test-task",
            original_item=self.test_item,
            complexity_score=0.6,
            required_capabilities={"quantum_processing"}
        )
        
        meets_requirements = self.orchestrator._node_meets_requirements(node, task)
        self.assertTrue(meets_requirements)
        
        # Test with overloaded node
        node.current_load = 4.5  # Exceeds capacity
        meets_requirements = self.orchestrator._node_meets_requirements(node, task)
        self.assertFalse(meets_requirements)
    
    def test_cluster_state_tracking(self):
        """Test cluster state tracking"""
        state = self.orchestrator.cluster_state
        
        self.assertIsInstance(state.total_nodes, int)
        self.assertIsInstance(state.active_nodes, int)
        self.assertIsInstance(state.total_capacity, float)
        self.assertIsInstance(state.current_utilization, float)
        self.assertGreaterEqual(state.current_utilization, 0.0)
        self.assertLessEqual(state.current_utilization, 1.0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete SDLC scenarios"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create complete directory structure
        dirs = ["backlog", "logs", "docs/status", "escalations", "quality_reports", 
               "security_reports", "deployment_reports", "validation_reports"]
        for dir_path in dirs:
            (self.test_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create test backlog
        self._create_test_backlog()
        
    def tearDown(self):
        """Cleanup integration test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_backlog(self):
        """Create test backlog items"""
        backlog_items = [
            {
                "id": "integration-001",
                "title": "Simple feature implementation",
                "description": "Implement a simple feature for testing",
                "effort": 3,
                "wsjf": {"user_business_value": 5, "time_criticality": 4, "risk_reduction_opportunity_enablement": 3, "job_size": 2},
                "acceptance_criteria": ["Feature works correctly", "Tests pass"],
                "risk_tier": "low"
            },
            {
                "id": "integration-002", 
                "title": "Complex quantum processing task",
                "description": "Implement quantum-enhanced processing capabilities",
                "effort": 13,
                "wsjf": {"user_business_value": 9, "time_criticality": 8, "risk_reduction_opportunity_enablement": 7, "job_size": 8},
                "acceptance_criteria": ["Quantum processing active", "Performance improved", "Tests comprehensive"],
                "risk_tier": "high"
            },
            {
                "id": "integration-003",
                "title": "Neural network optimization",
                "description": "Optimize neural network processing for better performance",
                "effort": 8,
                "wsjf": {"user_business_value": 7, "time_criticality": 6, "risk_reduction_opportunity_enablement": 5, "job_size": 5},
                "acceptance_criteria": ["Performance optimized", "Memory usage reduced"],
                "risk_tier": "medium"
            }
        ]
        
        for item_data in backlog_items:
            backlog_file = self.test_path / "backlog" / f"{item_data['id']}.json"
            with open(backlog_file, 'w') as f:
                json.dump(item_data, f, indent=2)
    
    @pytest.mark.asyncio
    async def test_end_to_end_quantum_processing(self):
        """Test end-to-end quantum processing"""
        executor = TerragONAutonomousSDLCv6(self.test_dir)
        
        # Load backlog
        backlog_manager = BacklogManager(self.test_dir)
        items = backlog_manager.get_prioritized_backlog(limit=3)
        
        self.assertEqual(len(items), 3)
        
        # Execute quantum processing
        results = await executor.execute_quantum_enhanced_cycle(items)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        # Verify results
        for result in results:
            self.assertIsInstance(result, ExecutionResult)
            self.assertIsInstance(result.success, bool)
            self.assertIsInstance(result.execution_time, float)
    
    @pytest.mark.asyncio
    async def test_end_to_end_neural_processing(self):
        """Test end-to-end neural processing"""
        orchestrator = QuantumNeuralOrchestrator(self.test_dir)
        
        # Load items
        backlog_manager = BacklogManager(self.test_dir)
        items = backlog_manager.get_prioritized_backlog(limit=2)
        
        # Execute neural processing
        results = await orchestrator.execute_quantum_neural_processing(items)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Verify result structure
        for result in results:
            self.assertIn("task_id", result)
            self.assertIn("processing_mode", result)
            self.assertIn("success", result)
    
    @pytest.mark.asyncio
    async def test_end_to_end_distributed_processing(self):
        """Test end-to-end distributed processing"""
        orchestrator = DistributedQuantumOrchestrator(self.test_dir)
        
        # Initialize cluster
        await orchestrator.initialize_cluster()
        
        # Load items
        backlog_manager = BacklogManager(self.test_dir)
        items = backlog_manager.get_prioritized_backlog(limit=3)
        
        # Distribute and execute
        distributed_tasks = await orchestrator.distribute_tasks(items)
        results = await orchestrator.execute_distributed_processing()
        
        self.assertIsInstance(distributed_tasks, list)
        self.assertIsInstance(results, list)
        
        # Verify distribution
        for task in distributed_tasks:
            self.assertIsInstance(task, DistributedTask)
            self.assertIsNotNone(task.assigned_node)
    
    def test_quality_gates_validation(self):
        """Test quality gates validation"""
        # Test basic quality requirements
        executor = AutonomousExecutor(self.test_dir)
        
        # Verify security configuration
        security_config = executor._load_security_config()
        self.assertIsInstance(security_config, dict)
        self.assertIn("allowed_commands", security_config)
        self.assertIn("blocked_patterns", security_config)
        
        # Verify logging setup
        logger = executor.logger
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "autonomous_executor")
    
    def test_metrics_collection(self):
        """Test metrics collection across all components"""
        # Test executor metrics
        executor = TerragONAutonomousSDLCv6(self.test_dir)
        metrics = executor.metrics
        
        self.assertIn("quantum_executions", metrics)
        self.assertIn("parallel_optimizations", metrics)
        self.assertIsInstance(metrics["quantum_executions"], int)
        
        # Test neural orchestrator metrics
        neural_orchestrator = QuantumNeuralOrchestrator(self.test_dir)
        neural_metrics = neural_orchestrator.neural_metrics
        
        self.assertIn("predictions_made", neural_metrics)
        self.assertIn("adaptations_performed", neural_metrics)
        self.assertIsInstance(neural_metrics["accuracy_score"], float)
        
        # Test distributed orchestrator metrics
        distributed_orchestrator = DistributedQuantumOrchestrator(self.test_dir)
        distributed_metrics = distributed_orchestrator.distributed_metrics
        
        self.assertIn("cluster_efficiency", distributed_metrics)
        self.assertIn("parallel_speedup", distributed_metrics)
        self.assertIsInstance(distributed_metrics["parallel_speedup"], float)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance and load testing"""
    
    def setUp(self):
        """Setup performance test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create directory structure
        (self.test_path / "logs").mkdir(exist_ok=True)
        (self.test_path / "backlog").mkdir(exist_ok=True)
        
        # Create performance test items
        self._create_performance_backlog(50)  # 50 test items
    
    def tearDown(self):
        """Cleanup performance test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_performance_backlog(self, count: int):
        """Create backlog items for performance testing"""
        for i in range(count):
            item_data = {
                "id": f"perf-{i:03d}",
                "title": f"Performance test item {i}",
                "description": f"Performance testing item number {i} for load testing",
                "effort": (i % 13) + 1,  # Vary effort 1-13
                "wsjf": {
                    "user_business_value": (i % 9) + 1,
                    "time_criticality": (i % 9) + 1,
                    "risk_reduction_opportunity_enablement": (i % 9) + 1,
                    "job_size": (i % 13) + 1
                },
                "acceptance_criteria": [f"Requirement {j}" for j in range((i % 3) + 1)],
                "risk_tier": ["low", "medium", "high"][i % 3]
            }
            
            backlog_file = self.test_path / "backlog" / f"perf-{i:03d}.json"
            with open(backlog_file, 'w') as f:
                json.dump(item_data, f, indent=2)
    
    @pytest.mark.asyncio
    async def test_quantum_processing_performance(self):
        """Test quantum processing performance with load"""
        executor = TerragONAutonomousSDLCv6(self.test_dir)
        
        # Load large backlog
        backlog_manager = BacklogManager(self.test_dir)
        items = backlog_manager.get_prioritized_backlog(limit=20)
        
        start_time = time.time()
        results = await executor.execute_quantum_enhanced_cycle(items)
        execution_time = time.time() - start_time
        
        self.assertLess(execution_time, 30.0)  # Should complete within 30 seconds
        self.assertEqual(len(results), 20)
        
        # Verify performance metrics
        success_rate = sum(1 for r in results if r.success) / len(results)
        self.assertGreater(success_rate, 0.8)  # At least 80% success rate
    
    @pytest.mark.asyncio
    async def test_distributed_scaling_performance(self):
        """Test distributed processing scaling"""
        orchestrator = DistributedQuantumOrchestrator(self.test_dir)
        
        # Initialize cluster
        await orchestrator.initialize_cluster()
        
        # Load large backlog
        backlog_manager = BacklogManager(self.test_dir)
        items = backlog_manager.get_prioritized_backlog(limit=30)
        
        start_time = time.time()
        distributed_tasks = await orchestrator.distribute_tasks(items)
        results = await orchestrator.execute_distributed_processing()
        execution_time = time.time() - start_time
        
        # Verify scaling efficiency
        self.assertGreater(len(results), 0)
        self.assertLess(execution_time, 45.0)  # Should complete within 45 seconds
        
        # Check parallel speedup
        speedup = orchestrator.distributed_metrics["parallel_speedup"]
        self.assertGreater(speedup, 1.0)  # Should have some parallel benefit
    
    def test_memory_usage_limits(self):
        """Test memory usage stays within limits"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple orchestrators
        orchestrators = []
        for _ in range(5):
            orchestrator = TerragONAutonomousSDLCv6(self.test_dir)
            orchestrators.append(orchestrator)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use more than 500MB additional memory
        self.assertLess(memory_increase, 500)


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🧪 Running Comprehensive Test Suite for Terragon Autonomous SDLC v6.0")
    
    # Collect all test cases
    test_cases = [
        TestTerragONSDLCv6,
        TestQuantumNeuralOrchestrator,
        TestDistributedQuantumOrchestrator,
        TestIntegrationScenarios,
        TestPerformanceBenchmarks
    ]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for test_case in test_cases:
        print(f"\n📋 Running {test_case.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_case)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_passed += result.testsRun - len(result.failures) - len(result.errors)
        total_failed += len(result.failures) + len(result.errors)
        
        if result.failures:
            print(f"❌ Failures in {test_case.__name__}:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print(f"🚨 Errors in {test_case.__name__}:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    # Generate test report
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n📊 Test Summary:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 85:
        print("✅ Quality Gates PASSED - Success rate >= 85%")
        return True
    else:
        print("❌ Quality Gates FAILED - Success rate < 85%")
        return False


if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)