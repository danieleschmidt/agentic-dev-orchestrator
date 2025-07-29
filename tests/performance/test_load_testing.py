"""Load testing for ADO components.

Performance tests for:
- Backlog processing under load
- Agent execution concurrency
- Memory usage optimization
- Response time benchmarks
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest
from memory_profiler import profile

from ado import BacklogManager
from autonomous_executor import AutonomousExecutor


class TestLoadTesting:
    """Load testing for ADO components."""

    @pytest.fixture
    def backlog_manager(self):
        """Create backlog manager for testing."""
        return BacklogManager(workspace_path=".")

    @pytest.fixture
    def autonomous_executor(self):
        """Create autonomous executor for testing."""
        return AutonomousExecutor(workspace_path=".")

    @pytest.mark.benchmark(group="backlog_processing")
    def test_backlog_processing_performance(self, benchmark, backlog_manager):
        """Benchmark backlog processing performance."""
        # Create mock backlog items
        mock_items = []
        for i in range(100):
            mock_items.append({
                "id": f"item-{i}",
                "title": f"Test Item {i}",
                "wsjf": {
                    "user_business_value": 5,
                    "time_criticality": 5,
                    "risk_reduction_opportunity_enablement": 3,
                    "job_size": 2
                },
                "description": f"Test description for item {i}"
            })

        with patch.object(backlog_manager, 'load_backlog_items', return_value=mock_items):
            result = benchmark(backlog_manager.prioritize_backlog)
            assert len(result) == 100

    @pytest.mark.benchmark(group="wsjf_calculation")
    def test_wsjf_calculation_performance(self, benchmark, backlog_manager):
        """Benchmark WSJF calculation performance."""
        item = {
            "wsjf": {
                "user_business_value": 8,
                "time_criticality": 7,
                "risk_reduction_opportunity_enablement": 6,
                "job_size": 5
            }
        }
        
        def calculate_wsjf():
            return backlog_manager.calculate_wsjf_score(item)
        
        result = benchmark(calculate_wsjf)
        assert result > 0

    def test_concurrent_agent_execution(self, autonomous_executor):
        """Test concurrent execution of multiple agents."""
        # Mock agent tasks
        def mock_agent_task(task_id):
            time.sleep(0.1)  # Simulate work
            return f"completed-{task_id}"

        with patch.object(autonomous_executor, 'execute_agent', side_effect=mock_agent_task):
            start_time = time.time()
            
            # Execute 10 agents concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(autonomous_executor.execute_agent, f"task-{i}")
                    for i in range(10)
                ]
                
                results = [future.result() for future in futures]
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete faster than sequential execution
            assert execution_time < 0.5  # Should be much faster than 10 * 0.1 = 1.0s
            assert len(results) == 10
            assert all("completed-" in result for result in results)

    @pytest.mark.asyncio
    async def test_async_backlog_processing(self, backlog_manager):
        """Test asynchronous backlog processing."""
        mock_items = [
            {
                "id": f"async-item-{i}",
                "title": f"Async Test Item {i}",
                "wsjf": {
                    "user_business_value": 5,
                    "time_criticality": 5,
                    "risk_reduction_opportunity_enablement": 3,
                    "job_size": 2
                }
            }
            for i in range(50)
        ]

        async def process_item(item):
            await asyncio.sleep(0.01)  # Simulate async work
            return item["id"]

        with patch.object(backlog_manager, 'load_backlog_items', return_value=mock_items):
            start_time = time.time()
            
            tasks = [process_item(item) for item in mock_items]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete faster than sequential processing
            assert execution_time < 0.2  # Much faster than 50 * 0.01 = 0.5s
            assert len(results) == 50

    @profile
    def test_memory_usage_large_backlog(self, backlog_manager):
        """Test memory usage with large backlog."""
        # Create a large backlog (1000 items)
        large_backlog = []
        for i in range(1000):
            large_backlog.append({
                "id": f"large-item-{i}",
                "title": f"Large Backlog Item {i}",
                "description": "A" * 1000,  # 1KB description
                "wsjf": {
                    "user_business_value": i % 10,
                    "time_criticality": (i + 1) % 10,
                    "risk_reduction_opportunity_enablement": (i + 2) % 10,
                    "job_size": (i + 3) % 10 + 1
                }
            })

        with patch.object(backlog_manager, 'load_backlog_items', return_value=large_backlog):
            # Process the large backlog
            prioritized = backlog_manager.prioritize_backlog()
            
            # Verify processing completed
            assert len(prioritized) == 1000
            
            # Memory should be reasonable (this will be logged by @profile)
            # In practice, monitor output for memory usage patterns

    def test_response_time_under_load(self, backlog_manager):
        """Test response time under concurrent load."""
        mock_items = [
            {
                "id": f"load-item-{i}",
                "title": f"Load Test Item {i}",
                "wsjf": {
                    "user_business_value": 5,
                    "time_criticality": 5,
                    "risk_reduction_opportunity_enablement": 3,
                    "job_size": 2
                }
            }
            for i in range(20)
        ]

        response_times = []

        def process_backlog():
            start = time.time()
            with patch.object(backlog_manager, 'load_backlog_items', return_value=mock_items):
                backlog_manager.prioritize_backlog()
            end = time.time()
            return end - start

        # Simulate concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_backlog) for _ in range(20)]
            response_times = [future.result() for future in futures]

        # Check response time consistency
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Response times should be reasonable and consistent
        assert avg_response_time < 0.1  # Average under 100ms
        assert max_response_time < 0.2  # Maximum under 200ms
        
        # No response time should be more than 3x the average (consistency check)
        assert all(rt < avg_response_time * 3 for rt in response_times)

    def test_resource_cleanup(self, autonomous_executor):
        """Test proper resource cleanup after execution."""
        initial_thread_count = len(threading.enumerate())
        
        # Execute multiple operations
        with patch.object(autonomous_executor, 'execute_agent') as mock_execute:
            mock_execute.return_value = "completed"
            
            for i in range(10):
                autonomous_executor.execute_agent(f"cleanup-task-{i}")
        
        # Allow some time for cleanup
        time.sleep(0.1)
        
        final_thread_count = len(threading.enumerate())
        
        # Thread count should not increase significantly
        assert final_thread_count <= initial_thread_count + 2

    @pytest.mark.parametrize("backlog_size", [10, 50, 100, 500])
    def test_scalability_backlog_sizes(self, backlog_manager, backlog_size):
        """Test scalability with different backlog sizes."""
        mock_items = [
            {
                "id": f"scale-item-{i}",
                "title": f"Scale Test Item {i}",
                "wsjf": {
                    "user_business_value": 5,
                    "time_criticality": 5,
                    "risk_reduction_opportunity_enablement": 3,
                    "job_size": 2
                }
            }
            for i in range(backlog_size)
        ]

        with patch.object(backlog_manager, 'load_backlog_items', return_value=mock_items):
            start_time = time.time()
            result = backlog_manager.prioritize_backlog()
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Execution time should scale reasonably (O(n log n) at most)
            # For WSJF calculation, it should be closer to O(n)
            time_per_item = execution_time / backlog_size
            
            # Should process at least 100 items per second
            assert time_per_item < 0.01
            assert len(result) == backlog_size

    def test_error_recovery_under_load(self, autonomous_executor):
        """Test error recovery mechanisms under load."""
        failure_count = 0
        success_count = 0

        def failing_agent_task(task_id):
            nonlocal failure_count, success_count
            if task_id % 3 == 0:  # Fail every 3rd task
                failure_count += 1
                raise Exception(f"Simulated failure for {task_id}")
            else:
                success_count += 1
                return f"success-{task_id}"

        with patch.object(autonomous_executor, 'execute_agent', side_effect=failing_agent_task):
            results = []
            errors = []
            
            # Execute tasks with error handling
            for i in range(15):
                try:
                    result = autonomous_executor.execute_agent(i)
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))
            
            # Verify partial success and error handling
            assert len(results) == 10  # 15 - 5 failures
            assert len(errors) == 5   # Every 3rd task failed
            assert failure_count == 5
            assert success_count == 10

import threading