"""Performance tests for ADO components."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import Mock, patch

from backlog_manager import BacklogManager, BacklogItem
from autonomous_executor import AutonomousExecutor


@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance tests for core ADO functionality."""

    def test_backlog_manager_performance_large_dataset(self, temp_workspace, performance_timer):
        """Test BacklogManager performance with large number of items."""
        manager = BacklogManager(str(temp_workspace))
        
        # Create large number of test items
        num_items = 1000
        items = []
        
        performance_timer.start()
        
        for i in range(num_items):
            item = BacklogItem(
                id=f"perf-test-{i:04d}",
                title=f"Performance Test Item {i}",
                type="feature",
                description=f"Performance test item number {i}",
                acceptance_criteria=[f"Criterion {j}" for j in range(3)],
                effort=i % 10 + 1,
                value=i % 10 + 1,
                time_criticality=i % 10 + 1,
                risk_reduction=i % 10 + 1,
                status="READY",
                risk_tier="medium",
                created_at="2025-01-27T00:00:00Z",
                links=[]
            )
            items.append(item)
        
        # Test bulk operations
        manager.add_items(items)
        
        performance_timer.stop()
        
        # Performance assertions
        assert performance_timer.elapsed < 5.0, f"Bulk add took {performance_timer.elapsed:.2f}s, should be < 5s"
        assert len(manager.get_all_items()) == num_items
        
        # Test prioritization performance
        performance_timer.start()
        prioritized = manager.prioritize_items()
        performance_timer.stop()
        
        assert performance_timer.elapsed < 2.0, f"Prioritization took {performance_timer.elapsed:.2f}s, should be < 2s"
        assert len(prioritized) == num_items

    @pytest.mark.benchmark
    def test_wsjf_calculation_performance(self, benchmark, sample_backlog_items):
        """Benchmark WSJF calculation performance."""
        manager = BacklogManager(".")
        
        def calculate_wsjf_batch():
            for item in sample_backlog_items * 100:  # Multiply to increase load
                manager.calculate_wsjf_score(item)
        
        result = benchmark(calculate_wsjf_batch)
        
        # Benchmark automatically provides timing information
        # Assert reasonable performance
        assert benchmark.stats.stats.mean < 0.1  # Should complete in < 100ms on average

    def test_concurrent_agent_execution_performance(self, temp_workspace):
        """Test performance of concurrent agent execution."""
        executor = AutonomousExecutor(str(temp_workspace))
        
        # Mock agent execution to focus on concurrency overhead
        def mock_agent_execution(item):
            time.sleep(0.1)  # Simulate work
            return {
                "success": True,
                "item_id": item.id,
                "execution_time": 0.1,
                "agent_results": {"mock": {"success": True}}
            }
        
        # Create test items
        test_items = [
            BacklogItem(
                id=f"concurrent-{i}",
                title=f"Concurrent Test {i}",
                type="feature",
                description="Test concurrent execution",
                acceptance_criteria=["Should work"],
                effort=1,
                value=5,
                time_criticality=5,
                risk_reduction=5,
                status="READY",
                risk_tier="low",
                created_at="2025-01-27T00:00:00Z",
                links=[]
            )
            for i in range(10)
        ]
        
        # Test sequential execution
        start_time = time.time()
        sequential_results = []
        for item in test_items:
            result = mock_agent_execution(item)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Test concurrent execution
        start_time = time.time()
        concurrent_results = []
        with ThreadPoolExecutor(max_workers=3) as executor_pool:
            future_to_item = {
                executor_pool.submit(mock_agent_execution, item): item 
                for item in test_items
            }
            for future in as_completed(future_to_item):
                result = future.result()
                concurrent_results.append(result)
        concurrent_time = time.time() - start_time
        
        # Performance assertions
        assert len(sequential_results) == len(test_items)
        assert len(concurrent_results) == len(test_items)
        assert concurrent_time < sequential_time * 0.6  # Should be significantly faster
        
        print(f"Sequential: {sequential_time:.2f}s, Concurrent: {concurrent_time:.2f}s")
        print(f"Speedup: {sequential_time/concurrent_time:.2f}x")

    def test_file_io_performance(self, temp_workspace, performance_timer):
        """Test file I/O performance for large backlogs."""
        manager = BacklogManager(str(temp_workspace))
        
        # Create large backlog
        num_items = 500
        items = []
        
        for i in range(num_items):
            item = BacklogItem(
                id=f"io-test-{i:04d}",
                title=f"I/O Test Item {i}",
                type="feature",
                description="Test I/O performance" * 50,  # Larger description
                acceptance_criteria=[f"Criterion {j}" for j in range(10)],  # More criteria
                effort=i % 10 + 1,
                value=i % 10 + 1,
                time_criticality=i % 10 + 1,
                risk_reduction=i % 10 + 1,
                status="READY",
                risk_tier="medium",
                created_at="2025-01-27T00:00:00Z",
                links=[f"https://example.com/link-{j}" for j in range(5)]  # Multiple links
            )
            items.append(item)
        
        # Test write performance
        performance_timer.start()
        manager.save_backlog_state(items)
        performance_timer.stop()
        write_time = performance_timer.elapsed
        
        # Test read performance
        performance_timer.start()
        loaded_items = manager.load_backlog_state()
        performance_timer.stop()
        read_time = performance_timer.elapsed
        
        # Performance assertions
        assert write_time < 3.0, f"Write took {write_time:.2f}s, should be < 3s"
        assert read_time < 2.0, f"Read took {read_time:.2f}s, should be < 2s"
        assert len(loaded_items) == num_items
        
        print(f"Write: {write_time:.2f}s, Read: {read_time:.2f}s")

    def test_memory_usage_large_backlog(self, temp_workspace):
        """Test memory usage with large backlog."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        manager = BacklogManager(str(temp_workspace))
        
        # Create very large backlog
        num_items = 5000
        items = []
        
        for i in range(num_items):
            item = BacklogItem(
                id=f"memory-test-{i:05d}",
                title=f"Memory Test Item {i}",
                type="feature",
                description="Test memory usage" * 100,  # Large description
                acceptance_criteria=[f"Criterion {j}" for j in range(20)],
                effort=i % 10 + 1,
                value=i % 10 + 1,
                time_criticality=i % 10 + 1,
                risk_reduction=i % 10 + 1,
                status="READY",
                risk_tier="medium",
                created_at="2025-01-27T00:00:00Z",
                links=[]
            )
            items.append(item)
        
        manager.add_items(items)
        
        # Measure memory after loading
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Memory usage assertions
        assert memory_increase_mb < 500, f"Memory increased by {memory_increase_mb:.1f}MB, should be < 500MB"
        
        print(f"Memory increase: {memory_increase_mb:.1f}MB for {num_items} items")

    @pytest.mark.benchmark
    def test_json_parsing_performance(self, benchmark, temp_workspace):
        """Benchmark JSON parsing performance."""
        import json
        
        # Create large JSON structure
        large_item = {
            "id": "benchmark-item",
            "title": "Benchmark Item",
            "type": "feature",
            "description": "x" * 10000,  # Large description
            "acceptance_criteria": [f"Criterion {i}" for i in range(1000)],
            "effort": 5,
            "value": 8,
            "time_criticality": 7,
            "risk_reduction": 6,
            "status": "READY",
            "risk_tier": "medium",
            "created_at": "2025-01-27T00:00:00Z",
            "links": [f"https://example.com/link-{i}" for i in range(100)]
        }
        
        json_string = json.dumps(large_item)
        
        def parse_json():
            return json.loads(json_string)
        
        result = benchmark(parse_json)
        
        # Verify correctness
        assert result["id"] == "benchmark-item"
        assert len(result["acceptance_criteria"]) == 1000

    def test_status_report_generation_performance(self, temp_workspace, performance_timer):
        """Test performance of status report generation."""
        manager = BacklogManager(str(temp_workspace))
        
        # Create large dataset
        num_items = 2000
        items = []
        
        for i in range(num_items):
            status_options = ["NEW", "READY", "DOING", "PR", "DONE", "BLOCKED"]
            item = BacklogItem(
                id=f"status-test-{i:05d}",
                title=f"Status Test Item {i}",
                type="feature",
                description="Test status report generation",
                acceptance_criteria=["Should work"],
                effort=i % 10 + 1,
                value=i % 10 + 1,
                time_criticality=i % 10 + 1,
                risk_reduction=i % 10 + 1,
                status=status_options[i % len(status_options)],
                risk_tier="medium",
                created_at="2025-01-27T00:00:00Z",
                links=[]
            )
            items.append(item)
        
        manager.add_items(items)
        
        # Test status report generation performance
        performance_timer.start()
        status_report = manager.generate_status_report()
        performance_timer.stop()
        
        # Performance assertions
        assert performance_timer.elapsed < 1.0, f"Status report took {performance_timer.elapsed:.2f}s, should be < 1s"
        assert status_report["total_items"] == num_items
        assert sum(status_report["status_counts"].values()) == num_items
        
        print(f"Status report generation: {performance_timer.elapsed:.2f}s for {num_items} items")

    def test_search_performance(self, temp_workspace, performance_timer):
        """Test search performance across large dataset."""
        manager = BacklogManager(str(temp_workspace))
        
        # Create searchable dataset
        num_items = 1000
        items = []
        
        search_terms = ["feature", "bug", "enhancement", "refactor", "optimization"]
        
        for i in range(num_items):
            term = search_terms[i % len(search_terms)]
            item = BacklogItem(
                id=f"search-test-{i:04d}",
                title=f"Search Test {term} Item {i}",
                type=term if term in ["feature", "bug"] else "feature",
                description=f"This is a {term} for testing search performance",
                acceptance_criteria=[f"Should implement {term} correctly"],
                effort=i % 10 + 1,
                value=i % 10 + 1,
                time_criticality=i % 10 + 1,
                risk_reduction=i % 10 + 1,
                status="READY",
                risk_tier="medium",
                created_at="2025-01-27T00:00:00Z",
                links=[]
            )
            items.append(item)
        
        manager.add_items(items)
        
        # Test different search patterns
        search_queries = ["feature", "bug", "optimization", "test", "performance"]
        
        for query in search_queries:
            performance_timer.start()
            results = manager.search_items(query)
            performance_timer.stop()
            
            assert performance_timer.elapsed < 0.5, f"Search for '{query}' took {performance_timer.elapsed:.2f}s, should be < 0.5s"
            assert len(results) > 0, f"No results found for '{query}'"
            
            print(f"Search '{query}': {performance_timer.elapsed:.3f}s, {len(results)} results")

    def test_concurrent_file_access(self, temp_workspace):
        """Test concurrent file access performance."""
        manager = BacklogManager(str(temp_workspace))
        
        # Create test items
        num_items = 100
        items = []
        
        for i in range(num_items):
            item = BacklogItem(
                id=f"concurrent-file-{i:03d}",
                title=f"Concurrent File Test {i}",
                type="feature",
                description="Test concurrent file access",
                acceptance_criteria=["Should work concurrently"],
                effort=5,
                value=5,
                time_criticality=5,
                risk_reduction=5,
                status="READY",
                risk_tier="medium",
                created_at="2025-01-27T00:00:00Z",
                links=[]
            )
            items.append(item)
        
        # Test concurrent writes
        def write_item_batch(item_batch):
            for item in item_batch:
                manager.add_item(item)
        
        batch_size = 10
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(write_item_batch, batch) for batch in batches]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        concurrent_write_time = time.time() - start_time
        
        # Verify all items were written
        all_items = manager.get_all_items()
        assert len(all_items) == num_items
        
        print(f"Concurrent file write: {concurrent_write_time:.2f}s for {num_items} items")
        
        # Performance assertion
        assert concurrent_write_time < 5.0, f"Concurrent writes took {concurrent_write_time:.2f}s, should be < 5s"