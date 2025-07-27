#!/usr/bin/env python3
"""
Performance tests for ADO components
"""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

from backlog_manager import BacklogManager


@pytest.mark.slow
class TestPerformance:
    """Performance test cases."""
    
    def test_backlog_loading_performance(
        self, 
        performance_timer,
        temp_workspace,
        backlog_item_factory
    ):
        """Test backlog loading performance with many items."""
        # Create many backlog items
        manager = BacklogManager(str(temp_workspace))
        
        # Generate 1000 items
        items = []
        for i in range(1000):
            item = backlog_item_factory(id_suffix=f"{i:03d}")
            items.append(item)
        
        manager.items = items
        
        # Time the WSJF calculation
        performance_timer.start()
        prioritized = manager.get_prioritized_items()
        performance_timer.stop()
        
        # Performance assertion
        assert performance_timer.elapsed < 1.0  # Should complete in under 1 second
        assert len(prioritized) == 1000
    
    def test_wsjf_calculation_performance(self, performance_timer):
        """Test WSJF calculation performance."""
        performance_timer.start()
        
        # Calculate WSJF for many items
        for i in range(10000):
            score = BacklogManager._calculate_wsjf_score(
                value=i % 10 + 1,
                time_criticality=i % 10 + 1,
                risk_reduction=i % 10 + 1,
                effort=i % 5 + 1
            )
            assert score > 0
        
        performance_timer.stop()
        
        # Should be very fast
        assert performance_timer.elapsed < 0.1
    
    @pytest.mark.benchmark
    def test_concurrent_backlog_access(
        self,
        temp_workspace,
        backlog_item_factory
    ):
        """Test concurrent access to backlog manager."""
        manager = BacklogManager(str(temp_workspace))
        
        # Create test items
        items = []
        for i in range(100):
            item = backlog_item_factory(id_suffix=f"{i:03d}")
            items.append(item)
        
        manager.items = items
        
        def access_backlog():
            """Simulate concurrent backlog access."""
            prioritized = manager.get_prioritized_items()
            ready = manager.get_ready_items()
            report = manager.generate_status_report()
            return len(prioritized), len(ready), report['total_items']
        
        # Test concurrent access
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_backlog) for _ in range(50)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        
        # All operations should complete successfully
        assert len(results) == 50
        assert all(result[0] == 100 for result in results)  # All items prioritized
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0