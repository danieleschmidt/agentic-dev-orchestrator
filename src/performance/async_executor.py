#!/usr/bin/env python3
"""
Asynchronous execution engine for high-performance task processing
"""

import asyncio
import time
import concurrent.futures
from typing import Dict, List, Optional, Awaitable, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from autonomous_executor import ExecutionResult, AutonomousExecutor
from backlog_manager import BacklogManager, BacklogItem


@dataclass
class ExecutionStats:
    """Performance execution statistics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    concurrent_executions: int = 0
    throughput_per_second: float = 0.0
    execution_times: List[float] = field(default_factory=list)


class AsyncAutonomousExecutor:
    """High-performance asynchronous autonomous executor"""
    
    def __init__(self, repo_root: str = ".", max_concurrent: int = 5, 
                 enable_caching: bool = True):
        self.repo_root = Path(repo_root)
        self.backlog_manager = BacklogManager(repo_root)
        self.base_executor = AutonomousExecutor(repo_root)
        self.max_concurrent = max_concurrent
        self.enable_caching = enable_caching
        self.logger = logging.getLogger("async_executor")
        self.stats = ExecutionStats()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.cache = {} if enable_caching else None
        
    async def execute_task_async(self, item: BacklogItem) -> ExecutionResult:
        """Execute single task asynchronously"""
        async with self.semaphore:
            self.stats.concurrent_executions += 1
            start_time = time.time()
            
            try:
                # Check cache first
                if self.enable_caching:
                    cache_key = f"{item.id}:{hash(str(item.description))}"
                    if cache_key in self.cache:
                        self.logger.info(f"Cache hit for task {item.id}")
                        result = self.cache[cache_key]
                        return result
                
                # Execute in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    result = await loop.run_in_executor(
                        executor, 
                        self.base_executor.execute_micro_cycle_full, 
                        item
                    )
                
                # Cache successful results
                if self.enable_caching and result.success:
                    self.cache[cache_key] = result
                
                execution_time = time.time() - start_time
                self._update_stats(execution_time, result.success)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Async execution failed for {item.id}: {e}")
                self._update_stats(execution_time, False)
                
                return ExecutionResult(
                    success=False,
                    item_id=item.id,
                    error_message=str(e),
                    execution_time=execution_time
                )
            finally:
                self.stats.concurrent_executions -= 1
    
    def _update_stats(self, execution_time: float, success: bool):
        """Update execution statistics"""
        self.stats.total_tasks += 1
        self.stats.execution_times.append(execution_time)
        
        if success:
            self.stats.completed_tasks += 1
        else:
            self.stats.failed_tasks += 1
        
        # Update timing stats
        self.stats.max_execution_time = max(self.stats.max_execution_time, execution_time)
        self.stats.min_execution_time = min(self.stats.min_execution_time, execution_time)
        
        if self.stats.execution_times:
            self.stats.avg_execution_time = sum(self.stats.execution_times) / len(self.stats.execution_times)
    
    async def execute_batch_async(self, items: List[BacklogItem]) -> List[ExecutionResult]:
        """Execute multiple tasks concurrently"""
        self.logger.info(f"Starting async batch execution of {len(items)} tasks")
        start_time = time.time()
        
        # Execute all tasks concurrently
        tasks = [self.execute_task_async(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ExecutionResult(
                    success=False,
                    item_id=items[i].id,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        # Calculate throughput
        total_time = time.time() - start_time
        self.stats.throughput_per_second = len(items) / total_time if total_time > 0 else 0
        
        self.logger.info(f"Batch execution completed in {total_time:.2f}s, throughput: {self.stats.throughput_per_second:.2f} tasks/sec")
        return processed_results
    
    async def continuous_execution_loop(self, max_iterations: int = 100) -> Dict:
        """Continuous asynchronous execution loop"""
        self.logger.info("Starting continuous async execution loop")
        results = {
            "start_time": time.time(),
            "completed_items": [],
            "failed_items": [],
            "escalated_items": [],
            "performance_stats": {}
        }
        
        for iteration in range(max_iterations):
            self.logger.info(f"Async iteration {iteration + 1}")
            
            # Load and discover tasks
            try:
                self.backlog_manager.load_backlog()
                new_items = self.backlog_manager.continuous_discovery()
                if new_items > 0:
                    self.logger.info(f"Discovered {new_items} new items")
            except Exception as e:
                self.logger.error(f"Failed to load backlog: {e}")
                continue
            
            # Get ready items
            ready_items = [item for item in self.backlog_manager.items if item.status == "READY"]
            if not ready_items:
                self.logger.info("No ready items found, sleeping...")
                await asyncio.sleep(1)
                continue
            
            # Filter out high-risk items (would escalate)
            safe_items = [item for item in ready_items 
                         if not self.base_executor.is_high_risk_or_ambiguous(item)]
            
            if not safe_items:
                self.logger.info("No safe items for async execution")
                await asyncio.sleep(1)
                continue
            
            # Execute batch
            batch_results = await self.execute_batch_async(safe_items[:self.max_concurrent])
            
            # Process results
            for result in batch_results:
                if result.success:
                    results["completed_items"].append({
                        "id": result.item_id,
                        "execution_time": result.execution_time
                    })
                else:
                    results["failed_items"].append({
                        "id": result.item_id,
                        "error": result.error_message
                    })
            
            # Save progress
            self.backlog_manager.save_backlog()
            
            # Short sleep between iterations
            await asyncio.sleep(0.1)
        
        # Final stats
        results["performance_stats"] = {
            "total_tasks": self.stats.total_tasks,
            "completed_tasks": self.stats.completed_tasks,
            "failed_tasks": self.stats.failed_tasks,
            "success_rate": self.stats.completed_tasks / max(self.stats.total_tasks, 1),
            "avg_execution_time": self.stats.avg_execution_time,
            "max_execution_time": self.stats.max_execution_time,
            "throughput_per_second": self.stats.throughput_per_second,
            "cache_hits": len(self.cache) if self.cache else 0
        }
        
        results["end_time"] = time.time()
        total_time = results["end_time"] - results["start_time"]
        results["total_duration"] = total_time
        
        self.logger.info(f"Async execution completed in {total_time:.2f}s")
        return results
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            "stats": {
                "total_tasks": self.stats.total_tasks,
                "completed_tasks": self.stats.completed_tasks,
                "failed_tasks": self.stats.failed_tasks,
                "success_rate": self.stats.completed_tasks / max(self.stats.total_tasks, 1),
                "avg_execution_time": self.stats.avg_execution_time,
                "max_execution_time": self.stats.max_execution_time,
                "min_execution_time": self.stats.min_execution_time if self.stats.min_execution_time != float('inf') else 0,
                "concurrent_executions": self.stats.concurrent_executions,
                "throughput_per_second": self.stats.throughput_per_second
            },
            "configuration": {
                "max_concurrent": self.max_concurrent,
                "caching_enabled": self.enable_caching,
                "cache_size": len(self.cache) if self.cache else 0
            },
            "timestamp": time.time()
        }
    
    def clear_cache(self):
        """Clear execution cache"""
        if self.cache:
            self.cache.clear()
            self.logger.info("Execution cache cleared")


class LoadBalancer:
    """Load balancer for distributing tasks across multiple executors"""
    
    def __init__(self, executors: List[AsyncAutonomousExecutor]):
        self.executors = executors
        self.current_index = 0
        self.logger = logging.getLogger("load_balancer")
    
    def get_next_executor(self) -> AsyncAutonomousExecutor:
        """Get next executor using round-robin"""
        executor = self.executors[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.executors)
        return executor
    
    def get_least_loaded_executor(self) -> AsyncAutonomousExecutor:
        """Get executor with lowest current load"""
        return min(self.executors, key=lambda e: e.stats.concurrent_executions)
    
    async def distribute_tasks(self, items: List[BacklogItem]) -> List[ExecutionResult]:
        """Distribute tasks across executors"""
        self.logger.info(f"Distributing {len(items)} tasks across {len(self.executors)} executors")
        
        # Group tasks by executor
        executor_tasks = {executor: [] for executor in self.executors}
        
        for item in items:
            executor = self.get_least_loaded_executor()
            executor_tasks[executor].append(item)
        
        # Execute all groups concurrently
        batch_tasks = []
        for executor, tasks in executor_tasks.items():
            if tasks:
                batch_tasks.append(executor.execute_batch_async(tasks))
        
        if not batch_tasks:
            return []
        
        # Gather all results
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        return all_results


async def main():
    """Async main function for testing"""
    executor = AsyncAutonomousExecutor(max_concurrent=10, enable_caching=True)
    
    # Run continuous execution
    results = await executor.continuous_execution_loop(max_iterations=5)
    
    print("Async Execution Results:")
    print(json.dumps(results, indent=2, default=str))
    
    print("\nPerformance Metrics:")
    metrics = executor.get_performance_metrics()
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())