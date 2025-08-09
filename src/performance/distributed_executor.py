#!/usr/bin/env python3
"""
Distributed Task Execution Engine for ADO
Implements concurrent processing with load balancing and auto-scaling
"""

import asyncio
import concurrent.futures
import json
import logging
import multiprocessing as mp
import queue
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import psutil
import statistics
from collections import defaultdict, deque


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ExecutorType(Enum):
    """Types of task executors"""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_IO = "async_io"
    DISTRIBUTED = "distributed"


@dataclass
class TaskDefinition:
    """Definition of a task to be executed"""
    id: str
    func_name: str
    args: Tuple = ()
    kwargs: Dict = None
    priority: int = 1
    max_retries: int = 3
    timeout: float = 300.0
    required_memory_mb: int = 512
    required_cpu_cores: int = 1
    executor_type: ExecutorType = ExecutorType.THREAD_POOL
    dependencies: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    memory_used_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    retry_count: int = 0
    

@dataclass
class WorkerMetrics:
    """Metrics for a worker instance"""
    worker_id: str
    is_active: bool
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    last_task_time: Optional[str] = None
    worker_type: ExecutorType = ExecutorType.THREAD_POOL


class LoadBalancer:
    """Intelligent load balancing for task distribution"""
    
    def __init__(self):
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.task_queue_sizes: Dict[str, int] = defaultdict(int)
        self.worker_load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
    def select_worker(self, task: TaskDefinition, available_workers: List[str]) -> Optional[str]:
        """Select the best worker for a given task"""
        if not available_workers:
            return None
            
        # Filter workers that can handle the task requirements
        suitable_workers = []
        for worker_id in available_workers:
            metrics = self.worker_metrics.get(worker_id)
            if metrics and metrics.is_active:
                # Check resource requirements (simplified)
                if self._worker_can_handle_task(worker_id, task):
                    suitable_workers.append(worker_id)
                    
        if not suitable_workers:
            return available_workers[0] if available_workers else None
            
        # Score workers based on multiple factors
        worker_scores = {}
        for worker_id in suitable_workers:
            score = self._calculate_worker_score(worker_id, task)
            worker_scores[worker_id] = score
            
        # Select worker with highest score (lowest load, best performance)
        best_worker = max(worker_scores, key=worker_scores.get)
        return best_worker
        
    def _worker_can_handle_task(self, worker_id: str, task: TaskDefinition) -> bool:
        """Check if worker can handle task requirements"""
        # For now, assume all workers can handle any task
        # In a real implementation, this would check:
        # - Available memory vs required_memory_mb
        # - Available CPU cores vs required_cpu_cores
        # - Worker type compatibility
        return True
        
    def _calculate_worker_score(self, worker_id: str, task: TaskDefinition) -> float:
        """Calculate a score for worker suitability (higher is better)"""
        metrics = self.worker_metrics.get(worker_id)
        if not metrics:
            return 0.0
            
        score = 100.0  # Base score
        
        # Factor 1: Success rate (higher is better)
        total_tasks = metrics.tasks_completed + metrics.tasks_failed
        if total_tasks > 0:
            success_rate = metrics.tasks_completed / total_tasks
            score += success_rate * 50
            
        # Factor 2: Queue size (lower is better)
        queue_size = self.task_queue_sizes.get(worker_id, 0)
        score -= queue_size * 10
        
        # Factor 3: Resource utilization (moderate is better)
        if metrics.avg_cpu_usage > 0:
            # Prefer workers with moderate CPU usage (not idle, not overloaded)
            optimal_cpu = 60.0  # 60% CPU usage is optimal
            cpu_penalty = abs(metrics.avg_cpu_usage - optimal_cpu) / 10
            score -= cpu_penalty
            
        # Factor 4: Average execution time (lower is better for simple tasks)
        if metrics.total_execution_time > 0 and metrics.tasks_completed > 0:
            avg_execution_time = metrics.total_execution_time / metrics.tasks_completed
            if avg_execution_time < task.timeout / 2:  # Fast worker
                score += 20
            elif avg_execution_time > task.timeout * 0.8:  # Slow worker
                score -= 20
                
        # Factor 5: Recent load history
        load_history = self.worker_load_history.get(worker_id, deque())
        if load_history:
            recent_avg_load = statistics.mean(load_history)
            score -= recent_avg_load * 5
            
        return max(0.0, score)
        
    def update_worker_metrics(self, worker_id: str, metrics: WorkerMetrics):
        """Update metrics for a worker"""
        self.worker_metrics[worker_id] = metrics
        
        # Update load history
        current_load = metrics.avg_cpu_usage + (self.task_queue_sizes.get(worker_id, 0) * 10)
        self.worker_load_history[worker_id].append(current_load)
        
    def update_queue_size(self, worker_id: str, size: int):
        """Update task queue size for a worker"""
        self.task_queue_sizes[worker_id] = size


class DistributedExecutor:
    """Distributed task execution engine with auto-scaling capabilities"""
    
    def __init__(
        self,
        max_workers: int = None,
        min_workers: int = 2,
        auto_scale: bool = True,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        metrics_collection_interval: float = 5.0
    ):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.min_workers = min_workers
        self.auto_scale = auto_scale
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.metrics_collection_interval = metrics_collection_interval
        
        # Core components
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.result_queue: queue.Queue = queue.Queue()
        self.load_balancer = LoadBalancer()
        
        # Worker management
        self.workers: Dict[str, concurrent.futures.Executor] = {}
        self.worker_futures: Dict[str, List[concurrent.futures.Future]] = defaultdict(list)
        self.worker_threads: List[threading.Thread] = []
        
        # Task tracking
        self.tasks: Dict[str, TaskDefinition] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.task_dependencies: Dict[str, set] = defaultdict(set)
        
        # Metrics and monitoring
        self.metrics_lock = threading.RLock()
        self.is_running = False
        self.start_time = None
        
        # Function registry for distributed execution
        self.function_registry: Dict[str, Callable] = {}
        
        self.logger = logging.getLogger("distributed_executor")
        
    def register_function(self, name: str, func: Callable):
        """Register a function for distributed execution"""
        self.function_registry[name] = func
        self.logger.debug(f"Registered function: {name}")
        
    def start(self):
        """Start the distributed executor"""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start with minimum number of workers
        for i in range(self.min_workers):
            self._create_worker(f"worker_{i}", ExecutorType.THREAD_POOL)
            
        # Start metrics collection thread
        metrics_thread = threading.Thread(target=self._metrics_collector, daemon=True)
        metrics_thread.start()
        self.worker_threads.append(metrics_thread)
        
        # Start auto-scaling thread if enabled
        if self.auto_scale:
            scaling_thread = threading.Thread(target=self._auto_scaler, daemon=True)
            scaling_thread.start()
            self.worker_threads.append(scaling_thread)
            
        # Start task dispatcher thread
        dispatcher_thread = threading.Thread(target=self._task_dispatcher, daemon=True)
        dispatcher_thread.start()
        self.worker_threads.append(dispatcher_thread)
        
        self.logger.info(f"Distributed executor started with {len(self.workers)} workers")
        
    def stop(self, timeout: float = 30.0):
        """Stop the distributed executor"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Shutdown all worker executors
        for worker_id, executor in self.workers.items():
            try:
                executor.shutdown(wait=False)
            except Exception as e:
                self.logger.warning(f"Error shutting down worker {worker_id}: {e}")
                
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=timeout / len(self.worker_threads) if self.worker_threads else timeout)
            
        self.workers.clear()
        self.worker_futures.clear()
        self.worker_threads.clear()
        
        self.logger.info("Distributed executor stopped")
        
    def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task for execution"""
        if not self.is_running:
            raise RuntimeError("Executor is not running")
            
        # Store task
        self.tasks[task.id] = task
        
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.task_results or self.task_results[dep_id].status != TaskStatus.COMPLETED:
                self.task_dependencies[task.id].add(dep_id)
                
        # Add to queue (priority queue: lower priority number = higher priority)
        # Negate priority so higher numbers come first
        self.task_queue.put((-task.priority, time.time(), task))
        
        # Create initial result
        self.task_results[task.id] = TaskResult(
            task_id=task.id,
            status=TaskStatus.PENDING
        )
        
        self.logger.debug(f"Submitted task {task.id} with priority {task.priority}")
        return task.id
        
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result for a specific task"""
        start_time = time.time()
        
        while self.is_running:
            result = self.task_results.get(task_id)
            if result and result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return result
                
            if timeout and (time.time() - start_time) > timeout:
                break
                
            time.sleep(0.1)
            
        return self.task_results.get(task_id)
        
    def get_all_results(self) -> Dict[str, TaskResult]:
        """Get all task results"""
        return self.task_results.copy()
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        result = self.task_results.get(task_id)
        if result and result.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            result.status = TaskStatus.CANCELLED
            result.end_time = datetime.now().isoformat()
            self.logger.info(f"Cancelled task {task_id}")
            return True
        return False
        
    def _create_worker(self, worker_id: str, executor_type: ExecutorType):
        """Create a new worker"""
        if executor_type == ExecutorType.THREAD_POOL:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix=f"ADO-{worker_id}"
            )
        elif executor_type == ExecutorType.PROCESS_POOL:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=2
            )
        else:
            # Default to thread pool
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix=f"ADO-{worker_id}"
            )
            
        self.workers[worker_id] = executor
        self.worker_futures[worker_id] = []
        
        # Initialize worker metrics
        metrics = WorkerMetrics(
            worker_id=worker_id,
            is_active=True,
            worker_type=executor_type
        )
        self.load_balancer.update_worker_metrics(worker_id, metrics)
        
        self.logger.debug(f"Created worker {worker_id} of type {executor_type.value}")
        
    def _remove_worker(self, worker_id: str):
        """Remove a worker"""
        if worker_id in self.workers:
            try:
                self.workers[worker_id].shutdown(wait=False)
                del self.workers[worker_id]
                del self.worker_futures[worker_id]
                
                # Update metrics
                metrics = self.load_balancer.worker_metrics.get(worker_id)
                if metrics:
                    metrics.is_active = False
                    
                self.logger.debug(f"Removed worker {worker_id}")
            except Exception as e:
                self.logger.error(f"Error removing worker {worker_id}: {e}")
                
    def _task_dispatcher(self):
        """Dispatch tasks to available workers"""
        while self.is_running:
            try:
                # Get next task from queue (with timeout to allow checking is_running)
                try:
                    priority, queued_at, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Check if task is cancelled
                result = self.task_results.get(task.id)
                if result and result.status == TaskStatus.CANCELLED:
                    continue
                    
                # Check dependencies
                if task.id in self.task_dependencies and self.task_dependencies[task.id]:
                    # Check if all dependencies are completed
                    pending_deps = set()
                    for dep_id in self.task_dependencies[task.id]:
                        dep_result = self.task_results.get(dep_id)
                        if not dep_result or dep_result.status != TaskStatus.COMPLETED:
                            pending_deps.add(dep_id)
                            
                    if pending_deps:
                        # Put task back in queue with slight delay
                        time.sleep(0.5)
                        self.task_queue.put((priority, time.time(), task))
                        continue
                    else:
                        # All dependencies satisfied
                        del self.task_dependencies[task.id]
                        
                # Select worker for task
                available_workers = [wid for wid, w in self.workers.items() if w]
                selected_worker = self.load_balancer.select_worker(task, available_workers)
                
                if not selected_worker:
                    # No available workers, put task back
                    self.task_queue.put((priority, time.time(), task))
                    time.sleep(1.0)
                    continue
                    
                # Execute task on selected worker
                self._execute_task(task, selected_worker)
                
            except Exception as e:
                self.logger.error(f"Error in task dispatcher: {e}")
                time.sleep(1.0)
                
    def _execute_task(self, task: TaskDefinition, worker_id: str):
        """Execute a task on a specific worker"""
        executor = self.workers.get(worker_id)
        if not executor:
            return
            
        # Update task status
        result = self.task_results[task.id]
        result.status = TaskStatus.RUNNING
        result.start_time = datetime.now().isoformat()
        result.worker_id = worker_id
        
        # Get function to execute
        func = self.function_registry.get(task.func_name)
        if not func:
            result.status = TaskStatus.FAILED
            result.error = f"Function '{task.func_name}' not registered"
            result.end_time = datetime.now().isoformat()
            return
            
        # Submit task to executor
        try:
            future = executor.submit(self._execute_task_wrapper, task, func)
            self.worker_futures[worker_id].append(future)
            
            # Set up completion callback
            future.add_done_callback(
                lambda f: self._handle_task_completion(task.id, worker_id, f)
            )
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now().isoformat()
            self.logger.error(f"Failed to submit task {task.id}: {e}")
            
    def _execute_task_wrapper(self, task: TaskDefinition, func: Callable) -> Any:
        """Wrapper for task execution with monitoring"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Execute the actual function
            result = func(*task.args, **task.kwargs)
            return result
            
        except Exception as e:
            raise e
        finally:
            # Collect execution metrics
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = max(0, end_memory - start_memory)
            
            # Store metrics in task result (will be updated by completion handler)
            result = self.task_results.get(task.id)
            if result:
                result.execution_time = execution_time
                result.memory_used_mb = memory_used
                
    def _handle_task_completion(self, task_id: str, worker_id: str, future: concurrent.futures.Future):
        """Handle task completion"""
        result = self.task_results.get(task_id)
        if not result:
            return
            
        try:
            if future.cancelled():
                result.status = TaskStatus.CANCELLED
            elif future.exception():
                result.status = TaskStatus.FAILED
                result.error = str(future.exception())
                
                # Check if we should retry
                task = self.tasks.get(task_id)
                if task and result.retry_count < task.max_retries:
                    result.retry_count += 1
                    result.status = TaskStatus.RETRYING
                    
                    # Resubmit task with delay
                    retry_delay = min(2.0 ** result.retry_count, 60.0)  # Exponential backoff
                    threading.Timer(retry_delay, lambda: self.task_queue.put((-task.priority, time.time(), task))).start()
                    return
            else:
                result.status = TaskStatus.COMPLETED
                result.result = future.result()
                
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            
        result.end_time = datetime.now().isoformat()
        
        # Clean up future from worker list
        if worker_id in self.worker_futures:
            try:
                self.worker_futures[worker_id].remove(future)
            except ValueError:
                pass
                
        self.logger.debug(f"Task {task_id} completed with status {result.status.value}")
        
    def _metrics_collector(self):
        """Collect metrics from workers"""
        while self.is_running:
            try:
                with self.metrics_lock:
                    for worker_id, executor in self.workers.items():
                        metrics = self.load_balancer.worker_metrics.get(worker_id)
                        if metrics:
                            # Update queue size
                            queue_size = len(self.worker_futures.get(worker_id, []))
                            self.load_balancer.update_queue_size(worker_id, queue_size)
                            
                            # Update task counts from results
                            completed = sum(1 for r in self.task_results.values() 
                                          if r.worker_id == worker_id and r.status == TaskStatus.COMPLETED)
                            failed = sum(1 for r in self.task_results.values() 
                                       if r.worker_id == worker_id and r.status == TaskStatus.FAILED)
                                       
                            metrics.tasks_completed = completed
                            metrics.tasks_failed = failed
                            
                            # Calculate total execution time
                            total_time = sum(r.execution_time for r in self.task_results.values() 
                                           if r.worker_id == worker_id and r.execution_time > 0)
                            metrics.total_execution_time = total_time
                            
                            # Update system metrics
                            try:
                                process = psutil.Process()
                                metrics.avg_cpu_usage = process.cpu_percent()
                                metrics.avg_memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
                            self.load_balancer.update_worker_metrics(worker_id, metrics)
                            
                time.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.metrics_collection_interval)
                
    def _auto_scaler(self):
        """Auto-scale workers based on load"""
        while self.is_running:
            try:
                time.sleep(10.0)  # Check every 10 seconds
                
                if not self.auto_scale:
                    continue
                    
                # Calculate current load
                total_pending = self.task_queue.qsize()
                total_running = sum(len(futures) for futures in self.worker_futures.values())
                current_workers = len(self.workers)
                
                # Calculate load ratio
                total_tasks = total_pending + total_running
                load_ratio = total_tasks / max(current_workers, 1)
                
                # Scale up if load is high
                if (load_ratio > self.scale_up_threshold and 
                    current_workers < self.max_workers and 
                    total_pending > 2):
                    
                    new_worker_id = f"worker_{int(time.time())}"
                    self._create_worker(new_worker_id, ExecutorType.THREAD_POOL)
                    self.logger.info(f"Scaled up: created worker {new_worker_id} (load ratio: {load_ratio:.2f})")
                    
                # Scale down if load is low
                elif (load_ratio < self.scale_down_threshold and 
                      current_workers > self.min_workers and 
                      total_pending == 0):
                    
                    # Find worker with least active tasks
                    worker_loads = {wid: len(futures) 
                                  for wid, futures in self.worker_futures.items()}
                    
                    if worker_loads:
                        least_busy_worker = min(worker_loads, key=worker_loads.get)
                        if worker_loads[least_busy_worker] == 0:  # No active tasks
                            self._remove_worker(least_busy_worker)
                            self.logger.info(f"Scaled down: removed worker {least_busy_worker} (load ratio: {load_ratio:.2f})")
                            
            except Exception as e:
                self.logger.error(f"Error in auto-scaler: {e}")
                
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        with self.metrics_lock:
            total_tasks = len(self.task_results)
            completed_tasks = sum(1 for r in self.task_results.values() if r.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for r in self.task_results.values() if r.status == TaskStatus.FAILED)
            pending_tasks = self.task_queue.qsize()
            running_tasks = sum(1 for r in self.task_results.values() if r.status == TaskStatus.RUNNING)
            
            # Calculate throughput
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            throughput = completed_tasks / max(uptime / 60, 1)  # tasks per minute
            
            # Worker metrics
            worker_metrics = [asdict(metrics) for metrics in self.load_balancer.worker_metrics.values()]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime,
                'total_workers': len(self.workers),
                'active_workers': sum(1 for m in self.load_balancer.worker_metrics.values() if m.is_active),
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'pending_tasks': pending_tasks,
                'running_tasks': running_tasks,
                'success_rate': completed_tasks / max(total_tasks, 1),
                'throughput_per_minute': throughput,
                'worker_metrics': worker_metrics
            }


# Example usage functions for testing
def example_cpu_intensive_task(iterations: int = 1000000) -> int:
    """Example CPU-intensive task"""
    total = 0
    for i in range(iterations):
        total += i * i
    return total
    

def example_io_task(duration: float = 1.0) -> str:
    """Example I/O task (simulated with sleep)"""
    import time
    time.sleep(duration)
    return f"Completed after {duration} seconds"
    

def example_failing_task(should_fail: bool = True) -> str:
    """Example task that can fail"""
    if should_fail:
        raise ValueError("This task is designed to fail")
    return "Task succeeded"


def main():
    """CLI entry point for distributed executor"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python distributed_executor.py <command> [options]")
        print("Commands:")
        print("  demo - Run demonstration")
        print("  benchmark <num_tasks> - Run performance benchmark")
        return
        
    command = sys.argv[1]
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    executor = DistributedExecutor(
        max_workers=8,
        min_workers=2,
        auto_scale=True
    )
    
    # Register example functions
    executor.register_function('cpu_task', example_cpu_intensive_task)
    executor.register_function('io_task', example_io_task)
    executor.register_function('failing_task', example_failing_task)
    
    try:
        executor.start()
        
        if command == "demo":
            print("🚀 Running distributed execution demo...")
            
            # Submit various tasks
            tasks = [
                TaskDefinition("cpu_1", "cpu_task", args=(500000,), priority=1),
                TaskDefinition("io_1", "io_task", args=(2.0,), priority=2),
                TaskDefinition("cpu_2", "cpu_task", args=(1000000,), priority=1),
                TaskDefinition("fail_1", "failing_task", args=(True,), max_retries=2, priority=3),
                TaskDefinition("io_2", "io_task", args=(1.0,), priority=2),
            ]
            
            # Submit all tasks
            task_ids = []
            for task in tasks:
                task_id = executor.submit_task(task)
                task_ids.append(task_id)
                print(f"Submitted task: {task_id}")
                
            # Wait for completion
            print("\nWaiting for tasks to complete...")
            start_time = time.time()
            
            while time.time() - start_time < 30:  # 30 second timeout
                completed = 0
                for task_id in task_ids:
                    result = executor.get_result(task_id)
                    if result and result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        completed += 1
                        
                if completed == len(task_ids):
                    break
                    
                time.sleep(0.5)
                
            # Print results
            print("\n📈 Results:")
            for task_id in task_ids:
                result = executor.get_result(task_id)
                if result:
                    print(f"  {task_id}: {result.status.value}")
                    if result.error:
                        print(f"    Error: {result.error}")
                    if result.execution_time > 0:
                        print(f"    Time: {result.execution_time:.2f}s")
                        
            # Print system metrics
            print("\n📄 System Metrics:")
            metrics = executor.get_system_metrics()
            print(f"  Total workers: {metrics['total_workers']}")
            print(f"  Success rate: {metrics['success_rate']:.2%}")
            print(f"  Throughput: {metrics['throughput_per_minute']:.2f} tasks/min")
            
        elif command == "benchmark" and len(sys.argv) > 2:
            num_tasks = int(sys.argv[2])
            print(f"🏁 Running benchmark with {num_tasks} tasks...")
            
            # Submit many CPU tasks
            start_time = time.time()
            task_ids = []
            
            for i in range(num_tasks):
                task = TaskDefinition(f"bench_{i}", "cpu_task", args=(100000,))
                task_id = executor.submit_task(task)
                task_ids.append(task_id)
                
            print(f"Submitted {num_tasks} tasks in {time.time() - start_time:.2f}s")
            
            # Wait for completion
            completed = 0
            while completed < num_tasks:
                time.sleep(1.0)
                completed = sum(1 for task_id in task_ids 
                               if executor.get_result(task_id) and 
                               executor.get_result(task_id).status in [TaskStatus.COMPLETED, TaskStatus.FAILED])
                print(f"Completed: {completed}/{num_tasks} ({completed/num_tasks:.1%})")
                
            total_time = time.time() - start_time
            print(f"\nBenchmark completed in {total_time:.2f}s")
            print(f"Throughput: {num_tasks/total_time:.2f} tasks/second")
            
            # Final metrics
            metrics = executor.get_system_metrics()
            print(f"Final worker count: {metrics['total_workers']}")
            print(f"Success rate: {metrics['success_rate']:.2%}")
            
        else:
            print(f"Unknown command: {command}")
            
    finally:
        print("\nShutting down executor...")
        executor.stop()
        print("Done.")


if __name__ == "__main__":
    main()
