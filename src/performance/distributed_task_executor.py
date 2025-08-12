#!/usr/bin/env python3
"""
Distributed Task Execution System
High-performance, scalable task execution with load balancing and auto-scaling
"""

import asyncio
import json
import time
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
import queue
import resource
import psutil
import os
import signal
from abc import ABC, abstractmethod
import pickle
from contextlib import contextmanager
import heapq


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ExecutorType(Enum):
    """Types of task executors"""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class TaskDefinition:
    """Definition of a distributed task"""
    task_id: str
    func_name: str
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)  # cpu, memory, io
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    estimated_duration: Optional[float] = None
    tags: Set[str] = field(default_factory=set)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value > other.priority.value  # Higher priority first


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    worker_id: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if task is in a terminal state"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]


@dataclass
class WorkerStats:
    """Statistics for a worker"""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    current_task: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate"""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time"""
        return self.total_execution_time / self.tasks_completed if self.tasks_completed > 0 else 0.0


class ResourceMonitor:
    """Monitor system resources for auto-scaling decisions"""
    
    def __init__(self):
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.max_samples = 60  # Keep last 60 samples
        
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Store samples for trend analysis
        self.cpu_samples.append(cpu_percent)
        self.memory_samples.append(memory_percent)
        
        # Keep only recent samples
        if len(self.cpu_samples) > self.max_samples:
            self.cpu_samples.pop(0)
        if len(self.memory_samples) > self.max_samples:
            self.memory_samples.pop(0)
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_usage": disk_usage,
            "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }
    
    def get_resource_trends(self) -> Dict[str, float]:
        """Analyze resource usage trends"""
        if len(self.cpu_samples) < 10:
            return {"cpu_trend": 0.0, "memory_trend": 0.0}
        
        # Simple trend calculation (recent average vs historical average)
        recent_cpu = sum(self.cpu_samples[-10:]) / 10
        historical_cpu = sum(self.cpu_samples[:-10]) / len(self.cpu_samples[:-10])
        
        recent_memory = sum(self.memory_samples[-10:]) / 10
        historical_memory = sum(self.memory_samples[:-10]) / len(self.memory_samples[:-10])
        
        return {
            "cpu_trend": recent_cpu - historical_cpu,
            "memory_trend": recent_memory - historical_memory,
            "cpu_average": recent_cpu,
            "memory_average": recent_memory
        }


class TaskQueue:
    """Thread-safe priority task queue with dependency handling"""
    
    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self._queue = []
        self._task_map: Dict[str, TaskDefinition] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._completed_tasks: Set[str] = set()
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        
    def put(self, task: TaskDefinition) -> bool:
        """Add task to queue"""
        with self._not_empty:
            if self.max_size and len(self._queue) >= self.max_size:
                return False
            
            heapq.heappush(self._queue, task)
            self._task_map[task.task_id] = task
            
            # Update dependency graph
            if task.dependencies:
                self._dependency_graph[task.task_id] = set(task.dependencies)
            
            self._not_empty.notify()
            return True
    
    def get_ready_task(self) -> Optional[TaskDefinition]:
        """Get next task that has no pending dependencies"""
        with self._not_empty:
            while True:
                if not self._queue:
                    return None
                
                # Find first task with satisfied dependencies
                for i, task in enumerate(self._queue):
                    if self._are_dependencies_satisfied(task.task_id):
                        # Remove task from queue
                        self._queue.pop(i)
                        heapq.heapify(self._queue)  # Restore heap property
                        return task
                
                # No ready tasks available
                return None
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied"""
        dependencies = self._dependency_graph.get(task_id, set())
        return dependencies.issubset(self._completed_tasks)
    
    def mark_completed(self, task_id: str):
        """Mark task as completed for dependency resolution"""
        with self._lock:
            self._completed_tasks.add(task_id)
            self._task_map.pop(task_id, None)
    
    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self._queue) == 0


class WorkerProcess:
    """Individual worker process for task execution"""
    
    def __init__(self, worker_id: str, task_registry: Dict[str, Callable]):
        self.worker_id = worker_id
        self.task_registry = task_registry
        self.current_task: Optional[str] = None
        self.stats = WorkerStats(worker_id=worker_id)
        
    def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute a single task"""
        result = TaskResult(task_id=task.task_id, status=TaskStatus.RUNNING)
        result.start_time = datetime.now()
        result.worker_id = self.worker_id
        
        self.current_task = task.task_id
        
        try:
            # Get function to execute
            if task.func_name not in self.task_registry:
                raise ValueError(f"Unknown task function: {task.func_name}")
            
            func = self.task_registry[task.func_name]
            
            # Set resource limits if specified
            self._set_resource_limits(task.resource_requirements)
            
            # Execute with timeout
            if task.timeout:
                result.result = self._execute_with_timeout(func, task.args, task.kwargs, task.timeout)
            else:
                result.result = func(*task.args, **task.kwargs)
            
            result.status = TaskStatus.COMPLETED
            self.stats.tasks_completed += 1
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            self.stats.tasks_failed += 1
        
        finally:
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
            self.stats.total_execution_time += result.execution_time
            self.current_task = None
        
        return result
    
    def _set_resource_limits(self, requirements: Dict[str, float]):
        """Set resource limits for task execution"""
        try:
            if 'memory' in requirements:
                # Set memory limit (in MB)
                memory_limit = int(requirements['memory'] * 1024 * 1024)  # Convert to bytes
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            if 'cpu' in requirements:
                # Set CPU time limit (in seconds)
                cpu_limit = int(requirements['cpu'])
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        
        except Exception as e:
            logging.warning(f"Could not set resource limits: {e}")
    
    def _execute_with_timeout(self, func: Callable, args: Tuple, kwargs: Dict, timeout: float) -> Any:
        """Execute function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task execution timed out after {timeout} seconds")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            return func(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


class DistributedTaskExecutor:
    """High-performance distributed task execution system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Core components
        self.task_queue = TaskQueue(max_size=self.config["queue"]["max_size"])
        self.task_registry: Dict[str, Callable] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.worker_stats: Dict[str, WorkerStats] = {}
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Execution pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # State management
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Auto-scaling
        self.min_workers = self.config["scaling"]["min_workers"]
        self.max_workers = self.config["scaling"]["max_workers"]
        self.current_workers = 0
        self.scaling_cooldown = timedelta(seconds=self.config["scaling"]["cooldown_seconds"])
        self.last_scaling_action = datetime.now() - self.scaling_cooldown
        
        # Monitoring and metrics
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "average_queue_size": 0.0,
            "worker_utilization": 0.0,
            "scaling_events": 0
        }
        
        # Setup logging
        self.logger = logging.getLogger("distributed_executor")
        
        # Initialize
        self._initialize_executors()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "queue": {
                "max_size": 10000
            },
            "scaling": {
                "enabled": True,
                "min_workers": 2,
                "max_workers": mp.cpu_count() * 2,
                "target_cpu_usage": 70.0,
                "target_memory_usage": 80.0,
                "scale_up_threshold": 80.0,
                "scale_down_threshold": 30.0,
                "cooldown_seconds": 60
            },
            "execution": {
                "default_timeout": 300.0,
                "heartbeat_interval": 30.0,
                "task_retry_delay": 2.0,
                "resource_monitoring": True
            },
            "performance": {
                "batch_size": 10,
                "prefetch_tasks": True,
                "optimize_for_throughput": True
            }
        }
    
    def _initialize_executors(self):
        """Initialize execution pools"""
        initial_workers = max(self.min_workers, mp.cpu_count())
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=initial_workers,
            thread_name_prefix="task_thread"
        )
        
        self.process_pool = ProcessPoolExecutor(
            max_workers=initial_workers
        )
        
        self.current_workers = initial_workers
        self.logger.info(f"Initialized executor pools with {initial_workers} workers")
    
    def register_task(self, name: str, func: Callable):
        """Register a task function"""
        self.task_registry[name] = func
        self.logger.info(f"Registered task function: {name}")
    
    def submit_task(self, task: TaskDefinition) -> str:
        """Submit task for execution"""
        if not self.is_running:
            raise RuntimeError("Executor is not running")
        
        if task.func_name not in self.task_registry:
            raise ValueError(f"Task function '{task.func_name}' not registered")
        
        # Add task to queue
        if self.task_queue.put(task):
            self.metrics["tasks_submitted"] += 1
            self.logger.debug(f"Task {task.task_id} submitted to queue")
            return task.task_id
        else:
            raise RuntimeError("Task queue is full")
    
    async def submit_task_async(self, func_name: str, *args, **kwargs) -> str:
        """Async convenience method to submit task"""
        task_id = hashlib.sha256(f"{func_name}_{time.time()}_{args}_{kwargs}".encode()).hexdigest()[:16]
        
        task = TaskDefinition(
            task_id=task_id,
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            priority=kwargs.pop('priority', TaskPriority.NORMAL),
            timeout=kwargs.pop('timeout', None),
            max_retries=kwargs.pop('max_retries', 3)
        )
        
        return self.submit_task(task)
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get task result (blocking)"""
        start_time = time.time()
        
        while True:
            if task_id in self.task_results:
                return self.task_results[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    async def get_result_async(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get task result (async)"""
        start_time = time.time()
        
        while True:
            if task_id in self.task_results:
                return self.task_results[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            await asyncio.sleep(0.1)
    
    def start(self):
        """Start the distributed executor"""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start background threads
        threading.Thread(target=self._task_dispatcher, daemon=True).start()
        threading.Thread(target=self._resource_monitor_thread, daemon=True).start()
        threading.Thread(target=self._auto_scaler_thread, daemon=True).start()
        threading.Thread(target=self._metrics_collector_thread, daemon=True).start()
        
        self.logger.info("Distributed task executor started")
    
    def stop(self, wait: bool = True):
        """Stop the distributed executor"""
        if not self.is_running:
            return
        
        self.logger.info("Shutting down distributed task executor...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        if wait:
            # Wait for current tasks to complete
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            if self.process_pool:
                self.process_pool.shutdown(wait=True)
        
        self.logger.info("Distributed task executor stopped")
    
    def _task_dispatcher(self):
        """Main task dispatching loop"""
        while self.is_running:
            try:
                # Get ready task from queue
                task = self.task_queue.get_ready_task()
                
                if task is None:
                    time.sleep(0.1)
                    continue
                
                # Dispatch task based on type
                self._dispatch_task(task)
                
            except Exception as e:
                self.logger.error(f"Error in task dispatcher: {e}")
                time.sleep(1.0)
    
    def _dispatch_task(self, task: TaskDefinition):
        """Dispatch single task to appropriate executor"""
        try:
            # Determine executor type based on task characteristics
            executor_type = self._choose_executor_type(task)
            
            if executor_type == ExecutorType.THREAD:
                future = self.thread_pool.submit(self._execute_task_wrapper, task)
            elif executor_type == ExecutorType.PROCESS:
                future = self.process_pool.submit(self._execute_task_wrapper, task)
            else:
                # Default to thread pool
                future = self.thread_pool.submit(self._execute_task_wrapper, task)
            
            # Add callback for result handling
            future.add_done_callback(lambda f: self._handle_task_completion(task.task_id, f))
            
        except Exception as e:
            # Create failed result
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            self.task_results[task.task_id] = result
            self.metrics["tasks_failed"] += 1
    
    def _choose_executor_type(self, task: TaskDefinition) -> ExecutorType:
        """Choose appropriate executor type for task"""
        # Use process pool for CPU-intensive tasks
        if 'cpu_intensive' in task.tags:
            return ExecutorType.PROCESS
        
        # Use thread pool for I/O tasks
        if 'io_intensive' in task.tags:
            return ExecutorType.THREAD
        
        # Default to thread pool for most tasks
        return ExecutorType.THREAD
    
    def _execute_task_wrapper(self, task: TaskDefinition) -> TaskResult:
        """Wrapper for task execution with retry logic"""
        worker = WorkerProcess(
            worker_id=f"worker_{threading.current_thread().ident}",
            task_registry=self.task_registry
        )
        
        for attempt in range(task.max_retries + 1):
            try:
                result = worker.execute_task(task)
                
                if result.status == TaskStatus.COMPLETED:
                    return result
                
                # Task failed, check if should retry
                if attempt < task.max_retries:
                    self.logger.warning(f"Task {task.task_id} failed (attempt {attempt + 1}/{task.max_retries + 1}), retrying...")
                    time.sleep(task.retry_delay * (2 ** attempt))  # Exponential backoff
                    result.retry_count = attempt + 1
                else:
                    return result
                    
            except Exception as e:
                if attempt < task.max_retries:
                    self.logger.warning(f"Task {task.task_id} exception (attempt {attempt + 1}/{task.max_retries + 1}): {e}")
                    time.sleep(task.retry_delay * (2 ** attempt))
                else:
                    return TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        retry_count=attempt
                    )
        
        # Should not reach here
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Max retries exceeded"
        )
    
    def _handle_task_completion(self, task_id: str, future):
        """Handle task completion"""
        try:
            result = future.result()
            self.task_results[task_id] = result
            
            # Mark as completed for dependency resolution
            if result.status == TaskStatus.COMPLETED:
                self.task_queue.mark_completed(task_id)
                self.metrics["tasks_completed"] += 1
                self.metrics["total_execution_time"] += result.execution_time or 0.0
            else:
                self.metrics["tasks_failed"] += 1
            
            # Cleanup old results to prevent memory leaks
            self._cleanup_old_results()
            
        except Exception as e:
            self.logger.error(f"Error handling task completion for {task_id}: {e}")
    
    def _cleanup_old_results(self):
        """Clean up old task results to prevent memory leaks"""
        if len(self.task_results) > 10000:  # Keep last 10k results
            # Remove oldest 1000 results
            oldest_tasks = sorted(self.task_results.items(), 
                                key=lambda x: x[1].end_time or datetime.now())[:1000]
            
            for task_id, _ in oldest_tasks:
                self.task_results.pop(task_id, None)
    
    def _resource_monitor_thread(self):
        """Monitor system resources"""
        while self.is_running:
            try:
                metrics = self.resource_monitor.collect_metrics()
                trends = self.resource_monitor.get_resource_trends()
                
                # Update worker stats
                for worker_id, stats in self.worker_stats.items():
                    stats.cpu_usage = metrics["cpu_percent"]
                    stats.memory_usage = metrics["memory_percent"]
                    stats.last_heartbeat = datetime.now()
                
                # Log resource usage periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    self.logger.info(f"Resource usage - CPU: {metrics['cpu_percent']:.1f}%, "
                                   f"Memory: {metrics['memory_percent']:.1f}%, "
                                   f"Queue size: {self.task_queue.size()}")
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
            
            time.sleep(self.config["execution"]["heartbeat_interval"])
    
    def _auto_scaler_thread(self):
        """Auto-scaling logic"""
        while self.is_running:
            try:
                if self.config["scaling"]["enabled"]:
                    self._evaluate_scaling_decision()
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaler: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def _evaluate_scaling_decision(self):
        """Evaluate whether to scale up or down"""
        # Check cooldown period
        if datetime.now() - self.last_scaling_action < self.scaling_cooldown:
            return
        
        metrics = self.resource_monitor.collect_metrics()
        trends = self.resource_monitor.get_resource_trends()
        queue_size = self.task_queue.size()
        
        # Scale up conditions
        should_scale_up = (
            (metrics["cpu_percent"] > self.config["scaling"]["scale_up_threshold"] or
             metrics["memory_percent"] > self.config["scaling"]["scale_up_threshold"] or
             queue_size > self.current_workers * 5) and
            self.current_workers < self.max_workers
        )
        
        # Scale down conditions
        should_scale_down = (
            metrics["cpu_percent"] < self.config["scaling"]["scale_down_threshold"] and
            metrics["memory_percent"] < self.config["scaling"]["scale_down_threshold"] and
            queue_size < self.current_workers and
            self.current_workers > self.min_workers
        )
        
        if should_scale_up:
            self._scale_up()
        elif should_scale_down:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up worker capacity"""
        new_worker_count = min(self.current_workers + 2, self.max_workers)
        
        if new_worker_count > self.current_workers:
            self.logger.info(f"Scaling up from {self.current_workers} to {new_worker_count} workers")
            
            # Resize thread pool
            self.thread_pool._max_workers = new_worker_count
            
            self.current_workers = new_worker_count
            self.last_scaling_action = datetime.now()
            self.metrics["scaling_events"] += 1
    
    def _scale_down(self):
        """Scale down worker capacity"""
        new_worker_count = max(self.current_workers - 1, self.min_workers)
        
        if new_worker_count < self.current_workers:
            self.logger.info(f"Scaling down from {self.current_workers} to {new_worker_count} workers")
            
            # Resize thread pool
            self.thread_pool._max_workers = new_worker_count
            
            self.current_workers = new_worker_count
            self.last_scaling_action = datetime.now()
            self.metrics["scaling_events"] += 1
    
    def _metrics_collector_thread(self):
        """Collect and update metrics"""
        while self.is_running:
            try:
                # Update queue metrics
                self.metrics["average_queue_size"] = (
                    (self.metrics["average_queue_size"] * 0.9) + 
                    (self.task_queue.size() * 0.1)
                )
                
                # Update worker utilization
                active_workers = sum(1 for stats in self.worker_stats.values() if stats.current_task)
                self.metrics["worker_utilization"] = active_workers / max(self.current_workers, 1)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
            
            time.sleep(10)  # Update every 10 seconds
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive executor metrics"""
        system_metrics = self.resource_monitor.collect_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "execution": {
                "tasks_submitted": self.metrics["tasks_submitted"],
                "tasks_completed": self.metrics["tasks_completed"],
                "tasks_failed": self.metrics["tasks_failed"],
                "success_rate": self.metrics["tasks_completed"] / max(self.metrics["tasks_submitted"], 1),
                "total_execution_time": self.metrics["total_execution_time"],
                "average_execution_time": self.metrics["total_execution_time"] / max(self.metrics["tasks_completed"], 1)
            },
            "queue": {
                "current_size": self.task_queue.size(),
                "average_size": self.metrics["average_queue_size"]
            },
            "workers": {
                "current_count": self.current_workers,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "utilization": self.metrics["worker_utilization"],
                "scaling_events": self.metrics["scaling_events"]
            },
            "resources": system_metrics,
            "registered_tasks": list(self.task_registry.keys())
        }
    
    def get_worker_stats(self) -> Dict[str, WorkerStats]:
        """Get detailed worker statistics"""
        return dict(self.worker_stats)


# Example task functions for demonstration
def cpu_intensive_task(n: int) -> int:
    """Example CPU-intensive task"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result


def io_intensive_task(file_path: str, content: str) -> str:
    """Example I/O-intensive task"""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    # Simulate some processing
    time.sleep(0.1)
    
    with open(temp_path, 'r') as f:
        result = f.read().upper()
    
    os.unlink(temp_path)
    return result


def network_task(url: str) -> Dict[str, Any]:
    """Example network task"""
    import urllib.request
    import json
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return {
                "status": response.getcode(),
                "content_length": len(response.read()),
                "headers": dict(response.headers)
            }
    except Exception as e:
        return {"error": str(e)}


async def main():
    """Demonstration of distributed task executor"""
    print("⚡ Distributed Task Executor Demo")
    print("=" * 50)
    
    # Create executor
    config = {
        "scaling": {
            "enabled": True,
            "min_workers": 2,
            "max_workers": 8,
            "target_cpu_usage": 60.0,
            "cooldown_seconds": 10
        }
    }
    
    executor = DistributedTaskExecutor(config)
    
    # Register task functions
    executor.register_task("cpu_task", cpu_intensive_task)
    executor.register_task("io_task", io_intensive_task)
    executor.register_task("network_task", network_task)
    
    # Start executor
    executor.start()
    
    try:
        print("🚀 Submitting tasks...")
        
        # Submit various types of tasks
        task_ids = []
        
        # CPU-intensive tasks
        for i in range(5):
            task = TaskDefinition(
                task_id=f"cpu_task_{i}",
                func_name="cpu_task",
                args=(100000 + i * 10000,),
                priority=TaskPriority.HIGH,
                tags={"cpu_intensive"}
            )
            task_id = executor.submit_task(task)
            task_ids.append(task_id)
        
        # I/O tasks with dependencies
        for i in range(3):
            task = TaskDefinition(
                task_id=f"io_task_{i}",
                func_name="io_task",
                args=(f"temp_file_{i}.txt", f"Content for file {i}\n" * 100),
                priority=TaskPriority.NORMAL,
                dependencies=[f"cpu_task_{i}"] if i < 5 else [],
                tags={"io_intensive"}
            )
            task_id = executor.submit_task(task)
            task_ids.append(task_id)
        
        print(f"📋 Submitted {len(task_ids)} tasks")
        
        # Wait for some tasks to complete
        completed_count = 0
        start_time = time.time()
        
        while completed_count < len(task_ids) and (time.time() - start_time) < 30:
            for task_id in task_ids:
                result = executor.get_result(task_id, timeout=0.1)
                if result and result.is_complete:
                    if task_id not in [r for r in task_ids if executor.task_results.get(r)]:
                        completed_count += 1
                        status_emoji = "✅" if result.status == TaskStatus.COMPLETED else "❌"
                        print(f"   {status_emoji} Task {task_id}: {result.status.value}")
            
            # Show progress
            if int(time.time()) % 5 == 0:
                metrics = executor.get_metrics()
                print(f"📊 Progress: {completed_count}/{len(task_ids)} tasks, "
                      f"Queue: {metrics['queue']['current_size']}, "
                      f"Workers: {metrics['workers']['current_count']}, "
                      f"CPU: {metrics['resources']['cpu_percent']:.1f}%")
            
            await asyncio.sleep(1)
        
        # Show final metrics
        print("\n📊 Final Metrics:")
        metrics = executor.get_metrics()
        
        print(f"   Tasks Completed: {metrics['execution']['tasks_completed']}")
        print(f"   Tasks Failed: {metrics['execution']['tasks_failed']}")
        print(f"   Success Rate: {metrics['execution']['success_rate']:.1%}")
        print(f"   Average Execution Time: {metrics['execution']['average_execution_time']:.2f}s")
        print(f"   Worker Utilization: {metrics['workers']['utilization']:.1%}")
        print(f"   Scaling Events: {metrics['workers']['scaling_events']}")
        print(f"   Current Workers: {metrics['workers']['current_count']}")
        
        print("\n🎯 Performance achieved:")
        throughput = metrics['execution']['tasks_completed'] / max((time.time() - start_time), 1)
        print(f"   Throughput: {throughput:.2f} tasks/second")
        print(f"   Resource Efficiency: {metrics['resources']['cpu_percent']:.1f}% CPU usage")
        
    finally:
        print("\n⏹️ Shutting down executor...")
        executor.stop(wait=True)
        print("✅ Shutdown complete")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    asyncio.run(main())