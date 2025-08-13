#!/usr/bin/env python3
"""
Auto-Scaling Manager v4.0
Intelligent workload distribution and resource scaling for ADO operations
"""

import os
import time
import psutil
import threading
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import json
import concurrent.futures

logger = logging.getLogger(__name__)


class ResourceState(Enum):
    """System resource states"""
    OPTIMAL = "optimal"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


@dataclass
class SystemMetrics:
    """Current system resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_read_mb_s: float
    disk_io_write_mb_s: float
    network_io_mb_s: float
    active_processes: int
    load_average: Tuple[float, float, float]
    available_memory_mb: float
    
    @property
    def resource_state(self) -> ResourceState:
        """Determine overall resource state"""
        if self.cpu_percent > 90 or self.memory_percent > 95:
            return ResourceState.CRITICAL
        elif self.cpu_percent > 75 or self.memory_percent > 85:
            return ResourceState.HIGH
        elif self.cpu_percent > 50 or self.memory_percent > 70:
            return ResourceState.MODERATE
        else:
            return ResourceState.OPTIMAL


@dataclass
class WorkloadProfile:
    """Profile for different types of workloads"""
    name: str
    cpu_weight: float
    memory_weight: float
    io_weight: float
    parallelizable: bool
    estimated_duration_seconds: float
    resource_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning"""
    direction: ScalingDirection
    target_workers: int
    reasoning: str
    confidence: float
    expected_improvement: float
    resource_impact: Dict[str, float] = field(default_factory=dict)


class WorkloadScheduler(ABC):
    """Abstract base for workload scheduling strategies"""
    
    @abstractmethod
    def schedule_tasks(self, tasks: List[Any], workers: int, system_metrics: SystemMetrics) -> List[List[Any]]:
        """Schedule tasks across available workers"""
        pass


class IntelligentRoundRobin(WorkloadScheduler):
    """Round-robin with load balancing intelligence"""
    
    def schedule_tasks(self, tasks: List[Any], workers: int, system_metrics: SystemMetrics) -> List[List[Any]]:
        """Distribute tasks considering system load"""
        if not tasks:
            return [[] for _ in range(workers)]
        
        # Adjust for system load
        effective_workers = workers
        if system_metrics.resource_state == ResourceState.HIGH:
            effective_workers = max(1, workers - 1)
        elif system_metrics.resource_state == ResourceState.CRITICAL:
            effective_workers = max(1, workers // 2)
        
        # Create worker queues
        worker_queues = [[] for _ in range(effective_workers)]
        
        # Distribute tasks
        for i, task in enumerate(tasks):
            worker_idx = i % effective_workers
            worker_queues[worker_idx].append(task)
        
        # Pad with empty queues for unused workers
        while len(worker_queues) < workers:
            worker_queues.append([])
        
        return worker_queues


class PriorityBasedScheduler(WorkloadScheduler):
    """Priority-based scheduling with resource awareness"""
    
    def schedule_tasks(self, tasks: List[Any], workers: int, system_metrics: SystemMetrics) -> List[List[Any]]:
        """Schedule tasks based on priority and resource requirements"""
        if not tasks:
            return [[] for _ in range(workers)]
        
        # Sort tasks by priority (assuming tasks have priority attribute)
        sorted_tasks = sorted(tasks, key=lambda x: getattr(x, 'priority', 0), reverse=True)
        
        # Create worker queues with capacity tracking
        worker_queues = [[] for _ in range(workers)]
        worker_loads = [0.0 for _ in range(workers)]
        
        # Distribute tasks to least loaded workers
        for task in sorted_tasks:
            # Find worker with minimum load
            min_load_idx = worker_loads.index(min(worker_loads))
            worker_queues[min_load_idx].append(task)
            
            # Update load estimation (simplified)
            task_weight = getattr(task, 'estimated_effort', 1.0)
            worker_loads[min_load_idx] += task_weight
        
        return worker_queues


class AutoScalingManager:
    """Intelligent auto-scaling manager for ADO workloads"""
    
    def __init__(self, 
                 min_workers: int = 1,
                 max_workers: int = None,
                 monitoring_interval: float = 30.0,
                 scale_cooldown: float = 300.0):
        
        self.min_workers = min_workers
        self.max_workers = max_workers or min(os.cpu_count(), 8)
        self.monitoring_interval = monitoring_interval
        self.scale_cooldown = scale_cooldown
        
        # Current state
        self.current_workers = min_workers
        self.last_scaling_time = datetime.min
        self.active_tasks: List[Any] = []
        self.completed_tasks: List[Tuple[Any, float]] = []  # Task and completion time
        
        # Metrics and history
        self.metrics_history: List[SystemMetrics] = []
        self.scaling_history: List[Tuple[datetime, ScalingDecision]] = []
        
        # Thread pool for parallel execution
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.current_workers)
        
        # Scheduling strategy
        self.scheduler = IntelligentRoundRobin()
        
        # Monitoring
        self._monitoring_thread = None
        self._shutdown = False
        
        # Performance tracking
        self.task_profiles: Dict[str, WorkloadProfile] = {}
        
        self.start_monitoring()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_mb = memory.available / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io and len(self.metrics_history) > 0:
                prev_disk_io = getattr(self.metrics_history[-1], '_disk_io_raw', disk_io)
                time_delta = 1.0  # Approximate time between measurements
                
                disk_read_mb_s = (disk_io.read_bytes - prev_disk_io.read_bytes) / (1024 * 1024 * time_delta)
                disk_write_mb_s = (disk_io.write_bytes - prev_disk_io.write_bytes) / (1024 * 1024 * time_delta)
            else:
                disk_read_mb_s = disk_write_mb_s = 0.0
            
            # Network I/O (simplified)
            network_io_mb_s = 0.0  # Would require more complex tracking
            
            # Process information
            active_processes = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                load_avg = (cpu_percent / 100, cpu_percent / 100, cpu_percent / 100)
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_io_read_mb_s=disk_read_mb_s,
                disk_io_write_mb_s=disk_write_mb_s,
                network_io_mb_s=network_io_mb_s,
                active_processes=active_processes,
                load_average=load_avg,
                available_memory_mb=available_memory_mb
            )
            
            # Store raw disk I/O for next calculation
            metrics._disk_io_raw = disk_io
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            # Return minimal metrics
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_io_read_mb_s=0.0,
                disk_io_write_mb_s=0.0,
                network_io_mb_s=0.0,
                active_processes=0,
                load_average=(0.0, 0.0, 0.0),
                available_memory_mb=1024.0
            )
    
    def make_scaling_decision(self, current_metrics: SystemMetrics) -> ScalingDecision:
        """Make intelligent scaling decision based on metrics and workload"""
        
        # Check cooldown period
        time_since_last_scale = datetime.now() - self.last_scaling_time
        if time_since_last_scale.total_seconds() < self.scale_cooldown:
            return ScalingDecision(
                direction=ScalingDirection.MAINTAIN,
                target_workers=self.current_workers,
                reasoning="Cooling down from last scaling decision",
                confidence=1.0,
                expected_improvement=0.0
            )
        
        # Analyze recent metrics trend
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        if not recent_metrics:
            return ScalingDecision(
                direction=ScalingDirection.MAINTAIN,
                target_workers=self.current_workers,
                reasoning="Insufficient metrics history",
                confidence=0.5,
                expected_improvement=0.0
            )
        
        # Calculate metrics trends
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        # Current load analysis
        current_load_score = (current_metrics.cpu_percent * 0.6 + 
                            current_metrics.memory_percent * 0.4)
        
        # Queue analysis
        pending_tasks = len(self.active_tasks)
        task_pressure = pending_tasks / max(self.current_workers, 1)
        
        # Decision logic
        reasoning = []
        confidence = 0.7
        
        # Scale up conditions
        if (current_load_score > 80 or 
            current_metrics.resource_state in [ResourceState.HIGH, ResourceState.CRITICAL] or
            task_pressure > 3):
            
            if self.current_workers < self.max_workers:
                new_workers = min(self.current_workers + 1, self.max_workers)
                
                reasoning.append(f"High system load ({current_load_score:.1f}%)")
                reasoning.append(f"Task pressure: {task_pressure:.1f}")
                
                expected_improvement = 0.3  # Expect 30% improvement
                
                return ScalingDecision(
                    direction=ScalingDirection.UP,
                    target_workers=new_workers,
                    reasoning="; ".join(reasoning),
                    confidence=confidence,
                    expected_improvement=expected_improvement,
                    resource_impact={"cpu": -20, "memory": -15}
                )
        
        # Scale down conditions
        elif (current_load_score < 30 and 
              avg_cpu < 40 and 
              task_pressure < 1 and 
              self.current_workers > self.min_workers):
            
            new_workers = max(self.current_workers - 1, self.min_workers)
            
            reasoning.append(f"Low system load ({current_load_score:.1f}%)")
            reasoning.append(f"Low task pressure: {task_pressure:.1f}")
            
            expected_improvement = 0.1  # Resource savings
            
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                target_workers=new_workers,
                reasoning="; ".join(reasoning),
                confidence=confidence,
                expected_improvement=expected_improvement,
                resource_impact={"cpu": 10, "memory": 15}
            )
        
        # Maintain current scale
        return ScalingDecision(
            direction=ScalingDirection.MAINTAIN,
            target_workers=self.current_workers,
            reasoning="System operating within optimal parameters",
            confidence=confidence,
            expected_improvement=0.0
        )
    
    def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply scaling decision and update resources"""
        if decision.direction == ScalingDirection.MAINTAIN:
            return True
        
        try:
            logger.info(f"Scaling {decision.direction.value}: "
                       f"{self.current_workers} -> {decision.target_workers} workers")
            logger.info(f"Reasoning: {decision.reasoning}")
            
            # Shutdown current executor
            self.executor.shutdown(wait=True, timeout=30)
            
            # Create new executor with target worker count
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=decision.target_workers
            )
            
            # Update state
            self.current_workers = decision.target_workers
            self.last_scaling_time = datetime.now()
            
            # Record scaling decision
            self.scaling_history.append((datetime.now(), decision))
            
            # Keep scaling history limited
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-100:]
            
            logger.info(f"Scaling completed successfully to {decision.target_workers} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply scaling decision: {e}")
            return False
    
    def execute_workload(self, tasks: List[Any], task_executor: Callable) -> List[Any]:
        """Execute workload with intelligent distribution and scaling"""
        if not tasks:
            return []
        
        start_time = time.time()
        
        # Collect current metrics
        current_metrics = self.collect_system_metrics()
        self.metrics_history.append(current_metrics)
        
        # Make scaling decision
        scaling_decision = self.make_scaling_decision(current_metrics)
        
        # Apply scaling if needed
        if scaling_decision.direction != ScalingDirection.MAINTAIN:
            self.apply_scaling_decision(scaling_decision)
        
        # Schedule tasks across workers
        task_batches = self.scheduler.schedule_tasks(tasks, self.current_workers, current_metrics)
        
        # Execute tasks in parallel
        self.active_tasks.extend(tasks)
        results = []
        
        try:
            # Submit batches to thread pool
            futures = []
            for batch in task_batches:
                if batch:  # Only submit non-empty batches
                    future = self.executor.submit(self._execute_batch, batch, task_executor)
                    futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures, timeout=3600):  # 1 hour timeout
                batch_results = future.result()
                results.extend(batch_results)
            
            # Update task tracking
            execution_time = time.time() - start_time
            for task in tasks:
                self.completed_tasks.append((task, execution_time / len(tasks)))
                if task in self.active_tasks:
                    self.active_tasks.remove(task)
            
            # Keep completed tasks history limited
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-1000:]
            
            logger.info(f"Workload completed: {len(tasks)} tasks in {execution_time:.2f}s "
                       f"using {self.current_workers} workers")
            
            return results
            
        except Exception as e:
            logger.error(f"Workload execution failed: {e}")
            # Clean up active tasks tracking
            for task in tasks:
                if task in self.active_tasks:
                    self.active_tasks.remove(task)
            return []
    
    def _execute_batch(self, batch: List[Any], task_executor: Callable) -> List[Any]:
        """Execute a batch of tasks sequentially"""
        results = []
        for task in batch:
            try:
                result = task_executor(task)
                results.append(result)
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                results.append(None)  # or some error indicator
        return results
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        def monitor_loop():
            while not self._shutdown:
                try:
                    metrics = self.collect_system_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep metrics history limited
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    
                    # Log system state periodically
                    if len(self.metrics_history) % 10 == 0:  # Every 10 intervals
                        logger.debug(f"System metrics - CPU: {metrics.cpu_percent:.1f}%, "
                                   f"Memory: {metrics.memory_percent:.1f}%, "
                                   f"State: {metrics.resource_state.value}")
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.warning(f"Monitoring error: {e}")
                    time.sleep(self.monitoring_interval)
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Auto-scaling monitoring started")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance and scaling summary"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-60:]  # Last 60 measurements
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        # Scaling effectiveness
        scaling_events = len(self.scaling_history)
        
        # Task completion stats
        if self.completed_tasks:
            avg_task_time = sum(time for _, time in self.completed_tasks) / len(self.completed_tasks)
            total_tasks = len(self.completed_tasks)
        else:
            avg_task_time = 0.0
            total_tasks = 0
        
        return {
            "current_workers": self.current_workers,
            "resource_utilization": {
                "cpu_avg": avg_cpu,
                "memory_avg": avg_memory,
                "current_state": recent_metrics[-1].resource_state.value if recent_metrics else "unknown"
            },
            "scaling_summary": {
                "total_scaling_events": scaling_events,
                "last_scaling": self.scaling_history[-1][0].isoformat() if self.scaling_history else None
            },
            "task_performance": {
                "total_completed": total_tasks,
                "avg_completion_time": avg_task_time,
                "active_tasks": len(self.active_tasks)
            },
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall system efficiency score"""
        if not self.metrics_history or not self.completed_tasks:
            return 0.5  # Neutral score
        
        recent_metrics = self.metrics_history[-30:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        # Efficiency factors
        resource_efficiency = 1.0 - ((avg_cpu + avg_memory) / 200)  # Lower usage = higher efficiency
        task_throughput = min(len(self.completed_tasks) / 100, 1.0)  # Normalize to 0-1
        scaling_stability = max(0, 1.0 - (len(self.scaling_history) / 50))  # Fewer scaling events = more stable
        
        efficiency_score = (resource_efficiency * 0.4 + 
                          task_throughput * 0.4 + 
                          scaling_stability * 0.2)
        
        return max(0.0, min(1.0, efficiency_score))
    
    def shutdown(self):
        """Shutdown auto-scaling manager gracefully"""
        logger.info("Shutting down auto-scaling manager...")
        
        self._shutdown = True
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        if self.executor:
            self.executor.shutdown(wait=True, timeout=30)
        
        logger.info("Auto-scaling manager shutdown complete")


# Global auto-scaling manager instance
auto_scaler = AutoScalingManager()


def scaled_execution(min_workers: int = 1, max_workers: int = None):
    """Decorator for auto-scaled function execution"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # For functions that process lists/iterables, enable auto-scaling
            if args and hasattr(args[0], '__iter__') and not isinstance(args[0], str):
                items = list(args[0])
                remaining_args = args[1:]
                
                def task_executor(item):
                    return func(item, *remaining_args, **kwargs)
                
                return auto_scaler.execute_workload(items, task_executor)
            else:
                # Regular function execution
                return func(*args, **kwargs)
        
        return wrapper
    return decorator