#!/usr/bin/env python3
"""
Advanced Auto-Scaling and Resource Optimization System
Intelligent resource management with predictive scaling and performance optimization
"""

import asyncio
import time
import json
import threading
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque, defaultdict
from enum import Enum
import statistics
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import gc
import os
import signal


logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingStrategy(Enum):
    """Scaling strategies"""
    REACTIVE = "reactive"          # React to current load
    PREDICTIVE = "predictive"      # Predict future load
    ADAPTIVE = "adaptive"          # Learn from patterns
    HYBRID = "hybrid"             # Combination of strategies


class ResourceType(Enum):
    """Types of resources to scale"""
    CPU_CORES = "cpu_cores"
    MEMORY_MB = "memory_mb"
    WORKERS = "workers"
    CONNECTIONS = "connections"
    THREADS = "threads"
    PROCESSES = "processes"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    load_average: float
    active_threads: int
    active_processes: int
    network_io_kb_per_sec: float
    disk_io_kb_per_sec: float
    response_time_ms: float
    request_rate: float
    error_rate: float
    queue_depth: int


@dataclass
class ScalingRule:
    """Auto-scaling rule definition"""
    name: str
    resource_type: ResourceType
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    min_value: int
    max_value: int
    cooldown_seconds: int = 300
    evaluation_periods: int = 3
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.8
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Record of a scaling event"""
    timestamp: datetime
    resource_type: ResourceType
    direction: ScalingDirection
    old_value: int
    new_value: int
    trigger_metric: str
    trigger_value: float
    reason: str
    success: bool


@dataclass
class WorkloadPrediction:
    """Workload prediction result"""
    timestamp: float
    predicted_cpu: float
    predicted_memory: float
    predicted_requests: float
    confidence_score: float
    time_horizon_minutes: int


class PerformanceOptimizer:
    """Performance optimization component"""
    
    def __init__(self):
        self.optimization_history: deque = deque(maxlen=100)
        self.baseline_metrics: Optional[ResourceMetrics] = None
        
    def optimize_gc_settings(self) -> Dict[str, Any]:
        """Optimize garbage collection settings"""
        try:
            # Get current GC stats
            gc_stats = gc.get_stats()
            
            # Calculate optimal thresholds based on memory usage
            memory_info = psutil.virtual_memory()
            available_mb = memory_info.available / (1024 * 1024)
            
            if available_mb > 2048:  # High memory system
                gc_thresholds = (1000, 15, 15)
            elif available_mb > 1024:  # Medium memory system
                gc_thresholds = (700, 10, 10)
            else:  # Low memory system
                gc_thresholds = (500, 8, 8)
            
            # Apply new thresholds
            old_thresholds = gc.get_threshold()
            gc.set_threshold(*gc_thresholds)
            
            optimization_result = {
                'action': 'gc_optimization',
                'old_thresholds': old_thresholds,
                'new_thresholds': gc_thresholds,
                'memory_available_mb': available_mb,
                'gc_collections': gc.get_count()
            }
            
            self.optimization_history.append(optimization_result)
            logger.info(f"GC optimization applied: {gc_thresholds}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"GC optimization failed: {e}")
            return {'action': 'gc_optimization', 'error': str(e)}
    
    def optimize_thread_pools(self, current_load: float) -> Dict[str, Any]:
        """Optimize thread pool sizes based on current load"""
        try:
            cpu_count = psutil.cpu_count()
            
            # Calculate optimal thread pool sizes
            if current_load > 0.8:  # High load
                io_threads = min(cpu_count * 4, 64)
                cpu_threads = cpu_count
            elif current_load > 0.5:  # Medium load
                io_threads = min(cpu_count * 2, 32)
                cpu_threads = cpu_count
            else:  # Low load
                io_threads = min(cpu_count, 16)
                cpu_threads = max(cpu_count // 2, 2)
            
            return {
                'action': 'thread_pool_optimization',
                'recommended_io_threads': io_threads,
                'recommended_cpu_threads': cpu_threads,
                'current_load': current_load
            }
            
        except Exception as e:
            logger.error(f"Thread pool optimization failed: {e}")
            return {'action': 'thread_pool_optimization', 'error': str(e)}
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        try:
            # Force garbage collection
            collected_objects = gc.collect()
            
            # Get memory stats after GC
            memory_info = psutil.virtual_memory()
            process_info = psutil.Process()
            
            return {
                'action': 'memory_optimization',
                'objects_collected': collected_objects,
                'memory_percent': memory_info.percent,
                'process_memory_mb': process_info.memory_info().rss / (1024 * 1024),
                'available_memory_mb': memory_info.available / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {'action': 'memory_optimization', 'error': str(e)}


class WorkloadPredictor:
    """Predictive workload analysis"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.pattern_cache: Dict[str, Any] = {}
    
    def add_metrics(self, metrics: ResourceMetrics):
        """Add metrics to history"""
        self.metrics_history.append(metrics)
    
    def predict_workload(self, minutes_ahead: int = 15) -> WorkloadPrediction:
        """Predict workload for the next N minutes"""
        try:
            if len(self.metrics_history) < 10:
                # Not enough data for prediction
                current = self.metrics_history[-1] if self.metrics_history else None
                return WorkloadPrediction(
                    timestamp=time.time(),
                    predicted_cpu=current.cpu_percent if current else 50.0,
                    predicted_memory=current.memory_percent if current else 50.0,
                    predicted_requests=current.request_rate if current else 10.0,
                    confidence_score=0.1,
                    time_horizon_minutes=minutes_ahead
                )
            
            # Get recent metrics for trend analysis
            recent_metrics = list(self.metrics_history)[-60:]  # Last 60 data points
            
            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
            request_trend = self._calculate_trend([m.request_rate for m in recent_metrics])
            
            # Get current values
            current = recent_metrics[-1]
            
            # Predict future values
            time_factor = minutes_ahead / 60.0  # Convert to hours for trend application
            
            predicted_cpu = max(0, min(100, current.cpu_percent + cpu_trend * time_factor))
            predicted_memory = max(0, min(100, current.memory_percent + memory_trend * time_factor))
            predicted_requests = max(0, current.request_rate + request_trend * time_factor)
            
            # Calculate confidence based on trend stability
            cpu_variance = statistics.variance([m.cpu_percent for m in recent_metrics[-20:]])
            confidence = max(0.1, min(0.9, 1.0 - (cpu_variance / 1000.0)))
            
            return WorkloadPrediction(
                timestamp=time.time(),
                predicted_cpu=predicted_cpu,
                predicted_memory=predicted_memory,
                predicted_requests=predicted_requests,
                confidence_score=confidence,
                time_horizon_minutes=minutes_ahead
            )
            
        except Exception as e:
            logger.error(f"Workload prediction failed: {e}")
            # Return safe default prediction
            return WorkloadPrediction(
                timestamp=time.time(),
                predicted_cpu=50.0,
                predicted_memory=50.0,
                predicted_requests=10.0,
                confidence_score=0.1,
                time_horizon_minutes=minutes_ahead
            )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0
        
        try:
            n = len(values)
            x_values = list(range(n))
            
            # Linear regression calculation
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
            
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect patterns in workload data"""
        try:
            if len(self.metrics_history) < 50:
                return {'patterns': [], 'confidence': 0.0}
            
            # Analyze cyclical patterns (daily, hourly)
            patterns = []
            
            # Check for hourly patterns
            hourly_pattern = self._detect_cyclical_pattern(60)  # 60 data points = 1 hour if 1 min intervals
            if hourly_pattern:
                patterns.append(hourly_pattern)
            
            # Check for daily patterns
            daily_pattern = self._detect_cyclical_pattern(1440)  # 1440 data points = 1 day
            if daily_pattern:
                patterns.append(daily_pattern)
            
            return {
                'patterns': patterns,
                'confidence': len(patterns) * 0.3,
                'history_size': len(self.metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {'patterns': [], 'confidence': 0.0, 'error': str(e)}
    
    def _detect_cyclical_pattern(self, cycle_length: int) -> Optional[Dict[str, Any]]:
        """Detect cyclical patterns in the data"""
        try:
            if len(self.metrics_history) < cycle_length * 2:
                return None
            
            # Extract CPU usage for pattern analysis
            cpu_values = [m.cpu_percent for m in self.metrics_history]
            
            # Compare current cycle with previous cycles
            current_cycle = cpu_values[-cycle_length:]
            previous_cycles = [
                cpu_values[-cycle_length*2:-cycle_length],
                cpu_values[-cycle_length*3:-cycle_length*2] if len(cpu_values) >= cycle_length*3 else None
            ]
            previous_cycles = [cycle for cycle in previous_cycles if cycle]
            
            if not previous_cycles:
                return None
            
            # Calculate correlation between cycles
            correlations = []
            for prev_cycle in previous_cycles:
                if len(prev_cycle) == len(current_cycle):
                    correlation = self._calculate_correlation(current_cycle, prev_cycle)
                    correlations.append(correlation)
            
            avg_correlation = statistics.mean(correlations) if correlations else 0.0
            
            if avg_correlation > 0.7:  # Strong correlation threshold
                return {
                    'type': 'cyclical',
                    'cycle_length': cycle_length,
                    'correlation': avg_correlation,
                    'strength': 'strong' if avg_correlation > 0.8 else 'moderate'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Cyclical pattern detection failed: {e}")
            return None
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two series"""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
            sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
            
            denominator = math.sqrt(sum_sq_x * sum_sq_y)
            
            return numerator / denominator if denominator != 0 else 0.0
            
        except (ValueError, ZeroDivisionError):
            return 0.0


class AutoScalingEngine:
    """Advanced auto-scaling engine"""
    
    def __init__(self, 
                 strategy: ScalingStrategy = ScalingStrategy.HYBRID,
                 metrics_window_minutes: int = 10,
                 enable_predictions: bool = True):
        
        self.strategy = strategy
        self.metrics_window_minutes = metrics_window_minutes
        self.enable_predictions = enable_predictions
        
        # Components
        self.optimizer = PerformanceOptimizer()
        self.predictor = WorkloadPredictor()
        
        # State
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=int(metrics_window_minutes * 60))  # 1 per second
        self.last_scaling_times: Dict[str, float] = {}
        
        # Resource limits
        self.current_resources: Dict[ResourceType, int] = {
            ResourceType.CPU_CORES: psutil.cpu_count(),
            ResourceType.MEMORY_MB: int(psutil.virtual_memory().total / (1024 * 1024)),
            ResourceType.WORKERS: mp.cpu_count(),
            ResourceType.CONNECTIONS: 1000,
            ResourceType.THREADS: 100,
            ResourceType.PROCESSES: 10
        }
        
        # Thread pool executors for scaling operations
        self.scaling_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="autoscaler")
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_scaling_events': 0,
            'successful_scaling_events': 0,
            'scaling_by_direction': defaultdict(int),
            'scaling_by_resource': defaultdict(int),
            'optimization_events': 0,
            'prediction_accuracy': deque(maxlen=100)
        }
        
        self._setup_default_rules()
        logger.info(f"Auto-scaling engine initialized with {strategy.value} strategy")
    
    def _setup_default_rules(self):
        """Setup default scaling rules"""
        
        # CPU scaling rule
        self.add_scaling_rule(ScalingRule(
            name="cpu_scaling",
            resource_type=ResourceType.WORKERS,
            metric_name="cpu_percent",
            scale_up_threshold=75.0,
            scale_down_threshold=25.0,
            min_value=2,
            max_value=mp.cpu_count() * 2,
            cooldown_seconds=300,
            evaluation_periods=3
        ))
        
        # Memory scaling rule
        self.add_scaling_rule(ScalingRule(
            name="memory_scaling",
            resource_type=ResourceType.WORKERS,
            metric_name="memory_percent",
            scale_up_threshold=80.0,
            scale_down_threshold=40.0,
            min_value=2,
            max_value=mp.cpu_count() * 2,
            cooldown_seconds=180,
            evaluation_periods=2
        ))
        
        # Request rate scaling
        self.add_scaling_rule(ScalingRule(
            name="request_rate_scaling",
            resource_type=ResourceType.THREADS,
            metric_name="request_rate",
            scale_up_threshold=100.0,
            scale_down_threshold=20.0,
            min_value=10,
            max_value=200,
            cooldown_seconds=120,
            evaluation_periods=2
        ))
        
        # Response time scaling
        self.add_scaling_rule(ScalingRule(
            name="response_time_scaling",
            resource_type=ResourceType.CONNECTIONS,
            metric_name="response_time_ms",
            scale_up_threshold=1000.0,
            scale_down_threshold=100.0,
            min_value=100,
            max_value=2000,
            cooldown_seconds=60,
            evaluation_periods=3
        ))
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule"""
        self.scaling_rules[rule.name] = rule
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule"""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            logger.info(f"Removed scaling rule: {rule_name}")
    
    def start_monitoring(self):
        """Start auto-scaling monitoring"""
        if self.monitoring_active:
            logger.warning("Auto-scaling monitoring is already active")
            return
        
        self.monitoring_active = True
        self.shutdown_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="autoscaling_monitor"
        )
        self.monitoring_thread.start()
        
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self.predictor.add_metrics(metrics)
                
                # Evaluate scaling rules
                scaling_decisions = self._evaluate_scaling_rules(metrics)
                
                # Apply scaling decisions
                for decision in scaling_decisions:
                    self._apply_scaling_decision(decision)
                
                # Perform optimizations
                if len(self.metrics_history) % 60 == 0:  # Every 60 iterations
                    self._perform_optimizations(metrics)
                
                # Predictive scaling
                if self.enable_predictions and len(self.metrics_history) % 120 == 0:  # Every 2 minutes
                    self._perform_predictive_scaling()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Auto-scaling monitoring error: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else cpu_percent / 100
            except (AttributeError, OSError):
                load_avg = cpu_percent / 100
            
            # Process metrics
            try:
                process_count = len(psutil.pids())
                thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                process_count = 0
                thread_count = 0
            
            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                # Calculate rate (simplified - would need previous values for accurate rate)
                network_io_rate = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
            except:
                network_io_rate = 0.0
            
            # Disk I/O
            try:
                disk_io = psutil.disk_io_counters()
                disk_io_rate = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024) if disk_io else 0.0
            except:
                disk_io_rate = 0.0
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                load_average=load_avg,
                active_threads=thread_count,
                active_processes=process_count,
                network_io_kb_per_sec=network_io_rate,
                disk_io_kb_per_sec=disk_io_rate,
                response_time_ms=0.0,  # Would be provided by application
                request_rate=0.0,      # Would be provided by application
                error_rate=0.0,        # Would be provided by application
                queue_depth=0          # Would be provided by application
            )
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            # Return safe default metrics
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=1024.0,
                load_average=0.0,
                active_threads=0,
                active_processes=0,
                network_io_kb_per_sec=0.0,
                disk_io_kb_per_sec=0.0,
                response_time_ms=0.0,
                request_rate=0.0,
                error_rate=0.0,
                queue_depth=0
            )
    
    def _evaluate_scaling_rules(self, current_metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """Evaluate all scaling rules and return scaling decisions"""
        decisions = []
        
        for rule_name, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            try:
                decision = self._evaluate_single_rule(rule, current_metrics)
                if decision:
                    decisions.append(decision)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
        
        return decisions
    
    def _evaluate_single_rule(self, rule: ScalingRule, metrics: ResourceMetrics) -> Optional[Dict[str, Any]]:
        """Evaluate a single scaling rule"""
        
        # Check cooldown period
        last_scaling = self.last_scaling_times.get(rule.name, 0)
        if time.time() - last_scaling < rule.cooldown_seconds:
            return None
        
        # Get metric value
        metric_value = getattr(metrics, rule.metric_name, None)
        if metric_value is None:
            logger.warning(f"Metric {rule.metric_name} not found in metrics")
            return None
        
        # Get historical values for evaluation periods
        historical_values = []
        for hist_metrics in list(self.metrics_history)[-rule.evaluation_periods:]:
            hist_value = getattr(hist_metrics, rule.metric_name, None)
            if hist_value is not None:
                historical_values.append(hist_value)
        
        if len(historical_values) < rule.evaluation_periods:
            return None  # Not enough history
        
        # Check if threshold is consistently breached
        avg_value = statistics.mean(historical_values)
        current_resource_value = self.current_resources.get(rule.resource_type, 0)
        
        # Scale up decision
        if avg_value >= rule.scale_up_threshold and current_resource_value < rule.max_value:
            new_value = min(rule.max_value, int(current_resource_value * rule.scale_up_factor))
            if new_value > current_resource_value:
                return {
                    'rule_name': rule.name,
                    'resource_type': rule.resource_type,
                    'direction': ScalingDirection.UP,
                    'current_value': current_resource_value,
                    'new_value': new_value,
                    'trigger_metric': rule.metric_name,
                    'trigger_value': avg_value,
                    'threshold': rule.scale_up_threshold,
                    'reason': f"{rule.metric_name} ({avg_value:.2f}) exceeded scale-up threshold ({rule.scale_up_threshold})"
                }
        
        # Scale down decision
        elif avg_value <= rule.scale_down_threshold and current_resource_value > rule.min_value:
            new_value = max(rule.min_value, int(current_resource_value * rule.scale_down_factor))
            if new_value < current_resource_value:
                return {
                    'rule_name': rule.name,
                    'resource_type': rule.resource_type,
                    'direction': ScalingDirection.DOWN,
                    'current_value': current_resource_value,
                    'new_value': new_value,
                    'trigger_metric': rule.metric_name,
                    'trigger_value': avg_value,
                    'threshold': rule.scale_down_threshold,
                    'reason': f"{rule.metric_name} ({avg_value:.2f}) below scale-down threshold ({rule.scale_down_threshold})"
                }
        
        return None
    
    def _apply_scaling_decision(self, decision: Dict[str, Any]):
        """Apply a scaling decision"""
        try:
            rule_name = decision['rule_name']
            resource_type = decision['resource_type']
            direction = decision['direction']
            new_value = decision['new_value']
            
            logger.info(f"Applying scaling decision: {rule_name} - {direction.value} to {new_value}")
            
            # Apply the scaling
            success = self._scale_resource(resource_type, new_value)
            
            # Record scaling event
            event = ScalingEvent(
                timestamp=datetime.now(),
                resource_type=resource_type,
                direction=direction,
                old_value=decision['current_value'],
                new_value=new_value,
                trigger_metric=decision['trigger_metric'],
                trigger_value=decision['trigger_value'],
                reason=decision['reason'],
                success=success
            )
            
            self.scaling_history.append(event)
            
            # Update statistics
            self.stats['total_scaling_events'] += 1
            if success:
                self.stats['successful_scaling_events'] += 1
                self.current_resources[resource_type] = new_value
            
            self.stats['scaling_by_direction'][direction.value] += 1
            self.stats['scaling_by_resource'][resource_type.value] += 1
            
            # Update last scaling time
            self.last_scaling_times[rule_name] = time.time()
            
            if success:
                logger.info(f"Successfully scaled {resource_type.value} to {new_value}")
            else:
                logger.error(f"Failed to scale {resource_type.value} to {new_value}")
            
        except Exception as e:
            logger.error(f"Error applying scaling decision: {e}")
    
    def _scale_resource(self, resource_type: ResourceType, new_value: int) -> bool:
        """Scale a specific resource type"""
        try:
            if resource_type == ResourceType.WORKERS:
                return self._scale_workers(new_value)
            elif resource_type == ResourceType.THREADS:
                return self._scale_threads(new_value)
            elif resource_type == ResourceType.PROCESSES:
                return self._scale_processes(new_value)
            elif resource_type == ResourceType.CONNECTIONS:
                return self._scale_connections(new_value)
            else:
                logger.warning(f"Scaling not implemented for resource type: {resource_type}")
                return False
                
        except Exception as e:
            logger.error(f"Resource scaling failed for {resource_type}: {e}")
            return False
    
    def _scale_workers(self, new_value: int) -> bool:
        """Scale worker processes/threads"""
        # This would integrate with your application's worker management
        logger.info(f"Scaling workers to {new_value}")
        # Implementation would depend on your specific worker architecture
        return True
    
    def _scale_threads(self, new_value: int) -> bool:
        """Scale thread pool size"""
        logger.info(f"Scaling thread pool to {new_value}")
        # Implementation would update thread pool sizes
        return True
    
    def _scale_processes(self, new_value: int) -> bool:
        """Scale process count"""
        logger.info(f"Scaling processes to {new_value}")
        # Implementation would manage process spawning/termination
        return True
    
    def _scale_connections(self, new_value: int) -> bool:
        """Scale connection pool size"""
        logger.info(f"Scaling connection pool to {new_value}")
        # Implementation would adjust connection limits
        return True
    
    def _perform_optimizations(self, metrics: ResourceMetrics):
        """Perform performance optimizations"""
        try:
            optimizations_applied = []
            
            # GC optimization
            if metrics.memory_percent > 70:
                gc_result = self.optimizer.optimize_gc_settings()
                optimizations_applied.append(gc_result)
            
            # Thread pool optimization
            current_load = metrics.cpu_percent / 100.0
            thread_result = self.optimizer.optimize_thread_pools(current_load)
            optimizations_applied.append(thread_result)
            
            # Memory optimization
            if metrics.memory_percent > 80:
                mem_result = self.optimizer.optimize_memory_usage()
                optimizations_applied.append(mem_result)
            
            self.stats['optimization_events'] += len(optimizations_applied)
            logger.info(f"Applied {len(optimizations_applied)} optimizations")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
    
    def _perform_predictive_scaling(self):
        """Perform predictive scaling based on workload forecasts"""
        try:
            if not self.enable_predictions:
                return
            
            # Get prediction for next 15 minutes
            prediction = self.predictor.predict_workload(15)
            
            if prediction.confidence_score < 0.5:
                return  # Not confident enough in prediction
            
            # Check if predicted load warrants proactive scaling
            current_metrics = self.metrics_history[-1] if self.metrics_history else None
            if not current_metrics:
                return
            
            # Predictive CPU scaling
            if prediction.predicted_cpu > 80 and current_metrics.cpu_percent < 60:
                logger.info(f"Predictive scaling: CPU expected to reach {prediction.predicted_cpu:.1f}%")
                # Proactively scale up workers
                current_workers = self.current_resources.get(ResourceType.WORKERS, 0)
                new_workers = min(current_workers + 2, self.scaling_rules.get('cpu_scaling', ScalingRule('', ResourceType.WORKERS, '', 0, 0, 0, 16)).max_value)
                if self._scale_resource(ResourceType.WORKERS, new_workers):
                    self.current_resources[ResourceType.WORKERS] = new_workers
            
            # Predictive memory scaling
            if prediction.predicted_memory > 85 and current_metrics.memory_percent < 70:
                logger.info(f"Predictive scaling: Memory expected to reach {prediction.predicted_memory:.1f}%")
                # Trigger memory optimization
                self.optimizer.optimize_memory_usage()
            
        except Exception as e:
            logger.error(f"Predictive scaling failed: {e}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics"""
        try:
            recent_events = [
                asdict(event) for event in list(self.scaling_history)[-10:]
            ]
            
            success_rate = (
                self.stats['successful_scaling_events'] / 
                max(self.stats['total_scaling_events'], 1)
            )
            
            # Get current resource utilization
            current_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            return {
                'total_scaling_events': self.stats['total_scaling_events'],
                'successful_scaling_events': self.stats['successful_scaling_events'],
                'success_rate': success_rate,
                'scaling_by_direction': dict(self.stats['scaling_by_direction']),
                'scaling_by_resource': dict(self.stats['scaling_by_resource']),
                'optimization_events': self.stats['optimization_events'],
                'current_resources': {k.value: v for k, v in self.current_resources.items()},
                'recent_events': recent_events,
                'current_metrics': asdict(current_metrics) if current_metrics else None,
                'active_rules': len([r for r in self.scaling_rules.values() if r.enabled]),
                'monitoring_active': self.monitoring_active
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling stats: {e}")
            return {'error': str(e)}


# Global auto-scaling instance
auto_scaler = AutoScalingEngine()


def enable_auto_scaling(**kwargs):
    """Enable auto-scaling with configuration"""
    global auto_scaler
    
    # Update configuration if provided
    for key, value in kwargs.items():
        if hasattr(auto_scaler, key):
            setattr(auto_scaler, key, value)
    
    auto_scaler.start_monitoring()
    logger.info("Auto-scaling enabled")


def disable_auto_scaling():
    """Disable auto-scaling"""
    auto_scaler.stop_monitoring()
    logger.info("Auto-scaling disabled")


if __name__ == "__main__":
    import asyncio
    
    async def test_auto_scaling():
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create auto-scaler
        scaler = AutoScalingEngine(strategy=ScalingStrategy.HYBRID)
        
        # Start monitoring
        scaler.start_monitoring()
        
        try:
            # Let it run for a bit
            await asyncio.sleep(30)
            
            # Get stats
            stats = scaler.get_scaling_stats()
            print(json.dumps(stats, indent=2, default=str))
            
            # Test workload prediction
            prediction = scaler.predictor.predict_workload(15)
            print(f"Workload prediction: {asdict(prediction)}")
            
        finally:
            scaler.stop_monitoring()
    
    # Run test
    asyncio.run(test_auto_scaling())