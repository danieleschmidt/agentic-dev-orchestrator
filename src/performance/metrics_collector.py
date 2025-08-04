#!/usr/bin/env python3
"""
Advanced metrics collection and performance analysis
"""

import time
import json
import statistics
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import datetime
import logging
from collections import defaultdict, deque


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MetricsCollector:
    """Advanced metrics collection with time-series analysis"""
    
    def __init__(self, retention_period: int = 86400):  # 24 hours default
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_period = retention_period
        self.logger = logging.getLogger("metrics_collector")
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: float, unit: str = "", tags: Dict = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        
        # Cleanup old metrics
        self._cleanup_old_metrics(name)
    
    def record_execution_time(self, operation: str, execution_time: float, tags: Dict = None):
        """Record execution time metric"""
        self.record_metric(f"{operation}_execution_time", execution_time, "seconds", tags)
    
    def record_throughput(self, operation: str, count: int, duration: float, tags: Dict = None):
        """Record throughput metric"""
        throughput = count / max(duration, 0.001)  # Avoid division by zero
        self.record_metric(f"{operation}_throughput", throughput, "ops/sec", tags)
    
    def record_error_rate(self, operation: str, errors: int, total: int, tags: Dict = None):
        """Record error rate metric"""
        error_rate = errors / max(total, 1)
        self.record_metric(f"{operation}_error_rate", error_rate, "ratio", tags)
    
    def record_resource_usage(self, resource: str, usage: float, unit: str = "%", tags: Dict = None):
        """Record resource usage metric"""
        self.record_metric(f"resource_{resource}_usage", usage, unit, tags)
    
    def get_metric_stats(self, name: str, time_window: Optional[float] = None) -> Dict:
        """Get statistical analysis of a metric"""
        if name not in self.metrics:
            return {}
        
        metrics = self.metrics[name]
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = time.time() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
            "latest": values[-1],
            "time_range": {
                "start": metrics[0].timestamp,
                "end": metrics[-1].timestamp,
                "duration": metrics[-1].timestamp - metrics[0].timestamp
            }
        }
    
    def get_trend_analysis(self, name: str, time_window: float = 3600) -> Dict:
        """Analyze trends in metric data"""
        if name not in self.metrics:
            return {}
        
        cutoff_time = time.time() - time_window
        recent_metrics = [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
        
        if len(recent_metrics) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend using linear regression
        timestamps = [m.timestamp for m in recent_metrics]
        values = [m.value for m in recent_metrics]
        
        # Normalize timestamps to start from 0
        min_timestamp = min(timestamps)
        x_values = [t - min_timestamp for t in timestamps]
        
        # Simple linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x_squared = sum(x * x for x in x_values)
        
        # Calculate slope (trend)
        denominator = n * sum_x_squared - sum_x * sum_x
        if denominator == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "slope": slope,
            "confidence": abs(slope) * 100,  # Simple confidence measure
            "data_points": len(recent_metrics),
            "time_window_hours": time_window / 3600
        }
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        summary = {
            "collection_period": time.time() - self.start_time,
            "metrics_count": len(self.metrics),
            "total_data_points": sum(len(metrics) for metrics in self.metrics.values()),
            "metrics": {}
        }
        
        for name, metrics in self.metrics.items():
            if metrics:
                summary["metrics"][name] = {
                    "stats": self.get_metric_stats(name),
                    "trend": self.get_trend_analysis(name),
                    "latest_value": metrics[-1].value,
                    "latest_timestamp": metrics[-1].timestamp
                }
        
        return summary
    
    def export_metrics(self, output_file: Path, format: str = "json") -> bool:
        """Export metrics to file"""
        try:
            if format == "json":
                data = {
                    "export_timestamp": time.time(),
                    "collection_period": time.time() - self.start_time,
                    "metrics": {}
                }
                
                for name, metrics in self.metrics.items():
                    data["metrics"][name] = [asdict(m) for m in metrics]
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            elif format == "csv":
                import csv
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["metric_name", "value", "unit", "timestamp", "tags"])
                    
                    for name, metrics in self.metrics.items():
                        for metric in metrics:
                            writer.writerow([
                                metric.name,
                                metric.value,
                                metric.unit,
                                metric.timestamp,
                                json.dumps(metric.tags)
                            ])
            
            self.logger.info(f"Metrics exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _cleanup_old_metrics(self, name: str):
        """Remove old metrics beyond retention period"""
        cutoff_time = time.time() - self.retention_period
        metrics = self.metrics[name]
        
        # Remove old metrics from the left
        while metrics and metrics[0].timestamp < cutoff_time:
            metrics.popleft()
    
    def create_dashboard_data(self) -> Dict:
        """Create data for dashboard visualization"""
        dashboard = {
            "timestamp": time.time(),
            "overview": {
                "uptime": time.time() - self.start_time,
                "metrics_tracked": len(self.metrics),
                "total_data_points": sum(len(m) for m in self.metrics.values())
            },
            "key_metrics": {},
            "alerts": []
        }
        
        # Key performance indicators
        for name in ["task_execution_time", "task_throughput", "task_error_rate"]:
            if name in self.metrics:
                stats = self.get_metric_stats(name, time_window=3600)  # Last hour
                trend = self.get_trend_analysis(name, time_window=3600)
                
                dashboard["key_metrics"][name] = {
                    "current": stats.get("latest", 0),
                    "avg_1h": stats.get("mean", 0),
                    "p95_1h": stats.get("p95", 0),
                    "trend": trend.get("trend", "unknown")
                }
                
                # Generate alerts
                if name == "task_error_rate" and stats.get("latest", 0) > 0.1:
                    dashboard["alerts"].append({
                        "level": "warning",
                        "message": f"High error rate: {stats['latest']:.1%}"
                    })
                
                if trend.get("trend") == "decreasing" and "throughput" in name:
                    dashboard["alerts"].append({
                        "level": "info",
                        "message": f"Decreasing throughput trend detected for {name}"
                    })
        
        return dashboard


class PerformanceProfiler:
    """Context manager for profiling code execution"""
    
    def __init__(self, operation_name: str, metrics_collector: MetricsCollector, 
                 tags: Dict = None):
        self.operation_name = operation_name
        self.metrics_collector = metrics_collector
        self.tags = tags or {}
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        # Record execution time
        self.metrics_collector.record_execution_time(
            self.operation_name, 
            execution_time, 
            self.tags
        )
        
        # Record success/failure
        if exc_type is None:
            self.metrics_collector.record_metric(
                f"{self.operation_name}_success", 1, "count", self.tags
            )
        else:
            self.metrics_collector.record_metric(
                f"{self.operation_name}_failure", 1, "count", self.tags
            )


def main():
    """Test metrics collection"""
    collector = MetricsCollector()
    
    # Simulate some metrics
    import random
    for i in range(100):
        collector.record_execution_time("test_operation", random.uniform(0.1, 2.0))
        collector.record_throughput("test_operation", random.randint(10, 100), 1.0)
        time.sleep(0.01)
    
    # Get performance summary
    summary = collector.get_performance_summary()
    print("Performance Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Test profiler
    with PerformanceProfiler("sample_task", collector) as profiler:
        time.sleep(0.1)  # Simulate work
        
    print("\nDashboard Data:")
    dashboard = collector.create_dashboard_data()
    print(json.dumps(dashboard, indent=2, default=str))


if __name__ == "__main__":
    main()