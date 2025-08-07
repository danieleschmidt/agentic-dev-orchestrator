"""
Performance optimization and monitoring for sentiment analysis
"""
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for sentiment analysis operations"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_usage_mb: float
    cpu_percent: float
    texts_processed: int
    cache_hit_rate: float = 0.0
    errors_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'operation': self.operation_name,
            'duration_ms': round(self.duration * 1000, 2),
            'memory_mb': round(self.memory_usage_mb, 2),
            'cpu_percent': round(self.cpu_percent, 2),
            'texts_processed': self.texts_processed,
            'throughput_per_sec': round(self.texts_processed / self.duration, 2) if self.duration > 0 else 0,
            'cache_hit_rate': round(self.cache_hit_rate, 2),
            'errors_count': self.errors_count,
            'timestamp': self.start_time
        }


class PerformanceMonitor:
    """Monitor and optimize sentiment analysis performance"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics_history: List[PerformanceMetrics] = []
        if HAS_PSUTIL:
            self.process = psutil.Process()
        else:
            self.process = None
            logger.warning("psutil not available - memory/CPU monitoring disabled")
        logger.info("PerformanceMonitor initialized")
    
    @contextmanager
    def monitor_operation(self, operation_name: str, texts_count: int = 1):
        """
        Context manager to monitor operation performance
        
        Args:
            operation_name: Name of the operation being monitored
            texts_count: Number of texts being processed
            
        Yields:
            Dictionary to collect additional metrics
        """
        # Collect start metrics
        start_time = time.time()
        if self.process:
            start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = self.process.cpu_percent()
        else:
            start_memory = 0.0
            start_cpu = 0.0
        
        additional_metrics = {}
        
        try:
            yield additional_metrics
            
        finally:
            # Collect end metrics
            end_time = time.time()
            if self.process:
                end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                end_cpu = self.process.cpu_percent()
            else:
                end_memory = 0.0
                end_cpu = 0.0
            
            duration = end_time - start_time
            memory_usage = max(end_memory, start_memory)  # Peak usage
            cpu_percent = (start_cpu + end_cpu) / 2  # Average
            
            # Create metrics object
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                texts_processed=texts_count,
                cache_hit_rate=additional_metrics.get('cache_hit_rate', 0.0),
                errors_count=additional_metrics.get('errors_count', 0)
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Log performance
            self._log_performance(metrics)
    
    def _log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        throughput = metrics.texts_processed / metrics.duration if metrics.duration > 0 else 0
        
        logger.info(
            f"Performance [{metrics.operation_name}]: "
            f"{metrics.duration:.3f}s, "
            f"{throughput:.1f} texts/sec, "
            f"{metrics.memory_usage_mb:.1f}MB, "
            f"{metrics.cpu_percent:.1f}% CPU"
        )
        
        # Warn about performance issues
        if metrics.duration > 5.0:  # Slow operation
            logger.warning(f"Slow operation detected: {metrics.operation_name} took {metrics.duration:.3f}s")
        
        if metrics.memory_usage_mb > 500:  # High memory usage
            logger.warning(f"High memory usage: {metrics.memory_usage_mb:.1f}MB for {metrics.operation_name}")
        
        if throughput < 10 and metrics.texts_processed > 10:  # Low throughput
            logger.warning(f"Low throughput: {throughput:.1f} texts/sec for {metrics.operation_name}")
    
    def get_performance_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics
        
        Args:
            last_n: Only consider last N operations (None for all)
            
        Returns:
            Performance summary dictionary
        """
        if not self.metrics_history:
            return {'message': 'No performance data available'}
        
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        # Calculate aggregate statistics
        total_duration = sum(m.duration for m in metrics)
        total_texts = sum(m.texts_processed for m in metrics)
        avg_memory = sum(m.memory_usage_mb for m in metrics) / len(metrics)
        avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in metrics) / len(metrics)
        total_errors = sum(m.errors_count for m in metrics)
        
        # Calculate throughput
        overall_throughput = total_texts / total_duration if total_duration > 0 else 0
        
        # Find performance extremes
        fastest_op = min(metrics, key=lambda m: m.duration / m.texts_processed if m.texts_processed > 0 else float('inf'))
        slowest_op = max(metrics, key=lambda m: m.duration / m.texts_processed if m.texts_processed > 0 else 0)
        
        return {
            'operations_count': len(metrics),
            'total_texts_processed': total_texts,
            'total_duration_seconds': round(total_duration, 3),
            'overall_throughput_per_sec': round(overall_throughput, 2),
            'average_memory_mb': round(avg_memory, 2),
            'average_cpu_percent': round(avg_cpu, 2),
            'average_cache_hit_rate': round(avg_cache_hit_rate, 2),
            'total_errors': total_errors,
            'error_rate_percent': round((total_errors / total_texts * 100), 2) if total_texts > 0 else 0,
            'fastest_operation': {
                'name': fastest_op.operation_name,
                'throughput': round(fastest_op.texts_processed / fastest_op.duration, 2) if fastest_op.duration > 0 else 0
            },
            'slowest_operation': {
                'name': slowest_op.operation_name,
                'throughput': round(slowest_op.texts_processed / slowest_op.duration, 2) if slowest_op.duration > 0 else 0
            },
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        # Analyze cache performance
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in metrics) / len(metrics)
        if avg_cache_hit_rate < 30:
            recommendations.append("Low cache hit rate - consider increasing cache size or TTL")
        
        # Analyze throughput
        throughputs = [m.texts_processed / m.duration for m in metrics if m.duration > 0]
        if throughputs and sum(throughputs) / len(throughputs) < 20:
            recommendations.append("Low throughput - consider using async processing for batch operations")
        
        # Analyze memory usage
        avg_memory = sum(m.memory_usage_mb for m in metrics) / len(metrics)
        if avg_memory > 300:
            recommendations.append("High memory usage - consider processing in smaller batches")
        
        # Analyze error rate
        total_texts = sum(m.texts_processed for m in metrics)
        total_errors = sum(m.errors_count for m in metrics)
        if total_texts > 0 and (total_errors / total_texts) > 0.05:
            recommendations.append("High error rate - review input validation and error handling")
        
        # Analyze operation patterns
        batch_ops = [m for m in metrics if 'batch' in m.operation_name.lower()]
        if len(batch_ops) > len(metrics) * 0.8:
            recommendations.append("Mostly batch operations - consider async processing for better performance")
        
        if not recommendations:
            recommendations.append("Performance looks good - no specific recommendations")
        
        return recommendations
    
    def clear_history(self):
        """Clear performance metrics history"""
        self.metrics_history.clear()
        logger.info("Performance metrics history cleared")
    
    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export all metrics as list of dictionaries"""
        return [metrics.to_dict() for metrics in self.metrics_history]


# Global performance monitor instance
performance_monitor = PerformanceMonitor()