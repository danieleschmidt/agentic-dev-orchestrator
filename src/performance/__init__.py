"""Performance optimization modules"""

# Import available modules conditionally
__all__ = []

try:
    from .adaptive_cache import AdaptiveCache, CacheStrategy, CacheBackend
    __all__.extend(['AdaptiveCache', 'CacheStrategy', 'CacheBackend'])
except ImportError:
    pass

try:
    from .distributed_executor import DistributedExecutor, TaskDefinition, TaskResult, LoadBalancer
    __all__.extend(['DistributedExecutor', 'TaskDefinition', 'TaskResult', 'LoadBalancer'])
except ImportError:
    pass

try:
    from .async_executor import AsyncExecutor
    __all__.append('AsyncExecutor')
except ImportError:
    pass

try:
    from .metrics_collector import MetricsCollector
    __all__.append('MetricsCollector')
except ImportError:
    pass