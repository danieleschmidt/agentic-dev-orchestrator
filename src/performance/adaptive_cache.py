#!/usr/bin/env python3
"""
Adaptive Caching System for ADO
Implements intelligent caching with automatic optimization and TTL management
"""

import json
import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, OrderedDict
from enum import Enum
import statistics
import pickle
from abc import ABC, abstractmethod


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CacheBackend(Enum):
    """Cache storage backends"""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    hit_rate: float = 0.0
    importance_score: float = 0.0
    
    @property
    def age(self) -> float:
        """Age of entry in seconds"""
        return time.time() - self.created_at
        
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
        
    def update_access(self):
        """Update access metadata"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    memory_efficiency: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests


class CacheBackendInterface(ABC):
    """Abstract interface for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key"""
        pass
        
    @abstractmethod
    def put(self, entry: CacheEntry) -> bool:
        """Store entry"""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete entry by key"""
        pass
        
    @abstractmethod
    def clear(self):
        """Clear all entries"""
        pass
        
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all keys"""
        pass
        
    @abstractmethod
    def size(self) -> int:
        """Get total size in bytes"""
        pass


class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend using OrderedDict for LRU"""
    
    def __init__(self, max_size_bytes: int = 100 * 1024 * 1024):  # 100MB default
        self.max_size_bytes = max_size_bytes
        self.storage: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            if key in self.storage:
                # Move to end for LRU
                entry = self.storage.pop(key)
                self.storage[key] = entry
                entry.update_access()
                return entry
        return None
        
    def put(self, entry: CacheEntry) -> bool:
        with self.lock:
            # Calculate entry size
            try:
                entry.size_bytes = len(pickle.dumps(entry.value))
            except Exception:
                entry.size_bytes = len(str(entry.value).encode())
                
            # Check if we need to evict entries
            while (self.size() + entry.size_bytes > self.max_size_bytes and 
                   len(self.storage) > 0):
                # Remove oldest entry (LRU)
                oldest_key = next(iter(self.storage))
                del self.storage[oldest_key]
                
            self.storage[entry.key] = entry
            return True
            
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.storage:
                del self.storage[key]
                return True
        return False
        
    def clear(self):
        with self.lock:
            self.storage.clear()
            
    def keys(self) -> List[str]:
        with self.lock:
            return list(self.storage.keys())
            
    def size(self) -> int:
        with self.lock:
            return sum(entry.size_bytes for entry in self.storage.values())


class DiskCacheBackend(CacheBackendInterface):
    """Disk-based cache backend"""
    
    def __init__(self, cache_dir: Path, max_size_bytes: int = 1024 * 1024 * 1024):  # 1GB default
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_bytes
        self.index_file = self.cache_dir / "index.json"
        self.lock = threading.RLock()
        self._load_index()
        
    def _load_index(self):
        """Load cache index from disk"""
        self.index = {}
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    self.index = {k: CacheEntry(**v) for k, v in data.items()}
            except Exception as e:
                logging.warning(f"Could not load cache index: {e}")
                
    def _save_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                data = {k: asdict(v) for k, v in self.index.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save cache index: {e}")
            
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
        
    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            if key in self.index:
                entry = self.index[key]
                if entry.is_expired:
                    self.delete(key)
                    return None
                    
                # Load value from disk
                file_path = self._get_file_path(key)
                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as f:
                            entry.value = pickle.load(f)
                        entry.update_access()
                        return entry
                    except Exception as e:
                        logging.warning(f"Could not load cache file {file_path}: {e}")
                        self.delete(key)
        return None
        
    def put(self, entry: CacheEntry) -> bool:
        with self.lock:
            file_path = self._get_file_path(entry.key)
            
            try:
                # Save value to disk
                with open(file_path, 'wb') as f:
                    pickle.dump(entry.value, f)
                    
                entry.size_bytes = file_path.stat().st_size
                
                # Check size limits and evict if needed
                while self._total_size() + entry.size_bytes > self.max_size_bytes:
                    if not self._evict_oldest():
                        break
                        
                # Don't store value in index (it's on disk)
                entry_copy = CacheEntry(
                    key=entry.key,
                    value=None,  # Value is on disk
                    created_at=entry.created_at,
                    accessed_at=entry.accessed_at,
                    access_count=entry.access_count,
                    size_bytes=entry.size_bytes,
                    ttl=entry.ttl,
                    hit_rate=entry.hit_rate,
                    importance_score=entry.importance_score
                )
                
                self.index[entry.key] = entry_copy
                self._save_index()
                return True
                
            except Exception as e:
                logging.warning(f"Could not save cache entry {entry.key}: {e}")
                return False
                
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.index:
                file_path = self._get_file_path(key)
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    logging.warning(f"Could not delete cache file {file_path}: {e}")
                    
                del self.index[key]
                self._save_index()
                return True
        return False
        
    def clear(self):
        with self.lock:
            for key in list(self.index.keys()):
                self.delete(key)
                
    def keys(self) -> List[str]:
        with self.lock:
            return list(self.index.keys())
            
    def size(self) -> int:
        return self._total_size()
        
    def _total_size(self) -> int:
        """Calculate total size of all entries"""
        return sum(entry.size_bytes for entry in self.index.values())
        
    def _evict_oldest(self) -> bool:
        """Evict oldest entry"""
        if not self.index:
            return False
            
        oldest_key = min(self.index.keys(), 
                        key=lambda k: self.index[k].created_at)
        return self.delete(oldest_key)


class AdaptiveCache:
    """Adaptive caching system with intelligent optimization"""
    
    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        backend: CacheBackend = CacheBackend.MEMORY,
        max_size_bytes: int = 100 * 1024 * 1024,
        default_ttl: Optional[float] = 3600.0,  # 1 hour
        cache_dir: Optional[Path] = None,
        enable_stats: bool = True
    ):
        self.strategy = strategy
        self.backend_type = backend
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        
        # Initialize backend
        if backend == CacheBackend.MEMORY:
            self.backend = MemoryCacheBackend(max_size_bytes)
        elif backend == CacheBackend.DISK:
            if cache_dir is None:
                cache_dir = Path("cache")
            self.backend = DiskCacheBackend(cache_dir, max_size_bytes)
        else:  # HYBRID
            # Use memory for small, frequently accessed items
            # and disk for larger, less frequent items
            self.memory_backend = MemoryCacheBackend(max_size_bytes // 4)
            self.disk_backend = DiskCacheBackend(
                cache_dir or Path("cache"), 
                max_size_bytes
            )
            
        # Metrics and monitoring
        self.metrics = CacheMetrics()
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.key_importance: Dict[str, float] = defaultdict(float)
        
        # Adaptive learning parameters
        self.learning_window = 1000  # Number of accesses to consider
        self.importance_decay = 0.95
        
        # Background maintenance
        self.lock = threading.RLock()
        self.maintenance_interval = 60.0  # 1 minute
        self.maintenance_thread = None
        self.running = True
        
        self.logger = logging.getLogger("adaptive_cache")
        self._start_maintenance()
        
    def get(self, key: str) -> Any:
        """Get value from cache"""
        start_time = time.time()
        
        with self.lock:
            self.metrics.total_requests += 1
            
            # Try to get from appropriate backend
            entry = self._get_from_backend(key)
            
            if entry is not None and not entry.is_expired:
                # Cache hit
                self.metrics.cache_hits += 1
                entry.update_access()
                
                # Update access patterns for adaptive learning
                self._update_access_pattern(key)
                
                # Update backend if necessary (for hybrid)
                self._maybe_promote_to_memory(entry)
                
                if self.enable_stats:
                    access_time = time.time() - start_time
                    self._update_access_time(access_time)
                    
                return entry.value
            else:
                # Cache miss or expired
                self.metrics.cache_misses += 1
                if entry and entry.is_expired:
                    self._delete_from_backend(key)
                    
        return None
        
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache"""
        if ttl is None:
            ttl = self.default_ttl
            
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            ttl=ttl
        )
        
        with self.lock:
            success = self._put_to_backend(entry)
            if success:
                # Initialize importance score
                self.key_importance[key] = 1.0
                
            return success
            
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            success = self._delete_from_backend(key)
            if success:
                self.access_patterns.pop(key, None)
                self.key_importance.pop(key, None)
            return success
            
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            if self.backend_type == CacheBackend.HYBRID:
                self.memory_backend.clear()
                self.disk_backend.clear()
            else:
                self.backend.clear()
                
            self.access_patterns.clear()
            self.key_importance.clear()
            self.metrics = CacheMetrics()
            
    def get_or_compute(self, key: str, compute_func: Callable[[], Any], 
                      ttl: Optional[float] = None) -> Any:
        """Get value from cache or compute and cache it"""
        value = self.get(key)
        if value is not None:
            return value
            
        # Compute value
        try:
            computed_value = compute_func()
            self.put(key, computed_value, ttl)
            return computed_value
        except Exception as e:
            self.logger.error(f"Error computing value for key {key}: {e}")
            raise
            
    def _get_from_backend(self, key: str) -> Optional[CacheEntry]:
        """Get entry from appropriate backend"""
        if self.backend_type == CacheBackend.HYBRID:
            # Try memory first, then disk
            entry = self.memory_backend.get(key)
            if entry:
                return entry
            return self.disk_backend.get(key)
        else:
            return self.backend.get(key)
            
    def _put_to_backend(self, entry: CacheEntry) -> bool:
        """Put entry to appropriate backend"""
        if self.backend_type == CacheBackend.HYBRID:
            # Decide based on entry characteristics
            if self._should_use_memory(entry):
                return self.memory_backend.put(entry)
            else:
                return self.disk_backend.put(entry)
        else:
            return self.backend.put(entry)
            
    def _delete_from_backend(self, key: str) -> bool:
        """Delete from appropriate backend"""
        if self.backend_type == CacheBackend.HYBRID:
            mem_deleted = self.memory_backend.delete(key)
            disk_deleted = self.disk_backend.delete(key)
            return mem_deleted or disk_deleted
        else:
            return self.backend.delete(key)
            
    def _should_use_memory(self, entry: CacheEntry) -> bool:
        """Decide whether to store entry in memory or disk"""
        # Use memory for small, important, frequently accessed items
        importance = self.key_importance.get(entry.key, 1.0)
        size_factor = max(0.1, 1.0 - (entry.size_bytes / (1024 * 1024)))  # Prefer smaller items
        
        return importance * size_factor > 0.5
        
    def _maybe_promote_to_memory(self, entry: CacheEntry):
        """Promote frequently accessed items to memory cache"""
        if (self.backend_type == CacheBackend.HYBRID and 
            entry.access_count > 10 and 
            self.key_importance.get(entry.key, 0) > 2.0):
            
            # Check if it's in disk cache and should be promoted
            if not self.memory_backend.get(entry.key):
                self.memory_backend.put(entry)
                
    def _update_access_pattern(self, key: str):
        """Update access pattern for adaptive learning"""
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses
        cutoff = current_time - 3600  # 1 hour window
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
        
        # Update importance score
        access_frequency = len(self.access_patterns[key]) / 3600  # accesses per second
        self.key_importance[key] = (
            self.key_importance.get(key, 1.0) * self.importance_decay + 
            access_frequency * 10
        )
        
    def _update_access_time(self, access_time: float):
        """Update average access time metric"""
        if self.metrics.total_requests == 1:
            self.metrics.avg_access_time = access_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_access_time = (
                alpha * access_time + 
                (1 - alpha) * self.metrics.avg_access_time
            )
            
    def _start_maintenance(self):
        """Start background maintenance thread"""
        def maintenance_loop():
            while self.running:
                try:
                    time.sleep(self.maintenance_interval)
                    if self.running:
                        self._run_maintenance()
                except Exception as e:
                    self.logger.error(f"Error in maintenance thread: {e}")
                    
        self.maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        
    def _run_maintenance(self):
        """Run maintenance tasks"""
        with self.lock:
            # Update metrics
            self.metrics.update_hit_rate()
            
            # Calculate memory efficiency
            if self.backend_type == CacheBackend.MEMORY:
                self.metrics.total_size_bytes = self.backend.size()
            elif self.backend_type == CacheBackend.DISK:
                self.metrics.total_size_bytes = self.backend.size()
            elif self.backend_type == CacheBackend.HYBRID:
                self.metrics.total_size_bytes = (
                    self.memory_backend.size() + self.disk_backend.size()
                )
                
            # Adaptive strategy optimization
            if self.strategy == CacheStrategy.ADAPTIVE:
                self._optimize_cache_strategy()
                
            # Cleanup expired entries
            self._cleanup_expired_entries()
            
            # Decay importance scores
            for key in self.key_importance:
                self.key_importance[key] *= self.importance_decay
                
    def _optimize_cache_strategy(self):
        """Optimize cache based on access patterns"""
        if not self.access_patterns:
            return
            
        # Analyze access patterns
        total_accesses = sum(len(accesses) for accesses in self.access_patterns.values())
        if total_accesses < 100:  # Not enough data
            return
            
        # Identify hot keys
        hot_keys = []
        for key, accesses in self.access_patterns.items():
            if len(accesses) > total_accesses * 0.1:  # More than 10% of total accesses
                hot_keys.append(key)
                
        # Increase importance of hot keys
        for key in hot_keys:
            self.key_importance[key] = max(self.key_importance[key], 5.0)
            
        # For hybrid backend, ensure hot keys are in memory
        if self.backend_type == CacheBackend.HYBRID:
            for key in hot_keys:
                entry = self.disk_backend.get(key)
                if entry and not self.memory_backend.get(key):
                    self.memory_backend.put(entry)
                    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        if self.backend_type == CacheBackend.HYBRID:
            expired_keys = []
            
            for key in self.memory_backend.keys():
                entry = self.memory_backend.get(key)
                if entry and entry.is_expired:
                    expired_keys.append(key)
                    
            for key in self.disk_backend.keys():
                entry = self.disk_backend.get(key)
                if entry and entry.is_expired:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                self.delete(key)
        else:
            expired_keys = []
            for key in self.backend.keys():
                entry = self.backend.get(key)
                if entry and entry.is_expired:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                self.delete(key)
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        with self.lock:
            self.metrics.update_hit_rate()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_requests': self.metrics.total_requests,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'hit_rate': self.metrics.hit_rate,
                'total_size_bytes': self.metrics.total_size_bytes,
                'avg_access_time_ms': self.metrics.avg_access_time * 1000,
                'strategy': self.strategy.value,
                'backend': self.backend_type.value,
                'total_keys': len(self.key_importance),
                'hot_keys': len([k for k, v in self.key_importance.items() if v > 2.0]),
                'memory_backend_size': self.memory_backend.size() if self.backend_type == CacheBackend.HYBRID else 0,
                'disk_backend_size': self.disk_backend.size() if self.backend_type == CacheBackend.HYBRID else 0
            }
            
    def stop(self):
        """Stop the cache and cleanup resources"""
        self.running = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5.0)


def main():
    """CLI entry point for adaptive cache testing"""
    import sys
    import random
    
    if len(sys.argv) < 2:
        print("Usage: python adaptive_cache.py <command> [options]")
        print("Commands:")
        print("  demo - Run demonstration")
        print("  benchmark - Run performance benchmark")
        return
        
    command = sys.argv[1]
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if command == "demo":
        print("🚀 Running adaptive cache demo...")
        
        # Create cache with hybrid backend
        cache = AdaptiveCache(
            strategy=CacheStrategy.ADAPTIVE,
            backend=CacheBackend.HYBRID,
            max_size_bytes=10 * 1024 * 1024,  # 10MB
            default_ttl=300.0  # 5 minutes
        )
        
        try:
            # Demo 1: Basic operations
            print("\n1. Basic cache operations:")
            cache.put("user:123", {"name": "John", "email": "john@example.com"})
            cache.put("config:app", {"debug": False, "max_connections": 100})
            
            user = cache.get("user:123")
            print(f"   Retrieved user: {user}")
            
            # Demo 2: Compute function
            print("\n2. Get-or-compute pattern:")
            
            def expensive_computation(x: int) -> int:
                """Simulate expensive computation"""
                time.sleep(0.1)  # Simulate work
                return x ** 2 + x * 10
                
            start_time = time.time()
            result1 = cache.get_or_compute("compute:100", lambda: expensive_computation(100))
            first_call_time = time.time() - start_time
            
            start_time = time.time()
            result2 = cache.get_or_compute("compute:100", lambda: expensive_computation(100))
            second_call_time = time.time() - start_time
            
            print(f"   First call (computed): {result1} in {first_call_time:.3f}s")
            print(f"   Second call (cached): {result2} in {second_call_time:.3f}s")
            print(f"   Speedup: {first_call_time/second_call_time:.1f}x")
            
            # Demo 3: Access patterns
            print("\n3. Simulating access patterns:")
            
            # Create some data with different access patterns
            hot_keys = [f"hot:{i}" for i in range(10)]
            cold_keys = [f"cold:{i}" for i in range(50)]
            
            # Put data
            for key in hot_keys + cold_keys:
                cache.put(key, f"value for {key}")
                
            # Simulate access patterns
            for _ in range(1000):
                if random.random() < 0.8:  # 80% chance to access hot keys
                    key = random.choice(hot_keys)
                else:
                    key = random.choice(cold_keys)
                cache.get(key)
                
            # Show metrics
            metrics = cache.get_metrics()
            print(f"   Total requests: {metrics['total_requests']}")
            print(f"   Hit rate: {metrics['hit_rate']:.2%}")
            print(f"   Hot keys identified: {metrics['hot_keys']}")
            print(f"   Memory cache size: {metrics['memory_backend_size']} bytes")
            print(f"   Disk cache size: {metrics['disk_backend_size']} bytes")
            
        finally:
            cache.stop()
            
    elif command == "benchmark":
        print("🏁 Running cache benchmark...")
        
        strategies = [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]
        backends = [CacheBackend.MEMORY, CacheBackend.DISK]
        
        results = {}
        
        for strategy in strategies:
            for backend in backends:
                print(f"\nTesting {strategy.value} with {backend.value} backend...")
                
                cache = AdaptiveCache(
                    strategy=strategy,
                    backend=backend,
                    max_size_bytes=1024 * 1024,  # 1MB
                    cache_dir=Path(f"benchmark_cache_{backend.value}")
                )
                
                try:
                    # Benchmark parameters
                    num_keys = 1000
                    num_operations = 10000
                    
                    # Pre-populate cache
                    for i in range(num_keys):
                        cache.put(f"key:{i}", f"value:{i}" * 100)  # ~800 bytes per entry
                        
                    # Run benchmark
                    start_time = time.time()
                    
                    for _ in range(num_operations):
                        key = f"key:{random.randint(0, num_keys - 1)}"
                        cache.get(key)
                        
                    end_time = time.time()
                    
                    # Collect results
                    metrics = cache.get_metrics()
                    results[f"{strategy.value}_{backend.value}"] = {
                        'duration': end_time - start_time,
                        'ops_per_sec': num_operations / (end_time - start_time),
                        'hit_rate': metrics['hit_rate'],
                        'avg_access_time_ms': metrics['avg_access_time_ms']
                    }
                    
                    print(f"   Operations/sec: {results[f'{strategy.value}_{backend.value}']['ops_per_sec']:.0f}")
                    print(f"   Hit rate: {metrics['hit_rate']:.2%}")
                    print(f"   Avg access time: {metrics['avg_access_time_ms']:.2f}ms")
                    
                finally:
                    cache.stop()
                    
        # Print comparison
        print("\n📄 Benchmark Summary:")
        for config, result in results.items():
            print(f"  {config}:")
            print(f"    {result['ops_per_sec']:.0f} ops/sec, {result['hit_rate']:.2%} hit rate")
            
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
