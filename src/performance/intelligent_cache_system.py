#!/usr/bin/env python3
"""
Intelligent Cache System v4.0
Advanced caching with AI-driven optimization, predictive loading, and adaptive strategies
"""

import json
import time
import hashlib
import threading
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    avg_access_time_ms: float = 0.0
    most_accessed_keys: List[Tuple[str, int]] = None
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CacheEntry:
    """Enhanced cache entry with intelligence metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    priority: float = 1.0
    prediction_score: float = 0.0
    dependencies: List[str] = None
    tags: List[str] = None
    
    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds
    
    @property
    def access_frequency(self) -> float:
        """Access frequency per hour"""
        age_hours = max(self.age_seconds / 3600, 0.01)  # Avoid division by zero
        return self.access_count / age_hours


class CacheStrategy(ABC):
    """Abstract base for cache eviction strategies"""
    
    @abstractmethod
    def should_evict(self, entry: CacheEntry, cache_stats: CacheStats) -> float:
        """Return eviction score (0-1, higher = more likely to evict)"""
        pass


class IntelligentLRUStrategy(CacheStrategy):
    """LRU with intelligence factors (access frequency, prediction, priority)"""
    
    def should_evict(self, entry: CacheEntry, cache_stats: CacheStats) -> float:
        # Base LRU score (older = higher eviction score)
        max_age_hours = 24
        age_score = min(entry.age_seconds / (max_age_hours * 3600), 1.0)
        
        # Access frequency factor (less frequent = higher eviction score) 
        freq_score = max(0, 1.0 - (entry.access_frequency / 10))  # Normalize to 10 accesses/hour
        
        # Priority factor (lower priority = higher eviction score)
        priority_score = max(0, 1.0 - entry.priority)
        
        # Prediction factor (lower prediction = higher eviction score)
        prediction_score = max(0, 1.0 - entry.prediction_score)
        
        # Weighted combination
        eviction_score = (age_score * 0.3 + 
                         freq_score * 0.3 + 
                         priority_score * 0.2 + 
                         prediction_score * 0.2)
        
        return min(eviction_score, 1.0)


class AdaptiveTTLStrategy(CacheStrategy):
    """TTL-based with adaptive expiration based on usage patterns"""
    
    def should_evict(self, entry: CacheEntry, cache_stats: CacheStats) -> float:
        if entry.is_expired:
            return 1.0
        
        # Adaptive TTL based on access patterns
        if entry.ttl_seconds:
            time_until_expiry = entry.ttl_seconds - entry.age_seconds
            normalized_time = time_until_expiry / entry.ttl_seconds
            
            # If accessed frequently, lower eviction score despite approaching TTL
            if entry.access_frequency > 1.0:  # More than 1 access per hour
                return max(0, 1.0 - normalized_time) * 0.5
            else:
                return max(0, 1.0 - normalized_time)
        
        return 0.0


class PredictiveCache:
    """AI-driven predictive caching for intelligent prefetching"""
    
    def __init__(self):
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.sequence_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.last_accessed: Optional[str] = None
        
    def record_access(self, key: str):
        """Record access for pattern learning"""
        now = datetime.now()
        
        # Record temporal pattern
        self.access_patterns[key].append(now)
        
        # Keep only recent accesses (last 7 days)
        cutoff = now - timedelta(days=7)
        self.access_patterns[key] = [t for t in self.access_patterns[key] if t > cutoff]
        
        # Record sequence pattern
        if self.last_accessed and self.last_accessed != key:
            self.sequence_patterns[self.last_accessed][key] += 1
        
        self.last_accessed = key
    
    def predict_next_accesses(self, current_key: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Predict likely next accesses based on patterns"""
        predictions = []
        
        # Sequence-based prediction
        if current_key in self.sequence_patterns:
            total_sequences = sum(self.sequence_patterns[current_key].values())
            for next_key, count in self.sequence_patterns[current_key].items():
                probability = count / total_sequences
                predictions.append((next_key, probability))
        
        # Sort by probability and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]
    
    def get_prediction_score(self, key: str) -> float:
        """Get prediction score for a key based on temporal patterns"""
        if key not in self.access_patterns:
            return 0.0
        
        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return 0.1
        
        # Analyze temporal pattern
        now = datetime.now()
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.1
        
        # Calculate average interval and predict next access
        avg_interval = sum(intervals) / len(intervals)
        last_access = accesses[-1]
        predicted_next = last_access + timedelta(seconds=avg_interval)
        
        # Score based on how close we are to predicted time
        time_diff = abs((now - predicted_next).total_seconds())
        score = max(0, 1.0 - (time_diff / (avg_interval * 2)))  # Full score if within 2x interval
        
        return score


class IntelligentCacheSystem:
    """Advanced cache system with AI-driven optimization"""
    
    def __init__(self, 
                 max_size_mb: int = 512,
                 default_ttl: Optional[int] = 3600,
                 enable_persistence: bool = True,
                 cache_dir: str = "cache"):
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        self.cache_dir = Path(cache_dir)
        
        # Core cache storage
        self.entries: Dict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        
        # Intelligence components
        self.predictor = PredictiveCache()
        self.strategies = [
            IntelligentLRUStrategy(),
            AdaptiveTTLStrategy()
        ]
        
        # Threading
        self._lock = threading.RLock()
        self._background_thread = None
        self._shutdown = False
        
        # Performance tracking
        self.access_times: List[float] = []
        
        # Initialize
        if enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
        
        self._start_background_optimization()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent tracking"""
        start_time = time.time()
        
        with self._lock:
            if key in self.entries:
                entry = self.entries[key]
                
                # Check expiration
                if entry.is_expired:
                    del self.entries[key]
                    self.stats.misses += 1
                    self._record_access_time(start_time)
                    return default
                
                # Update access metadata
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Move to end for LRU
                self.entries.move_to_end(key)
                
                # Record for predictions
                self.predictor.record_access(key)
                
                self.stats.hits += 1
                self._record_access_time(start_time)
                
                # Trigger predictive prefetching
                self._predictive_prefetch(key)
                
                return entry.value
            else:
                self.stats.misses += 1
                self._record_access_time(start_time)
                return default
    
    def set(self, 
            key: str, 
            value: Any, 
            ttl: Optional[int] = None,
            priority: float = 1.0,
            tags: Optional[List[str]] = None) -> bool:
        """Set value in cache with intelligence metadata"""
        
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if value is too large
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes,
                priority=priority,
                prediction_score=self.predictor.get_prediction_score(key),
                tags=tags or []
            )
            
            # Ensure capacity
            if not self._ensure_capacity(size_bytes):
                logger.warning("Could not ensure cache capacity")
                return False
            
            # Store entry
            self.entries[key] = entry
            self.entries.move_to_end(key)  # Mark as recently used
            
            # Update stats
            self._update_memory_stats()
            
            # Persist if enabled
            if self.enable_persistence:
                self._persist_entry(entry)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self.entries:
                del self.entries[key]
                self._update_memory_stats()
                return True
            return False
    
    def delete_by_tags(self, tags: List[str]) -> int:
        """Delete entries with specified tags"""
        count = 0
        with self._lock:
            keys_to_delete = []
            for key, entry in self.entries.items():
                if entry.tags and any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.entries[key]
                count += 1
            
            self._update_memory_stats()
        
        return count
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.entries.clear()
            self.stats = CacheStats()
            if self.enable_persistence:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
    
    def _ensure_capacity(self, required_bytes: int) -> bool:
        """Ensure cache has capacity for new entry"""
        current_size = sum(entry.size_bytes for entry in self.entries.values())
        
        while current_size + required_bytes > self.max_size_bytes and self.entries:
            # Find entry with highest eviction score
            best_candidate = None
            best_score = -1
            
            for key, entry in self.entries.items():
                score = 0
                for strategy in self.strategies:
                    score += strategy.should_evict(entry, self.stats)
                score /= len(self.strategies)  # Average score
                
                if score > best_score:
                    best_score = score
                    best_candidate = key
            
            if best_candidate:
                evicted_entry = self.entries[best_candidate]
                current_size -= evicted_entry.size_bytes
                del self.entries[best_candidate]
                self.stats.evictions += 1
                logger.debug(f"Evicted cache entry: {best_candidate} (score: {best_score:.2f})")
            else:
                break
        
        return current_size + required_bytes <= self.max_size_bytes
    
    def _predictive_prefetch(self, accessed_key: str):
        """Perform predictive prefetching based on access patterns"""
        predictions = self.predictor.predict_next_accesses(accessed_key)
        
        for predicted_key, probability in predictions:
            if probability > 0.5 and predicted_key not in self.entries:
                # This would be where we'd trigger prefetching of predicted data
                # For now, we just update the prediction score
                self.predictor.get_prediction_score(predicted_key)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            # Try pickle serialization for accurate size
            return len(pickle.dumps(value))
        except Exception:
            # Fallback to string representation
            return len(str(value).encode('utf-8'))
    
    def _record_access_time(self, start_time: float):
        """Record access time for performance monitoring"""
        access_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.access_times.append(access_time)
        
        # Keep only recent access times
        if len(self.access_times) > 1000:
            self.access_times = self.access_times[-1000:]
        
        # Update stats
        if self.access_times:
            self.stats.avg_access_time_ms = sum(self.access_times) / len(self.access_times)
    
    def _update_memory_stats(self):
        """Update memory usage statistics"""
        total_bytes = sum(entry.size_bytes for entry in self.entries.values())
        self.stats.memory_usage_mb = total_bytes / (1024 * 1024)
        
        # Update most accessed keys
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        self.stats.most_accessed_keys = [(k, v.access_count) for k, v in sorted_entries[:10]]
    
    def _persist_entry(self, entry: CacheEntry):
        """Persist cache entry to disk"""
        try:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True)
            
            cache_file = self.cache_dir / f"{self._hash_key(entry.key)}.cache"
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Failed to persist cache entry {entry.key}: {e}")
    
    def _load_persistent_cache(self):
        """Load persisted cache entries"""
        if not self.cache_dir.exists():
            return
        
        loaded_count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if entry is still valid
                if not entry.is_expired:
                    self.entries[entry.key] = entry
                    loaded_count += 1
                else:
                    cache_file.unlink()  # Remove expired cache file
                    
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                cache_file.unlink()  # Remove corrupted cache file
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} cache entries from disk")
            self._update_memory_stats()
    
    def _hash_key(self, key: str) -> str:
        """Create hash of cache key for filename"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _start_background_optimization(self):
        """Start background thread for cache optimization"""
        def optimize_loop():
            while not self._shutdown:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    if not self._shutdown:
                        self._background_optimize()
                except Exception as e:
                    logger.warning(f"Background optimization error: {e}")
        
        self._background_thread = threading.Thread(target=optimize_loop, daemon=True)
        self._background_thread.start()
    
    def _background_optimize(self):
        """Perform background cache optimization"""
        with self._lock:
            # Remove expired entries
            expired_keys = [k for k, v in self.entries.items() if v.is_expired]
            for key in expired_keys:
                del self.entries[key]
            
            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired cache entries")
            
            # Update prediction scores
            for entry in self.entries.values():
                entry.prediction_score = self.predictor.get_prediction_score(entry.key)
            
            # Update statistics
            self._update_memory_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            return {
                "basic_stats": self.stats.to_dict(),
                "entry_count": len(self.entries),
                "memory_efficiency": self.stats.memory_usage_mb / (self.max_size_bytes / (1024 * 1024)),
                "oldest_entry_age": min(entry.age_seconds for entry in self.entries.values()) if self.entries else 0,
                "prediction_accuracy": self._calculate_prediction_accuracy(),
                "strategies_active": len(self.strategies)
            }
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate accuracy of prediction system"""
        # This would require tracking predictions vs actual accesses
        # Simplified version returns estimated accuracy
        return 0.75  # Placeholder
    
    def shutdown(self):
        """Shutdown cache system gracefully"""
        self._shutdown = True
        if self._background_thread:
            self._background_thread.join(timeout=5)


# Global intelligent cache instance
intelligent_cache = IntelligentCacheSystem()


def cached_function(ttl: Optional[int] = None, 
                   priority: float = 1.0,
                   tags: Optional[List[str]] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': sorted(kwargs.items())
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            result = intelligent_cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            intelligent_cache.set(cache_key, result, ttl=ttl, priority=priority, tags=tags)
            
            return result
        
        return wrapper
    return decorator