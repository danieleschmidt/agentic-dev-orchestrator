#!/usr/bin/env python3
"""
Advanced Multi-Layer Caching System
High-performance caching with intelligent eviction, compression, and distributed support
"""

import asyncio
import time
import json
import pickle
import gzip
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import OrderedDict, defaultdict
from enum import Enum
import logging
import weakref
from functools import wraps
import psutil
from concurrent.futures import ThreadPoolExecutor
import sqlite3


logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache storage levels"""
    L1_MEMORY = "l1_memory"        # In-memory cache
    L2_COMPRESSED = "l2_compressed"  # Compressed memory cache
    L3_DISK = "l3_disk"           # Persistent disk cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache (Redis-like)


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    FIFO = "fifo"                 # First In, First Out
    ADAPTIVE = "adaptive"          # Adaptive based on usage patterns
    TTL = "ttl"                   # Time To Live only


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 1
    size_bytes: int = 0
    ttl: Optional[float] = None
    compressed: bool = False
    serialized: bool = False
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compressions: int = 0
    decompressions: int = 0
    serializations: int = 0
    deserializations: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    cache_levels_used: Dict[str, int] = None


class IntelligentCache:
    """Multi-layer intelligent caching system"""
    
    def __init__(self, 
                 l1_max_size: int = 10_000,           # Max items in L1 cache
                 l1_max_memory_mb: int = 512,         # Max memory for L1 cache
                 l2_max_size: int = 50_000,           # Max items in L2 cache  
                 l2_max_memory_mb: int = 1024,        # Max memory for L2 cache
                 l3_cache_dir: Optional[Path] = None, # L3 disk cache directory
                 l3_max_size_gb: float = 5.0,         # Max disk cache size
                 default_ttl: Optional[float] = 3600, # Default TTL in seconds
                 eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
                 enable_compression: bool = True,
                 compression_threshold: int = 1024):  # Compress items > 1KB
        
        # Configuration
        self.l1_max_size = l1_max_size
        self.l1_max_memory_mb = l1_max_memory_mb
        self.l2_max_size = l2_max_size
        self.l2_max_memory_mb = l2_max_memory_mb
        self.l3_max_size_gb = l3_max_size_gb
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # Storage layers
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()  # Fast in-memory
        self.l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()  # Compressed memory
        self.l3_cache_dir = l3_cache_dir or Path("cache/l3")
        self.l3_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # L3 disk cache database
        self.l3_db_path = self.l3_cache_dir / "cache.db"
        self._init_l3_database()
        
        # Access pattern tracking
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.key_frequency: Dict[str, int] = defaultdict(int)
        self.key_recency: Dict[str, float] = {}
        
        # Thread safety
        self.l1_lock = threading.RLock()
        self.l2_lock = threading.RLock()
        self.l3_lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats(cache_levels_used=defaultdict(int))
        self.stats_lock = threading.Lock()
        
        # Background maintenance
        self.maintenance_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache_maintenance")
        self._start_background_tasks()
        
        # Memory monitoring
        self.memory_pressure_threshold = 0.85  # Trigger eviction at 85% memory usage
        
        logger.info(f"Intelligent cache initialized with {len(CacheLevel)} levels")
    
    def _init_l3_database(self):
        """Initialize L3 disk cache database"""
        with sqlite3.connect(self.l3_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    size_bytes INTEGER,
                    ttl REAL,
                    compressed BOOLEAN,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON cache_entries(ttl)")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent level selection"""
        start_time = time.time()
        
        try:
            # L1 Cache check (fastest)
            with self.l1_lock:
                if key in self.l1_cache:
                    entry = self.l1_cache[key]
                    if self._is_valid(entry):
                        self._update_access_stats(entry)
                        self.l1_cache.move_to_end(key)  # LRU update
                        with self.stats_lock:
                            self.stats.hits += 1
                            self.stats.cache_levels_used[CacheLevel.L1_MEMORY.value] += 1
                        self._record_access_time(time.time() - start_time)
                        return entry.value
                    else:
                        del self.l1_cache[key]
            
            # L2 Cache check (compressed memory)
            with self.l2_lock:
                if key in self.l2_cache:
                    entry = self.l2_cache[key]
                    if self._is_valid(entry):
                        value = await self._decompress_value(entry)
                        self._update_access_stats(entry)
                        self.l2_cache.move_to_end(key)
                        
                        # Promote to L1 if frequently accessed
                        if self._should_promote_to_l1(entry):
                            await self._promote_to_l1(key, value, entry)
                        
                        with self.stats_lock:
                            self.stats.hits += 1
                            self.stats.cache_levels_used[CacheLevel.L2_COMPRESSED.value] += 1
                        self._record_access_time(time.time() - start_time)
                        return value
                    else:
                        del self.l2_cache[key]
            
            # L3 Cache check (disk)
            entry = await self._get_from_l3(key)
            if entry and self._is_valid(entry):
                value = await self._deserialize_value(entry)
                self._update_access_stats(entry)
                
                # Promote to appropriate level based on access pattern
                if self._should_promote_to_l2(entry):
                    await self._promote_to_l2(key, value, entry)
                elif self._should_promote_to_l1(entry):
                    await self._promote_to_l1(key, value, entry)
                
                with self.stats_lock:
                    self.stats.hits += 1
                    self.stats.cache_levels_used[CacheLevel.L3_DISK.value] += 1
                self._record_access_time(time.time() - start_time)
                return value
            
            # Cache miss
            with self.stats_lock:
                self.stats.misses += 1
            self._record_access_time(time.time() - start_time)
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None,
                  tags: Optional[List[str]] = None, force_level: Optional[CacheLevel] = None) -> bool:
        """Set value in cache with intelligent level selection"""
        try:
            current_time = time.time()
            entry_ttl = ttl or self.default_ttl
            
            # Calculate value size
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                size_bytes=size_bytes,
                ttl=entry_ttl,
                tags=tags or []
            )
            
            # Determine optimal cache level
            target_level = force_level or await self._determine_cache_level(key, value, size_bytes)
            
            # Store in appropriate level
            if target_level == CacheLevel.L1_MEMORY:
                await self._set_l1(key, entry)
            elif target_level == CacheLevel.L2_COMPRESSED:
                await self._set_l2(key, entry)
            elif target_level == CacheLevel.L3_DISK:
                await self._set_l3(key, entry)
            
            # Update access patterns
            self.key_frequency[key] += 1
            self.key_recency[key] = current_time
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        deleted = False
        
        try:
            # Remove from L1
            with self.l1_lock:
                if key in self.l1_cache:
                    del self.l1_cache[key]
                    deleted = True
            
            # Remove from L2
            with self.l2_lock:
                if key in self.l2_cache:
                    del self.l2_cache[key]
                    deleted = True
            
            # Remove from L3
            if await self._delete_from_l3(key):
                deleted = True
            
            # Clean up tracking data
            self.key_frequency.pop(key, None)
            self.key_recency.pop(key, None)
            self.access_patterns.pop(key, None)
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False
    
    async def clear(self, tags: Optional[List[str]] = None) -> int:
        """Clear cache entries, optionally by tags"""
        cleared_count = 0
        
        try:
            if tags:
                # Clear by tags
                keys_to_delete = []
                
                # Find keys with matching tags in L1
                with self.l1_lock:
                    for key, entry in self.l1_cache.items():
                        if entry.tags and any(tag in entry.tags for tag in tags):
                            keys_to_delete.append(key)
                
                # Find keys with matching tags in L2
                with self.l2_lock:
                    for key, entry in self.l2_cache.items():
                        if entry.tags and any(tag in entry.tags for tag in tags):
                            keys_to_delete.append(key)
                
                # Find keys with matching tags in L3
                l3_keys = await self._get_l3_keys_by_tags(tags)
                keys_to_delete.extend(l3_keys)
                
                # Delete all matching keys
                for key in set(keys_to_delete):
                    if await self.delete(key):
                        cleared_count += 1
            else:
                # Clear all
                with self.l1_lock:
                    cleared_count += len(self.l1_cache)
                    self.l1_cache.clear()
                
                with self.l2_lock:
                    cleared_count += len(self.l2_cache)
                    self.l2_cache.clear()
                
                cleared_count += await self._clear_l3()
                
                # Clear tracking data
                self.key_frequency.clear()
                self.key_recency.clear()
                self.access_patterns.clear()
            
            return cleared_count
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.stats_lock:
            stats_dict = asdict(self.stats)
        
        # Add current cache sizes
        stats_dict.update({
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache),
            'l3_size': await self._get_l3_size(),
            'hit_rate': self.stats.hits / max(self.stats.hits + self.stats.misses, 1),
            'memory_usage_mb': self._get_memory_usage_mb(),
            'disk_usage_gb': await self._get_disk_usage_gb(),
            'most_accessed_keys': self._get_most_accessed_keys(10)
        })
        
        return stats_dict
    
    async def _determine_cache_level(self, key: str, value: Any, size_bytes: int) -> CacheLevel:
        """Intelligently determine the best cache level for an item"""
        
        # Check memory pressure
        memory_usage = psutil.virtual_memory().percent / 100
        if memory_usage > self.memory_pressure_threshold:
            return CacheLevel.L3_DISK
        
        # Small, frequently accessed items go to L1
        if size_bytes < self.compression_threshold and self.key_frequency[key] > 10:
            return CacheLevel.L1_MEMORY
        
        # Medium-sized items or moderate frequency go to L2
        if size_bytes < self.compression_threshold * 10:
            return CacheLevel.L2_COMPRESSED
        
        # Large items go to L3
        return CacheLevel.L3_DISK
    
    async def _set_l1(self, key: str, entry: CacheEntry):
        """Set entry in L1 cache"""
        with self.l1_lock:
            # Check if we need to evict
            while (len(self.l1_cache) >= self.l1_max_size or 
                   self._get_l1_memory_usage() > self.l1_max_memory_mb * 1024 * 1024):
                await self._evict_from_l1()
            
            self.l1_cache[key] = entry
    
    async def _set_l2(self, key: str, entry: CacheEntry):
        """Set entry in L2 cache with compression"""
        # Compress if enabled and above threshold
        if self.enable_compression and entry.size_bytes > self.compression_threshold:
            entry.value = await self._compress_value(entry.value)
            entry.compressed = True
            with self.stats_lock:
                self.stats.compressions += 1
        
        with self.l2_lock:
            # Check if we need to evict
            while (len(self.l2_cache) >= self.l2_max_size or 
                   self._get_l2_memory_usage() > self.l2_max_memory_mb * 1024 * 1024):
                await self._evict_from_l2()
            
            self.l2_cache[key] = entry
    
    async def _set_l3(self, key: str, entry: CacheEntry):
        """Set entry in L3 disk cache"""
        with self.l3_lock:
            # Serialize value for disk storage
            serialized_value = await self._serialize_value(entry.value)
            entry.serialized = True
            
            with sqlite3.connect(self.l3_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, last_accessed, access_count, size_bytes, ttl, compressed, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key, serialized_value, entry.created_at, entry.last_accessed,
                    entry.access_count, entry.size_bytes, entry.ttl, entry.compressed,
                    json.dumps(entry.tags) if entry.tags else None,
                    json.dumps(entry.metadata) if entry.metadata else None
                ))
            
            # Check if we need to evict from L3
            if await self._get_disk_usage_gb() > self.l3_max_size_gb:
                await self._evict_from_l3()
    
    async def _evict_from_l1(self):
        """Evict item from L1 cache"""
        if not self.l1_cache:
            return
        
        # LRU eviction (item at the front is least recently used)
        key_to_evict = next(iter(self.l1_cache))
        entry = self.l1_cache.pop(key_to_evict)
        
        # Demote to L2 if valuable
        if self._should_demote_to_l2(entry):
            await self._set_l2(key_to_evict, entry)
        
        with self.stats_lock:
            self.stats.evictions += 1
    
    async def _evict_from_l2(self):
        """Evict item from L2 cache"""
        if not self.l2_cache:
            return
        
        key_to_evict = next(iter(self.l2_cache))
        entry = self.l2_cache.pop(key_to_evict)
        
        # Demote to L3 if valuable
        if self._should_demote_to_l3(entry):
            await self._set_l3(key_to_evict, entry)
        
        with self.stats_lock:
            self.stats.evictions += 1
    
    async def _evict_from_l3(self):
        """Evict items from L3 cache"""
        with sqlite3.connect(self.l3_db_path) as conn:
            # Remove least recently accessed items
            conn.execute("""
                DELETE FROM cache_entries 
                WHERE key IN (
                    SELECT key FROM cache_entries 
                    ORDER BY last_accessed ASC 
                    LIMIT 1000
                )
            """)
    
    async def _compress_value(self, value: Any) -> bytes:
        """Compress a value"""
        try:
            serialized = pickle.dumps(value)
            compressed = gzip.compress(serialized)
            return compressed
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return pickle.dumps(value)
    
    async def _decompress_value(self, entry: CacheEntry) -> Any:
        """Decompress a value"""
        try:
            if entry.compressed:
                decompressed = gzip.decompress(entry.value)
                value = pickle.loads(decompressed)
                with self.stats_lock:
                    self.stats.decompressions += 1
                return value
            else:
                return entry.value
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    async def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for disk storage"""
        try:
            with self.stats_lock:
                self.stats.serializations += 1
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    async def _deserialize_value(self, entry: CacheEntry) -> Any:
        """Deserialize value from disk storage"""
        try:
            with self.stats_lock:
                self.stats.deserializations += 1
            return pickle.loads(entry.value)
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        if entry.ttl is None:
            return True
        
        current_time = time.time()
        return (current_time - entry.created_at) < entry.ttl
    
    def _should_promote_to_l1(self, entry: CacheEntry) -> bool:
        """Determine if entry should be promoted to L1"""
        return (entry.access_count > 5 and 
                entry.size_bytes < self.compression_threshold and
                len(self.l1_cache) < self.l1_max_size * 0.8)
    
    def _should_promote_to_l2(self, entry: CacheEntry) -> bool:
        """Determine if entry should be promoted to L2"""
        return (entry.access_count > 2 and 
                len(self.l2_cache) < self.l2_max_size * 0.8)
    
    def _should_demote_to_l2(self, entry: CacheEntry) -> bool:
        """Determine if entry should be demoted to L2"""
        return entry.access_count > 1
    
    def _should_demote_to_l3(self, entry: CacheEntry) -> bool:
        """Determine if entry should be demoted to L3"""
        return entry.access_count > 1 or entry.size_bytes > self.compression_threshold * 2
    
    async def _promote_to_l1(self, key: str, value: Any, entry: CacheEntry):
        """Promote entry to L1 cache"""
        new_entry = CacheEntry(
            key=key, value=value, created_at=entry.created_at,
            last_accessed=time.time(), access_count=entry.access_count + 1,
            size_bytes=entry.size_bytes, ttl=entry.ttl, tags=entry.tags
        )
        await self._set_l1(key, new_entry)
    
    async def _promote_to_l2(self, key: str, value: Any, entry: CacheEntry):
        """Promote entry to L2 cache"""
        new_entry = CacheEntry(
            key=key, value=value, created_at=entry.created_at,
            last_accessed=time.time(), access_count=entry.access_count + 1,
            size_bytes=entry.size_bytes, ttl=entry.ttl, tags=entry.tags
        )
        await self._set_l2(key, new_entry)
    
    def _update_access_stats(self, entry: CacheEntry):
        """Update access statistics for entry"""
        current_time = time.time()
        entry.last_accessed = current_time
        entry.access_count += 1
        
        # Track access patterns
        self.access_patterns[entry.key].append(current_time)
        if len(self.access_patterns[entry.key]) > 100:
            self.access_patterns[entry.key] = self.access_patterns[entry.key][-100:]
    
    def _record_access_time(self, duration_seconds: float):
        """Record access time for performance tracking"""
        duration_ms = duration_seconds * 1000
        with self.stats_lock:
            # Simple moving average
            if self.stats.avg_access_time_ms == 0:
                self.stats.avg_access_time_ms = duration_ms
            else:
                self.stats.avg_access_time_ms = (self.stats.avg_access_time_ms * 0.9 + duration_ms * 0.1)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _get_l1_memory_usage(self) -> int:
        """Get L1 cache memory usage in bytes"""
        return sum(entry.size_bytes for entry in self.l1_cache.values())
    
    def _get_l2_memory_usage(self) -> int:
        """Get L2 cache memory usage in bytes"""
        return sum(entry.size_bytes for entry in self.l2_cache.values())
    
    def _get_memory_usage_mb(self) -> float:
        """Get total memory usage in MB"""
        l1_usage = self._get_l1_memory_usage()
        l2_usage = self._get_l2_memory_usage()
        return (l1_usage + l2_usage) / (1024 * 1024)
    
    async def _get_disk_usage_gb(self) -> float:
        """Get disk usage in GB"""
        try:
            total_size = 0
            for file_path in self.l3_cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024 * 1024)
        except:
            return 0.0
    
    async def _get_l3_size(self) -> int:
        """Get number of items in L3 cache"""
        try:
            with sqlite3.connect(self.l3_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                return cursor.fetchone()[0]
        except:
            return 0
    
    async def _get_from_l3(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L3 disk cache"""
        try:
            with sqlite3.connect(self.l3_db_path) as conn:
                cursor = conn.execute("""
                    SELECT value, created_at, last_accessed, access_count, size_bytes, ttl, compressed, tags, metadata
                    FROM cache_entries WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return CacheEntry(
                    key=key,
                    value=row[0],
                    created_at=row[1],
                    last_accessed=row[2],
                    access_count=row[3],
                    size_bytes=row[4],
                    ttl=row[5],
                    compressed=bool(row[6]),
                    tags=json.loads(row[7]) if row[7] else None,
                    metadata=json.loads(row[8]) if row[8] else None,
                    serialized=True
                )
        except Exception as e:
            logger.error(f"L3 get error: {e}")
            return None
    
    async def _delete_from_l3(self, key: str) -> bool:
        """Delete entry from L3 disk cache"""
        try:
            with sqlite3.connect(self.l3_db_path) as conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"L3 delete error: {e}")
            return False
    
    async def _clear_l3(self) -> int:
        """Clear L3 disk cache"""
        try:
            with sqlite3.connect(self.l3_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                count = cursor.fetchone()[0]
                conn.execute("DELETE FROM cache_entries")
                return count
        except Exception as e:
            logger.error(f"L3 clear error: {e}")
            return 0
    
    async def _get_l3_keys_by_tags(self, tags: List[str]) -> List[str]:
        """Get L3 cache keys by tags"""
        try:
            with sqlite3.connect(self.l3_db_path) as conn:
                keys = []
                for tag in tags:
                    cursor = conn.execute("""
                        SELECT key FROM cache_entries 
                        WHERE tags LIKE ?
                    """, (f'%"{tag}"%',))
                    keys.extend([row[0] for row in cursor.fetchall()])
                return list(set(keys))
        except Exception as e:
            logger.error(f"L3 tags query error: {e}")
            return []
    
    def _get_most_accessed_keys(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently accessed keys"""
        return sorted(self.key_frequency.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        def cleanup_expired():
            """Clean up expired entries"""
            try:
                current_time = time.time()
                
                # Clean L1
                with self.l1_lock:
                    expired_keys = [
                        key for key, entry in self.l1_cache.items()
                        if not self._is_valid(entry)
                    ]
                    for key in expired_keys:
                        del self.l1_cache[key]
                
                # Clean L2
                with self.l2_lock:
                    expired_keys = [
                        key for key, entry in self.l2_cache.items()
                        if not self._is_valid(entry)
                    ]
                    for key in expired_keys:
                        del self.l2_cache[key]
                
                # Clean L3
                with sqlite3.connect(self.l3_db_path) as conn:
                    conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE ttl IS NOT NULL AND (created_at + ttl) < ?
                    """, (current_time,))
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
        
        def optimize_cache():
            """Optimize cache layout based on access patterns"""
            try:
                # Implement cache optimization logic here
                pass
            except Exception as e:
                logger.error(f"Optimization task error: {e}")
        
        # Schedule background tasks
        def run_maintenance():
            while True:
                time.sleep(300)  # Run every 5 minutes
                cleanup_expired()
                time.sleep(3600)  # Run optimization every hour
                optimize_cache()
        
        self.maintenance_executor.submit(run_maintenance)


# Global cache instance
global_cache = IntelligentCache()


def cache_result(ttl: Optional[float] = None, 
                tags: Optional[List[str]] = None,
                key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
                cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await global_cache.set(cache_key, result, ttl=ttl, tags=tags)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
                cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
            
            # Try to get from cache (sync version)
            try:
                cached_result = asyncio.run(global_cache.get(cache_key))
                if cached_result is not None:
                    return cached_result
            except:
                pass
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            try:
                asyncio.run(global_cache.set(cache_key, result, ttl=ttl, tags=tags))
            except:
                pass
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


if __name__ == "__main__":
    import asyncio
    
    async def test_cache():
        cache = IntelligentCache(l1_max_size=5, l2_max_size=10)
        
        # Test basic operations
        await cache.set("key1", "value1")
        await cache.set("key2", {"data": "complex_value"}, ttl=60)
        await cache.set("key3", list(range(1000)), tags=["large_data"])
        
        print(f"key1: {await cache.get('key1')}")
        print(f"key2: {await cache.get('key2')}")
        print(f"key3 length: {len(await cache.get('key3'))}")
        
        # Test cache levels
        for i in range(20):
            await cache.set(f"test_{i}", f"value_{i}")
        
        # Get statistics
        stats = await cache.get_stats()
        print(f"Cache Stats: {json.dumps(stats, indent=2)}")
        
        # Test eviction
        print(f"L1 size after insertions: {len(cache.l1_cache)}")
        print(f"L2 size after insertions: {len(cache.l2_cache)}")
    
    # Run test
    asyncio.run(test_cache())