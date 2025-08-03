#!/usr/bin/env python3
"""
Cache Manager for ADO
Provides in-memory caching and optional Redis integration
"""

import json
import time
from typing import Any, Optional, Dict, Set
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with expiration"""
    value: Any
    expires_at: float
    created_at: float
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() > self.expires_at


class InMemoryCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl  # seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_counts: Dict[str, int] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if entry.is_expired():
            self.delete(key)
            return None
        
        # Track access
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        ttl = ttl or self.default_ttl
        now = time.time()
        
        entry = CacheEntry(
            value=value,
            expires_at=now + ttl,
            created_at=now
        )
        
        self._cache[key] = entry
        self._access_counts[key] = 0
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self._cache:
            del self._cache[key]
            self._access_counts.pop(key, None)
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._access_counts.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self._cache)
        expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'most_accessed_keys': sorted(
                self._access_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }


class FileCache:
    """File-based cache for persistent caching"""
    
    def __init__(self, cache_dir: str = ".ado/cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key"""
        # Use hash to avoid filesystem issues with special characters
        import hashlib
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        cache_file = self._get_cache_file(key)
        
        try:
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            entry = CacheEntry(**data)
            if entry.is_expired():
                self.delete(key)
                return None
            
            return entry.value
            
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache"""
        ttl = ttl or self.default_ttl
        now = time.time()
        
        entry = CacheEntry(
            value=value,
            expires_at=now + ttl,
            created_at=now
        )
        
        cache_file = self._get_cache_file(key)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'value': entry.value,
                    'expires_at': entry.expires_at,
                    'created_at': entry.created_at
                }, f, default=str)
            return True
            
        except Exception as e:
            logger.error(f"Failed to write cache file {cache_file}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from file cache"""
        cache_file = self._get_cache_file(key)
        
        try:
            if cache_file.exists():
                cache_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache file {cache_file}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return self.get(key) is not None
    
    def clear(self) -> int:
        """Clear all cache files"""
        deleted_count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
        
        return deleted_count
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files"""
        expired_count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    entry = CacheEntry(**data)
                    if entry.is_expired():
                        cache_file.unlink()
                        expired_count += 1
                        
                except Exception:
                    # Remove corrupted cache files
                    cache_file.unlink()
                    expired_count += 1
                    
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
        
        return expired_count


class CacheManager:
    """Main cache manager combining memory and file caching"""
    
    def __init__(self, 
                 use_memory_cache: bool = True,
                 use_file_cache: bool = True,
                 cache_dir: str = ".ado/cache",
                 default_ttl: int = 3600):
        
        self.memory_cache = InMemoryCache(default_ttl) if use_memory_cache else None
        self.file_cache = FileCache(cache_dir, default_ttl) if use_file_cache else None
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then file)"""
        # Try memory cache first
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                return value
        
        # Try file cache
        if self.file_cache:
            value = self.file_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                if self.memory_cache:
                    self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            memory_only: bool = False) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        success = True
        
        # Set in memory cache
        if self.memory_cache:
            self.memory_cache.set(key, value, ttl)
        
        # Set in file cache (unless memory_only)
        if self.file_cache and not memory_only:
            success = self.file_cache.set(key, value, ttl)
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete key from all caches"""
        memory_success = True
        file_success = True
        
        if self.memory_cache:
            memory_success = self.memory_cache.delete(key)
        
        if self.file_cache:
            file_success = self.file_cache.delete(key)
        
        return memory_success and file_success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any cache"""
        if self.memory_cache and self.memory_cache.exists(key):
            return True
        
        if self.file_cache and self.file_cache.exists(key):
            return True
        
        return False
    
    def clear(self) -> Dict[str, int]:
        """Clear all caches"""
        results = {}
        
        if self.memory_cache:
            self.memory_cache.clear()
            results['memory'] = 0  # Memory cache doesn't return count
        
        if self.file_cache:
            results['file'] = self.file_cache.clear()
        
        return results
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries from all caches"""
        results = {}
        
        if self.memory_cache:
            results['memory'] = self.memory_cache.cleanup_expired()
        
        if self.file_cache:
            results['file'] = self.file_cache.cleanup_expired()
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache layers"""
        stats = {}
        
        if self.memory_cache:
            stats['memory'] = self.memory_cache.get_stats()
        
        if self.file_cache:
            cache_files = list(self.file_cache.cache_dir.glob("*.json"))
            stats['file'] = {
                'total_files': len(cache_files),
                'cache_dir_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
            }
        
        return stats
    
    # Convenience methods for common caching patterns
    
    def cache_backlog_metrics(self, metrics: Dict[str, Any], ttl: int = 300):
        """Cache backlog metrics (5 min TTL)"""
        self.set('backlog_metrics', metrics, ttl)
    
    def get_backlog_metrics(self) -> Optional[Dict[str, Any]]:
        """Get cached backlog metrics"""
        return self.get('backlog_metrics')
    
    def cache_wsjf_scores(self, scores: Dict[str, float], ttl: int = 1800):
        """Cache WSJF scores (30 min TTL)"""
        self.set('wsjf_scores', scores, ttl)
    
    def get_wsjf_scores(self) -> Optional[Dict[str, float]]:
        """Get cached WSJF scores"""
        return self.get('wsjf_scores')
    
    def cache_github_data(self, repo: str, data: Any, ttl: int = 600):
        """Cache GitHub API data (10 min TTL)"""
        self.set(f'github_{repo}', data, ttl)
    
    def get_github_data(self, repo: str) -> Optional[Any]:
        """Get cached GitHub data"""
        return self.get(f'github_{repo}')


# Global cache instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(cache_dir: str = ".ado/cache") -> CacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir=cache_dir)
    return _cache_manager


def clear_global_cache():
    """Clear and reset global cache manager"""
    global _cache_manager
    if _cache_manager:
        _cache_manager.clear()
        _cache_manager = None