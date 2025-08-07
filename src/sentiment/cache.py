"""
Intelligent caching system for sentiment analysis
"""
import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import asdict
from ..cache.cache_manager import get_cache_manager
from .models import SentimentResult

logger = logging.getLogger(__name__)


class SentimentCache:
    """Smart caching for sentiment analysis results"""
    
    def __init__(self, cache_dir: str = ".ado/cache/sentiment", ttl: int = 3600):
        """
        Initialize sentiment cache
        
        Args:
            cache_dir: Directory for cache storage
            ttl: Time to live in seconds (default 1 hour)
        """
        self.cache_manager = get_cache_manager(cache_dir)
        self.default_ttl = ttl
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
        logger.info(f"SentimentCache initialized with TTL={ttl}s")
    
    def _generate_cache_key(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Generate unique cache key for text and metadata"""
        # Create deterministic key from text and metadata
        content = {
            'text': text,
            'metadata': metadata or {}
        }
        content_json = json.dumps(content, sort_keys=True)
        
        # Use SHA256 hash for reliable key generation
        cache_key = hashlib.sha256(content_json.encode('utf-8')).hexdigest()[:16]
        return f"sentiment_{cache_key}"
    
    def get(self, text: str, metadata: Optional[Dict] = None) -> Optional[SentimentResult]:
        """
        Get sentiment result from cache
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            Cached SentimentResult or None if not found
        """
        self.cache_stats['total_requests'] += 1
        
        try:
            cache_key = self._generate_cache_key(text, metadata)
            cached_data = self.cache_manager.get(cache_key)
            
            if cached_data is not None:
                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit for key {cache_key}")
                
                # Reconstruct SentimentResult from cached data
                return self._deserialize_result(cached_data)
            else:
                self.cache_stats['misses'] += 1
                logger.debug(f"Cache miss for key {cache_key}")
                return None
                
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, text: str, result: SentimentResult, ttl: Optional[int] = None) -> bool:
        """
        Store sentiment result in cache
        
        Args:
            text: Input text
            result: SentimentResult to cache
            ttl: Time to live (uses default if None)
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(text, result.metadata)
            cached_data = self._serialize_result(result)
            
            actual_ttl = ttl or self.default_ttl
            success = self.cache_manager.set(cache_key, cached_data, ttl=actual_ttl)
            
            if success:
                logger.debug(f"Cached result for key {cache_key} (TTL={actual_ttl}s)")
            else:
                logger.warning(f"Failed to cache result for key {cache_key}")
                
            return success
            
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False
    
    def _serialize_result(self, result: SentimentResult) -> Dict[str, Any]:
        """Serialize SentimentResult for caching"""
        return {
            'text': result.text,
            'scores': {
                'positive': result.scores.positive,
                'negative': result.scores.negative,
                'neutral': result.scores.neutral,
                'compound': result.scores.compound
            },
            'label': result.label.value,
            'confidence': result.confidence,
            'metadata': result.metadata or {},
            'cached_at': time.time()
        }
    
    def _deserialize_result(self, cached_data: Dict[str, Any]) -> SentimentResult:
        """Deserialize cached data to SentimentResult"""
        from .models import SentimentScore, SentimentLabel
        
        scores = SentimentScore(
            positive=cached_data['scores']['positive'],
            negative=cached_data['scores']['negative'],
            neutral=cached_data['scores']['neutral'],
            compound=cached_data['scores']['compound']
        )
        
        label = SentimentLabel(cached_data['label'])
        
        return SentimentResult(
            text=cached_data['text'],
            scores=scores,
            label=label,
            confidence=cached_data['confidence'],
            metadata=cached_data.get('metadata', {})
        )
    
    def clear_expired(self) -> int:
        """Clear expired cache entries"""
        try:
            cleared_count = self.cache_manager.clear_expired()
            logger.info(f"Cleared {cleared_count} expired cache entries")
            return cleared_count
        except Exception as e:
            logger.warning(f"Failed to clear expired entries: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total = self.cache_stats['total_requests']
        hits = self.cache_stats['hits']
        
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        return {
            'total_requests': total,
            'cache_hits': hits,
            'cache_misses': self.cache_stats['misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': getattr(self.cache_manager, 'size', lambda: 0)(),
            'cache_efficiency': 'excellent' if hit_rate > 80 else 'good' if hit_rate > 50 else 'poor'
        }
    
    def clear_all(self) -> bool:
        """Clear all cached sentiment results"""
        try:
            result = self.cache_manager.clear()
            self.cache_stats = {'hits': 0, 'misses': 0, 'total_requests': 0}
            logger.info("Cleared all sentiment cache")
            return result.get('success', False)
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return False