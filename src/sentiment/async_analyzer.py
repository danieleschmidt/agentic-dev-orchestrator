"""
Asynchronous sentiment analysis with concurrent processing
"""
import asyncio
import logging
from typing import List, Dict, Optional, Callable, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from .analyzer import SentimentAnalyzer
from .models import SentimentResult
from .cache import SentimentCache
from .exceptions import SentimentAnalysisError

logger = logging.getLogger(__name__)


class AsyncSentimentAnalyzer:
    """Asynchronous sentiment analyzer with concurrent processing"""
    
    def __init__(self, 
                 analyzer: Optional[SentimentAnalyzer] = None,
                 cache: Optional[SentimentCache] = None,
                 max_workers: int = 4,
                 batch_size: int = 10):
        """
        Initialize async analyzer
        
        Args:
            analyzer: Base sentiment analyzer (creates new if None)
            cache: Cache instance (creates new if None)
            max_workers: Maximum worker threads for concurrent processing
            batch_size: Optimal batch size for processing
        """
        self.analyzer = analyzer or SentimentAnalyzer()
        self.cache = cache or SentimentCache()
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"AsyncSentimentAnalyzer initialized with {max_workers} workers")
    
    async def analyze_async(self, text: str, metadata: Optional[Dict] = None) -> SentimentResult:
        """
        Analyze single text asynchronously with caching
        
        Args:
            text: Input text to analyze
            metadata: Optional metadata
            
        Returns:
            SentimentResult
        """
        # Check cache first
        cached_result = self.cache.get(text, metadata)
        if cached_result is not None:
            logger.debug("Returning cached sentiment result")
            return cached_result
        
        # Run analysis in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self.analyzer.analyze, 
            text, 
            metadata
        )
        
        # Cache the result
        self.cache.set(text, result)
        
        return result
    
    async def analyze_batch_async(self, texts: List[str], 
                                progress_callback: Optional[Callable[[int, int], None]] = None) -> List[SentimentResult]:
        """
        Analyze multiple texts concurrently with progress tracking
        
        Args:
            texts: List of texts to analyze
            progress_callback: Optional callback for progress updates (completed, total)
            
        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []
        
        logger.info(f"Starting async batch analysis of {len(texts)} texts")
        
        # Check cache for all texts first
        results = []
        pending_texts = []
        pending_indices = []
        
        for i, text in enumerate(texts):
            cached_result = self.cache.get(text, {'batch_index': i})
            if cached_result is not None:
                results.append((i, cached_result))
            else:
                pending_texts.append(text)
                pending_indices.append(i)
        
        cache_hits = len(results)
        logger.info(f"Cache hits: {cache_hits}/{len(texts)}")
        
        # Process remaining texts concurrently
        if pending_texts:
            # Split into optimal batch sizes
            batches = [pending_texts[i:i+self.batch_size] 
                      for i in range(0, len(pending_texts), self.batch_size)]
            
            batch_indices = [pending_indices[i:i+self.batch_size] 
                           for i in range(0, len(pending_indices), self.batch_size)]
            
            # Process batches concurrently
            futures = []
            for batch, indices in zip(batches, batch_indices):
                future = asyncio.create_task(
                    self._process_batch_async(batch, indices)
                )
                futures.append(future)
            
            # Wait for all batches with progress tracking
            completed_count = cache_hits
            for future in asyncio.as_completed(futures):
                batch_results = await future
                results.extend(batch_results)
                completed_count += len(batch_results)
                
                if progress_callback:
                    progress_callback(completed_count, len(texts))
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        final_results = [result for _, result in results]
        
        logger.info(f"Async batch analysis completed: {len(final_results)} results")
        return final_results
    
    async def _process_batch_async(self, texts: List[str], indices: List[int]) -> List[tuple]:
        """Process a batch of texts asynchronously"""
        tasks = []
        
        for text, index in zip(texts, indices):
            task = asyncio.create_task(
                self.analyze_async(text, {'batch_index': index})
            )
            tasks.append((index, task))
        
        results = []
        for index, task in tasks:
            try:
                result = await task
                results.append((index, result))
            except Exception as e:
                logger.warning(f"Failed to analyze text at index {index}: {e}")
                # Return neutral result for failed analysis
                from .models import SentimentScore, SentimentLabel
                neutral_result = SentimentResult(
                    text=texts[indices.index(index)] if index in indices else "",
                    scores=SentimentScore(0.0, 0.0, 1.0, 0.0),
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.0,
                    metadata={'batch_index': index, 'error': str(e)}
                )
                results.append((index, neutral_result))
        
        return results
    
    def analyze_stream(self, texts: List[str]) -> AsyncGenerator[SentimentResult, None]:
        """
        Stream sentiment analysis results as they become available
        
        Args:
            texts: List of texts to analyze
            
        Yields:
            SentimentResult objects as they complete
        """
        async def _stream_generator():
            if not texts:
                return
            
            # Submit all tasks
            tasks = []
            for i, text in enumerate(texts):
                task = asyncio.create_task(
                    self.analyze_async(text, {'stream_index': i})
                )
                tasks.append(task)
            
            # Yield results as they complete
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    yield result
                except Exception as e:
                    logger.warning(f"Stream analysis failed: {e}")
                    # Yield neutral result for failed analysis
                    from .models import SentimentScore, SentimentLabel
                    yield SentimentResult(
                        text="",
                        scores=SentimentScore(0.0, 0.0, 1.0, 0.0),
                        label=SentimentLabel.NEUTRAL,
                        confidence=0.0,
                        metadata={'error': str(e)}
                    )
        
        return _stream_generator()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        cache_stats = self.cache.get_stats()
        
        return {
            'cache_stats': cache_stats,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'executor_status': 'active' if not self.executor._shutdown else 'shutdown'
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("AsyncSentimentAnalyzer cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()