#!/usr/bin/env python3
"""
Advanced Retry Mechanism with Exponential Backoff and Jitter
Implements resilient retry patterns for ADO operations
"""

import time
import random
import logging
import functools
from typing import Callable, Any, Optional, Union, List, Type
from dataclasses import dataclass
from datetime import datetime
import threading
from enum import Enum


class BackoffStrategy(Enum):
    """Backoff strategies for retry mechanisms"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    delay: float
    exception: Optional[Exception]
    timestamp: datetime
    total_elapsed: float


@dataclass
class RetryResult:
    """Result of retry operation"""
    success: bool
    result: Any = None
    attempts: List[RetryAttempt] = None
    total_attempts: int = 0
    total_time: float = 0.0
    final_exception: Optional[Exception] = None


class RetryExhausted(Exception):
    """Exception raised when all retry attempts are exhausted"""
    def __init__(self, message: str, retry_result: RetryResult):
        super().__init__(message)
        self.retry_result = retry_result


class AdvancedRetry:
    """Advanced retry mechanism with multiple backoff strategies"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger("advanced_retry")
        self._fibonacci_cache = [1, 1]  # For fibonacci backoff
        
    def execute(self, func: Callable, *args, **kwargs) -> RetryResult:
        """Execute function with retry logic"""
        attempts = []
        start_time = time.time()
        
        for attempt in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Success!
                total_time = time.time() - start_time
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_attempts=attempt,
                    total_time=total_time
                )
                
            except Exception as e:
                attempt_end = time.time()
                
                # Check if exception is non-retryable
                if self._is_non_retryable_exception(e):
                    self.logger.error(f"Non-retryable exception: {e}")
                    return RetryResult(
                        success=False,
                        attempts=attempts,
                        total_attempts=attempt,
                        total_time=time.time() - start_time,
                        final_exception=e
                    )
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    self.logger.error(f"Non-configured retryable exception: {e}")
                    return RetryResult(
                        success=False,
                        attempts=attempts,
                        total_attempts=attempt,
                        total_time=time.time() - start_time,
                        final_exception=e
                    )
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt) if attempt < self.config.max_attempts else 0
                
                attempt_info = RetryAttempt(
                    attempt_number=attempt,
                    delay=delay,
                    exception=e,
                    timestamp=datetime.now(),
                    total_elapsed=time.time() - start_time
                )
                attempts.append(attempt_info)
                
                self.logger.warning(
                    f"Attempt {attempt}/{self.config.max_attempts} failed: {e}"
                    f"{f'. Retrying in {delay:.2f}s' if delay > 0 else ''}"
                )
                
                # If this was the last attempt, don't sleep
                if attempt < self.config.max_attempts and delay > 0:
                    time.sleep(delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        final_exception = attempts[-1].exception if attempts else None
        
        return RetryResult(
            success=False,
            attempts=attempts,
            total_attempts=self.config.max_attempts,
            total_time=total_time,
            final_exception=final_exception
        )
        
    def _calculate_delay(self, attempt_number: int) -> float:
        """Calculate delay for the given attempt number"""
        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.config.initial_delay
            
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.initial_delay * attempt_number
            
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.exponential_base ** (attempt_number - 1))
            
        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            delay = self.config.initial_delay * self._get_fibonacci(attempt_number)
            
        else:
            delay = self.config.initial_delay
            
        # Apply max delay constraint
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure minimum delay
            
        return delay
        
    def _get_fibonacci(self, n: int) -> int:
        """Get nth fibonacci number (cached)"""
        while len(self._fibonacci_cache) <= n:
            next_fib = self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            self._fibonacci_cache.append(next_fib)
        return self._fibonacci_cache[n]
        
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return isinstance(exception, self.config.retryable_exceptions)
        
    def _is_non_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception is explicitly non-retryable"""
        return isinstance(exception, self.config.non_retryable_exceptions)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    retryable_exceptions: tuple = (Exception,),
    non_retryable_exceptions: tuple = (),
    raise_on_failure: bool = True
):
    """Decorator for applying retry logic to functions"""
    def decorator(func: Callable) -> Callable:
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            backoff_strategy=backoff_strategy,
            retryable_exceptions=retryable_exceptions,
            non_retryable_exceptions=non_retryable_exceptions
        )
        
        retry_executor = AdvancedRetry(config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = retry_executor.execute(func, *args, **kwargs)
            
            if result.success:
                return result.result
            elif raise_on_failure:
                raise RetryExhausted(
                    f"Function {func.__name__} failed after {result.total_attempts} attempts",
                    result
                )
            else:
                return result
                
        # Attach retry executor to function for external access
        wrapper.retry_executor = retry_executor
        return wrapper
        
    return decorator


class RetryableOperation:
    """Base class for retryable operations with built-in retry logic"""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.retry_executor = AdvancedRetry(self.retry_config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def execute(self, *args, **kwargs) -> Any:
        """Execute the operation with retry logic"""
        result = self.retry_executor.execute(self._perform_operation, *args, **kwargs)
        
        if not result.success:
            self._handle_failure(result)
            
        return result.result if result.success else None
        
    def _perform_operation(self, *args, **kwargs) -> Any:
        """Override this method to implement the actual operation"""
        raise NotImplementedError("Subclasses must implement _perform_operation")
        
    def _handle_failure(self, result: RetryResult) -> None:
        """Handle operation failure - override for custom behavior"""
        self.logger.error(
            f"Operation failed after {result.total_attempts} attempts. "
            f"Total time: {result.total_time:.2f}s. "
            f"Final error: {result.final_exception}"
        )


class NetworkOperation(RetryableOperation):
    """Example retryable network operation"""
    
    def __init__(self, url: str, timeout: int = 30):
        # Configure for network-specific retries
        retry_config = RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=30.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
            ),
            non_retryable_exceptions=(
                ValueError,  # Bad URL format
                PermissionError,  # Authentication issues
            )
        )
        super().__init__(retry_config)
        self.url = url
        self.timeout = timeout
        
    def _perform_operation(self) -> str:
        """Perform HTTP request (mock implementation)"""
        import urllib.request
        import urllib.error
        
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
                return response.read().decode()
        except urllib.error.URLError as e:
            if isinstance(e.reason, ConnectionError):
                raise ConnectionError(f"Failed to connect to {self.url}")
            raise


class DatabaseOperation(RetryableOperation):
    """Example retryable database operation"""
    
    def __init__(self, query: str):
        # Configure for database-specific retries
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=10.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                # Add database-specific exceptions as needed
            ),
            non_retryable_exceptions=(
                ValueError,  # Bad query
                PermissionError,  # Access denied
            )
        )
        super().__init__(retry_config)
        self.query = query
        
    def _perform_operation(self) -> Any:
        """Execute database query (mock implementation)"""
        # This would integrate with actual database connections
        if "invalid" in self.query.lower():
            raise ValueError("Invalid query")
        
        # Simulate occasional connection issues
        if random.random() < 0.2:  # 20% chance of connection error
            raise ConnectionError("Database connection lost")
            
        return f"Query result for: {self.query}"


def main():
    """CLI entry point for testing retry mechanisms"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python retry_with_backoff.py <command> [args]")
        print("Commands:")
        print("  test-network <url> - Test network retry")
        print("  test-db <query> - Test database retry")
        print("  demo - Run demo scenarios")
        return
        
    command = sys.argv[1]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if command == "test-network" and len(sys.argv) > 2:
        url = sys.argv[2]
        operation = NetworkOperation(url)
        try:
            result = operation.execute()
            print(f"Success: Retrieved {len(result)} characters")
        except Exception as e:
            print(f"Failed: {e}")
            
    elif command == "test-db" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        operation = DatabaseOperation(query)
        try:
            result = operation.execute()
            print(f"Success: {result}")
        except Exception as e:
            print(f"Failed: {e}")
            
    elif command == "demo":
        print("Running retry mechanism demos...")
        
        # Demo 1: Function decorator
        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def flaky_function():
            if random.random() < 0.7:
                raise ConnectionError("Random connection error")
            return "Success!"
            
        try:
            result = flaky_function()
            print(f"Demo 1 result: {result}")
        except RetryExhausted as e:
            print(f"Demo 1 failed: {e}")
            
        # Demo 2: Database operation
        db_op = DatabaseOperation("SELECT * FROM users")
        result = db_op.execute()
        print(f"Demo 2 result: {result}")
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
