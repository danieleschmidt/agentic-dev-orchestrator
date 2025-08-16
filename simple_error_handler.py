#!/usr/bin/env python3
"""
Simple Error Handler for Generation 2 Testing
Provides basic error handling capabilities for testing
"""

import time
import logging
from typing import Any, Callable, Dict, Optional
from enum import Enum


class ErrorCategory(Enum):
    """Simple error categories"""
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


class RetryableError(Exception):
    """Exception that can be retried"""
    pass


class ErrorHandler:
    """Simple error handler for testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def categorize_error(self, error: Exception) -> str:
        """Categorize error into basic types"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Network and IO errors are usually transient
        if any(keyword in error_str for keyword in ['network', 'connection', 'timeout']):
            return 'transient'
        elif any(keyword in error_type for keyword in ['connection', 'timeout']):
            return 'transient'
        elif isinstance(error, RetryableError):
            return 'transient'
        # Permission and validation errors are usually permanent
        elif any(keyword in error_str for keyword in ['permission', 'access', 'auth']):
            return 'permanent'
        elif any(keyword in error_type for keyword in ['value', 'type', 'attribute']):
            return 'permanent'
        else:
            return 'unknown'
    
    def execute_with_retry(self, func: Callable, max_retries: int = 3, 
                          delay: float = 1.0) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                error_category = self.categorize_error(e)
                
                # Only retry transient errors
                if error_category != 'transient' or attempt == max_retries - 1:
                    break
                
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Unknown error during retry execution")


# For backward compatibility, create a simple version that tests expect
def main():
    """Demo the simple error handler"""
    handler = ErrorHandler()
    
    # Test error categorization
    test_errors = [
        ValueError("Invalid input"),
        ConnectionError("Network timeout"),
        RetryableError("Temporary failure"),
        PermissionError("Access denied")
    ]
    
    print("🔍 Error Categorization Test:")
    for error in test_errors:
        category = handler.categorize_error(error)
        print(f"  {type(error).__name__}: {error} -> {category}")
    
    # Test retry mechanism
    print("\n🔄 Retry Mechanism Test:")
    
    attempt_count = 0
    def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        print(f"    Attempt {attempt_count}")
        if attempt_count < 3:
            raise RetryableError("Temporary failure")
        return "success"
    
    try:
        result = handler.execute_with_retry(failing_operation, max_retries=3)
        print(f"  Result: {result}")
        print(f"  Total attempts: {attempt_count}")
    except Exception as e:
        print(f"  Failed after retries: {e}")

if __name__ == "__main__":
    main()