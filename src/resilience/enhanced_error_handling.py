#!/usr/bin/env python3
"""
Enhanced Error Handling and Recovery System
Comprehensive error handling, validation, and recovery mechanisms for robust operation
"""

import json
import logging
import traceback
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from contextlib import contextmanager, asynccontextmanager
import inspect
import threading
from pathlib import Path
import hashlib
import signal
import sys
from abc import ABC, abstractmethod


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    CIRCUIT_BREAK = "circuit_break"


class ErrorCategory(Enum):
    """Categories of errors for classification"""
    NETWORK = "network"
    IO = "io"
    VALIDATION = "validation"
    SECURITY = "security"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    DATA_CORRUPTION = "data_corruption"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Rich error context for analysis and recovery"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    exception_type: str
    message: str
    traceback_str: str
    function_name: str
    module_name: str
    line_number: int
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    
    @property
    def can_retry(self) -> bool:
        """Check if error can be retried"""
        return (self.recovery_attempts < self.max_recovery_attempts and 
                self.recovery_strategy in [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK])


@dataclass
class RecoveryResult:
    """Result of error recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    attempts_made: int
    recovery_time: float
    final_result: Any = None
    error_context: Optional[ErrorContext] = None


class ErrorHandler(ABC):
    """Abstract base class for error handlers"""
    
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this handler can process the error"""
        pass
    
    @abstractmethod
    async def handle(self, error_context: ErrorContext, original_func: Callable, 
                    *args, **kwargs) -> RecoveryResult:
        """Handle the error and attempt recovery"""
        pass


class RetryErrorHandler(ErrorHandler):
    """Handler for retryable errors with exponential backoff"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 backoff_multiplier: float = 2.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if error is retryable"""
        return (error_context.recovery_strategy == RecoveryStrategy.RETRY and
                error_context.category in [ErrorCategory.NETWORK, ErrorCategory.IO, 
                                         ErrorCategory.EXTERNAL_SERVICE, ErrorCategory.TIMEOUT])
    
    async def handle(self, error_context: ErrorContext, original_func: Callable, 
                    *args, **kwargs) -> RecoveryResult:
        """Handle error with retry logic"""
        start_time = time.time()
        
        for attempt in range(self.max_attempts):
            if attempt > 0:
                delay = min(self.base_delay * (self.backoff_multiplier ** (attempt - 1)), 
                          self.max_delay)
                await asyncio.sleep(delay)
            
            try:
                if asyncio.iscoroutinefunction(original_func):
                    result = await original_func(*args, **kwargs)
                else:
                    result = original_func(*args, **kwargs)
                
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RETRY,
                    attempts_made=attempt + 1,
                    recovery_time=time.time() - start_time,
                    final_result=result
                )
                
            except Exception as e:
                error_context.recovery_attempts = attempt + 1
                if attempt == self.max_attempts - 1:
                    # Final attempt failed
                    return RecoveryResult(
                        success=False,
                        strategy_used=RecoveryStrategy.RETRY,
                        attempts_made=attempt + 1,
                        recovery_time=time.time() - start_time,
                        error_context=error_context
                    )


class FallbackErrorHandler(ErrorHandler):
    """Handler that provides fallback functionality"""
    
    def __init__(self, fallback_functions: Dict[str, Callable] = None):
        self.fallback_functions = fallback_functions or {}
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if fallback is available"""
        return (error_context.recovery_strategy == RecoveryStrategy.FALLBACK and
                error_context.function_name in self.fallback_functions)
    
    async def handle(self, error_context: ErrorContext, original_func: Callable, 
                    *args, **kwargs) -> RecoveryResult:
        """Handle error with fallback function"""
        start_time = time.time()
        
        fallback_func = self.fallback_functions.get(error_context.function_name)
        if not fallback_func:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK,
                attempts_made=1,
                recovery_time=time.time() - start_time,
                error_context=error_context
            )
        
        try:
            if asyncio.iscoroutinefunction(fallback_func):
                result = await fallback_func(*args, **kwargs)
            else:
                result = fallback_func(*args, **kwargs)
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK,
                attempts_made=1,
                recovery_time=time.time() - start_time,
                final_result=result
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK,
                attempts_made=1,
                recovery_time=time.time() - start_time,
                error_context=error_context
            )


class CircuitBreakerErrorHandler(ErrorHandler):
    """Handler with circuit breaker pattern"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts: Dict[str, int] = {}
        self.circuit_open_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if circuit breaker should be applied"""
        return error_context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAK
    
    def _is_circuit_open(self, function_name: str) -> bool:
        """Check if circuit is open for function"""
        with self.lock:
            if function_name not in self.circuit_open_times:
                return False
            
            open_time = self.circuit_open_times[function_name]
            if datetime.now() - open_time > timedelta(seconds=self.recovery_timeout):
                # Circuit should be half-open, remove from tracking
                del self.circuit_open_times[function_name]
                self.failure_counts.pop(function_name, 0)
                return False
            
            return True
    
    def _record_failure(self, function_name: str):
        """Record failure and potentially open circuit"""
        with self.lock:
            self.failure_counts[function_name] = self.failure_counts.get(function_name, 0) + 1
            
            if self.failure_counts[function_name] >= self.failure_threshold:
                self.circuit_open_times[function_name] = datetime.now()
    
    def _record_success(self, function_name: str):
        """Record success and reset failure count"""
        with self.lock:
            self.failure_counts.pop(function_name, 0)
    
    async def handle(self, error_context: ErrorContext, original_func: Callable, 
                    *args, **kwargs) -> RecoveryResult:
        """Handle error with circuit breaker logic"""
        start_time = time.time()
        function_name = error_context.function_name
        
        # Check if circuit is open
        if self._is_circuit_open(function_name):
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAK,
                attempts_made=0,
                recovery_time=time.time() - start_time,
                error_context=error_context
            )
        
        try:
            if asyncio.iscoroutinefunction(original_func):
                result = await original_func(*args, **kwargs)
            else:
                result = original_func(*args, **kwargs)
            
            self._record_success(function_name)
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAK,
                attempts_made=1,
                recovery_time=time.time() - start_time,
                final_result=result
            )
            
        except Exception as e:
            self._record_failure(function_name)
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.CIRCUIT_BREAK,
                attempts_made=1,
                recovery_time=time.time() - start_time,
                error_context=error_context
            )


class EnhancedErrorHandling:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.handlers: List[ErrorHandler] = []
        self.error_history: List[ErrorContext] = []
        self.metrics: Dict[str, Any] = {
            "total_errors": 0,
            "total_recoveries": 0,
            "recovery_success_rate": 0.0,
            "error_categories": {},
            "function_error_rates": {}
        }
        
        # Initialize built-in handlers
        self._initialize_handlers()
        
        # Setup logging
        self.logger = logging.getLogger("enhanced_error_handling")
        
        # Error reporting
        self.error_report_path = Path(".terragon/error-reports")
        self.error_report_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load error handling configuration"""
        # Default configuration
        default_config = {
            "error_handling": {
                "enabled": True,
                "max_retry_attempts": 3,
                "retry_backoff_multiplier": 2.0,
                "circuit_breaker_threshold": 5,
                "circuit_breaker_timeout": 60.0,
                "fallback_enabled": True,
                "error_reporting": True,
                "metrics_collection": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if "error_handling" in loaded_config:
                        default_config["error_handling"].update(loaded_config["error_handling"])
            except Exception as e:
                self.logger.warning(f"Could not load error handling config: {e}")
        
        return default_config
    
    def _initialize_handlers(self):
        """Initialize error handlers based on configuration"""
        config = self.config.get("error_handling", {})
        
        # Retry handler
        self.handlers.append(RetryErrorHandler(
            max_attempts=config.get("max_retry_attempts", 3),
            backoff_multiplier=config.get("retry_backoff_multiplier", 2.0)
        ))
        
        # Fallback handler
        if config.get("fallback_enabled", True):
            self.handlers.append(FallbackErrorHandler())
        
        # Circuit breaker handler
        self.handlers.append(CircuitBreakerErrorHandler(
            failure_threshold=config.get("circuit_breaker_threshold", 5),
            recovery_timeout=config.get("circuit_breaker_timeout", 60.0)
        ))
    
    def add_handler(self, handler: ErrorHandler):
        """Add custom error handler"""
        self.handlers.append(handler)
    
    def add_fallback_function(self, original_function_name: str, fallback_function: Callable):
        """Add fallback function for specific function"""
        for handler in self.handlers:
            if isinstance(handler, FallbackErrorHandler):
                handler.fallback_functions[original_function_name] = fallback_function
                break
    
    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify error into category"""
        exception_type = type(exception).__name__.lower()
        
        if any(keyword in str(exception).lower() for keyword in ['network', 'connection', 'dns']):
            return ErrorCategory.NETWORK
        elif any(keyword in exception_type for keyword in ['io', 'file', 'permission']):
            return ErrorCategory.IO
        elif any(keyword in exception_type for keyword in ['validation', 'value', 'type']):
            return ErrorCategory.VALIDATION
        elif any(keyword in exception_type for keyword in ['security', 'permission', 'auth']):
            return ErrorCategory.SECURITY
        elif any(keyword in exception_type for keyword in ['timeout', 'time']):
            return ErrorCategory.TIMEOUT
        elif any(keyword in exception_type for keyword in ['memory', 'resource']):
            return ErrorCategory.RESOURCE
        elif any(keyword in str(exception).lower() for keyword in ['config', 'setting']):
            return ErrorCategory.CONFIGURATION
        elif any(keyword in exception_type for keyword in ['corruption', 'corrupt']):
            return ErrorCategory.DATA_CORRUPTION
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_recovery_strategy(self, error_category: ErrorCategory, 
                                   severity: ErrorSeverity) -> RecoveryStrategy:
        """Determine recovery strategy based on error characteristics"""
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE
        
        if error_category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.EXTERNAL_SERVICE]:
            return RecoveryStrategy.RETRY
        elif error_category in [ErrorCategory.IO, ErrorCategory.RESOURCE]:
            return RecoveryStrategy.FALLBACK
        elif error_category == ErrorCategory.VALIDATION:
            return RecoveryStrategy.IGNORE  # Let validation errors bubble up
        elif error_category == ErrorCategory.SECURITY:
            return RecoveryStrategy.ESCALATE
        else:
            return RecoveryStrategy.RETRY
    
    def _create_error_context(self, exception: Exception, frame_info) -> ErrorContext:
        """Create rich error context"""
        error_id = hashlib.sha256(
            f"{type(exception).__name__}{str(exception)}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        category = self._classify_error(exception)
        severity = self._determine_severity(exception, category)
        recovery_strategy = self._determine_recovery_strategy(category, severity)
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            exception_type=type(exception).__name__,
            message=str(exception),
            traceback_str=traceback.format_exc(),
            function_name=frame_info.function if frame_info else "unknown",
            module_name=frame_info.filename if frame_info else "unknown",
            line_number=frame_info.lineno if frame_info else 0,
            recovery_strategy=recovery_strategy
        )
    
    def _determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        if category in [ErrorCategory.SECURITY, ErrorCategory.DATA_CORRUPTION]:
            return ErrorSeverity.CRITICAL
        elif category in [ErrorCategory.RESOURCE, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_SERVICE]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _update_metrics(self, error_context: ErrorContext, recovery_result: Optional[RecoveryResult]):
        """Update error handling metrics"""
        self.metrics["total_errors"] += 1
        
        # Update category counts
        category = error_context.category.value
        self.metrics["error_categories"][category] = self.metrics["error_categories"].get(category, 0) + 1
        
        # Update function error rates
        func_name = error_context.function_name
        if func_name not in self.metrics["function_error_rates"]:
            self.metrics["function_error_rates"][func_name] = {"errors": 0, "recoveries": 0}
        
        self.metrics["function_error_rates"][func_name]["errors"] += 1
        
        if recovery_result and recovery_result.success:
            self.metrics["total_recoveries"] += 1
            self.metrics["function_error_rates"][func_name]["recoveries"] += 1
        
        # Update recovery success rate
        if self.metrics["total_errors"] > 0:
            self.metrics["recovery_success_rate"] = self.metrics["total_recoveries"] / self.metrics["total_errors"]
    
    def _save_error_report(self, error_context: ErrorContext, recovery_result: Optional[RecoveryResult]):
        """Save detailed error report"""
        if not self.config.get("error_handling", {}).get("error_reporting", True):
            return
        
        report = {
            "error_id": error_context.error_id,
            "timestamp": error_context.timestamp.isoformat(),
            "severity": error_context.severity.value,
            "category": error_context.category.value,
            "exception_type": error_context.exception_type,
            "message": error_context.message,
            "function_name": error_context.function_name,
            "module_name": error_context.module_name,
            "line_number": error_context.line_number,
            "traceback": error_context.traceback_str,
            "context_data": error_context.context_data,
            "recovery_attempts": error_context.recovery_attempts,
            "recovery_strategy": error_context.recovery_strategy.value,
            "recovery_result": {
                "success": recovery_result.success if recovery_result else False,
                "strategy_used": recovery_result.strategy_used.value if recovery_result else None,
                "attempts_made": recovery_result.attempts_made if recovery_result else 0,
                "recovery_time": recovery_result.recovery_time if recovery_result else 0.0
            } if recovery_result else None
        }
        
        report_file = self.error_report_path / f"error-{error_context.error_id}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save error report: {e}")
    
    async def handle_error(self, exception: Exception, original_func: Callable, 
                          *args, **kwargs) -> RecoveryResult:
        """Main error handling entry point"""
        # Get frame information
        frame_info = inspect.currentframe().f_back if inspect.currentframe() else None
        
        # Create error context
        error_context = self._create_error_context(exception, frame_info)
        
        # Add to error history
        self.error_history.append(error_context)
        
        # Find appropriate handler
        handler = None
        for h in self.handlers:
            if h.can_handle(error_context):
                handler = h
                break
        
        recovery_result = None
        
        if handler:
            self.logger.info(f"Handling error {error_context.error_id} with {type(handler).__name__}")
            recovery_result = await handler.handle(error_context, original_func, *args, **kwargs)
        else:
            self.logger.warning(f"No handler found for error {error_context.error_id}")
            recovery_result = RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.ESCALATE,
                attempts_made=0,
                recovery_time=0.0,
                error_context=error_context
            )
        
        # Update metrics and reporting
        self._update_metrics(error_context, recovery_result)
        self._save_error_report(error_context, recovery_result)
        
        return recovery_result
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get comprehensive error metrics"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_errors": self.metrics["total_errors"],
            "total_recoveries": self.metrics["total_recoveries"],
            "recovery_success_rate": self.metrics["recovery_success_rate"],
            "error_categories": dict(self.metrics["error_categories"]),
            "function_error_rates": dict(self.metrics["function_error_rates"]),
            "recent_errors": len([e for e in self.error_history if 
                                (datetime.now() - e.timestamp).total_seconds() < 3600]),  # Last hour
            "critical_errors": len([e for e in self.error_history if 
                                  e.severity == ErrorSeverity.CRITICAL])
        }
    
    def export_error_analysis(self, output_path: str = "error-analysis-report.md"):
        """Export comprehensive error analysis report"""
        metrics = self.get_error_metrics()
        
        with open(output_path, 'w') as f:
            f.write(f"""# 🛡️ Error Handling Analysis Report

Generated: {metrics['timestamp']}
Total Errors Handled: {metrics['total_errors']}
Recovery Success Rate: {metrics['recovery_success_rate']:.1%}

## 📊 Error Categories

""")
            
            for category, count in metrics['error_categories'].items():
                percentage = (count / max(metrics['total_errors'], 1)) * 100
                f.write(f"- **{category.title()}**: {count} errors ({percentage:.1f}%)\n")
            
            f.write(f"""
## 🔧 Function Error Rates

""")
            
            for func_name, stats in metrics['function_error_rates'].items():
                recovery_rate = stats['recoveries'] / max(stats['errors'], 1) * 100
                f.write(f"- **{func_name}**: {stats['errors']} errors, {recovery_rate:.1f}% recovery rate\n")
            
            f.write(f"""
## 🚨 Recent Activity

- Errors in last hour: {metrics['recent_errors']}
- Critical errors total: {metrics['critical_errors']}

## 💡 Recommendations

""")
            
            if metrics['recovery_success_rate'] < 0.8:
                f.write("- Consider reviewing error handling strategies for better recovery rates\n")
            
            if metrics['critical_errors'] > 0:
                f.write("- Investigate and address critical errors immediately\n")
            
            if metrics['recent_errors'] > 10:
                f.write("- High error rate detected, review system stability\n")
            
            f.write("\n---\n*Generated by Enhanced Error Handling System*\n")


# Decorator for automatic error handling
def with_error_handling(error_handler: Optional[EnhancedErrorHandling] = None,
                       fallback_function: Optional[Callable] = None,
                       max_retries: int = 3,
                       ignore_errors: bool = False):
    """
    Decorator to automatically apply error handling to functions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal error_handler
            if error_handler is None:
                error_handler = EnhancedErrorHandling()
            
            if fallback_function:
                error_handler.add_fallback_function(func.__name__, fallback_function)
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                if ignore_errors:
                    return None
                
                recovery_result = await error_handler.handle_error(e, func, *args, **kwargs)
                
                if recovery_result.success:
                    return recovery_result.final_result
                else:
                    # Re-raise if recovery failed
                    raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global error handler instance
global_error_handler = EnhancedErrorHandling()


def main():
    """Demonstration of enhanced error handling"""
    import random
    
    print("🛡️ Enhanced Error Handling System Demo")
    print("=" * 50)
    
    # Create error handler
    error_handler = EnhancedErrorHandling()
    
    # Demo functions that might fail
    @with_error_handling(error_handler, max_retries=2)
    async def flaky_network_call():
        """Simulate flaky network operation"""
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Network timeout")
        return {"status": "success", "data": "network_data"}
    
    @with_error_handling(error_handler)
    async def file_operation():
        """Simulate file operation that might fail"""
        if random.random() < 0.3:  # 30% failure rate
            raise IOError("File not found")
        return "file_content"
    
    @with_error_handling(error_handler)
    async def validation_check(value: str):
        """Simulate validation that might fail"""
        if not value or len(value) < 3:
            raise ValueError("Value too short")
        return f"validated_{value}"
    
    async def demo():
        print("\n🧪 Running error handling demo...")
        
        # Test network calls
        print("\n1. Testing network calls (high failure rate):")
        for i in range(5):
            try:
                result = await flaky_network_call()
                print(f"   Call {i+1}: ✅ Success - {result}")
            except Exception as e:
                print(f"   Call {i+1}: ❌ Failed - {str(e)}")
        
        # Test file operations
        print("\n2. Testing file operations:")
        for i in range(3):
            try:
                result = await file_operation()
                print(f"   Operation {i+1}: ✅ Success")
            except Exception as e:
                print(f"   Operation {i+1}: ❌ Failed - {str(e)}")
        
        # Test validation
        print("\n3. Testing validation:")
        test_values = ["ok", "x", "valid_value", ""]
        for value in test_values:
            try:
                result = await validation_check(value)
                print(f"   '{value}': ✅ Valid - {result}")
            except Exception as e:
                print(f"   '{value}': ❌ Invalid - {str(e)}")
        
        # Show metrics
        print("\n📊 Error Handling Metrics:")
        metrics = error_handler.get_error_metrics()
        print(f"   Total Errors: {metrics['total_errors']}")
        print(f"   Total Recoveries: {metrics['total_recoveries']}")
        print(f"   Recovery Success Rate: {metrics['recovery_success_rate']:.1%}")
        print(f"   Recent Errors (last hour): {metrics['recent_errors']}")
        
        print("\n📊 Error Categories:")
        for category, count in metrics['error_categories'].items():
            print(f"   {category.title()}: {count}")
        
        # Export analysis report
        error_handler.export_error_analysis("demo-error-analysis.md")
        print("\n📄 Error analysis report saved to: demo-error-analysis.md")
    
    # Run demo
    asyncio.run(demo())


if __name__ == "__main__":
    main()