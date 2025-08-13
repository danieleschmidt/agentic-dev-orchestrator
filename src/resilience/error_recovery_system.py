#!/usr/bin/env python3
"""
Advanced Error Recovery and Resilience System
Implements comprehensive error handling, recovery strategies, and system resilience
"""

import time
import json
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Type
import threading
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ESCALATE = "escalate"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    operation: str
    context_data: Dict[str, Any]
    stack_trace: str
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None


@dataclass 
class RecoveryResult:
    """Result of recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    error_resolved: bool
    fallback_value: Optional[Any] = None
    recovery_time_seconds: float = 0.0
    additional_context: Dict[str, Any] = None


class ErrorRecoveryHandler(ABC):
    """Abstract base for error recovery handlers"""
    
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this handler can recover from the error"""
        pass
    
    @abstractmethod
    def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt to recover from the error"""
        pass


class RetryRecoveryHandler(ErrorRecoveryHandler):
    """Handles recovery through retry with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Can handle transient errors that haven't exceeded retry limit"""
        transient_errors = [
            "ConnectionError",
            "TimeoutError", 
            "TemporaryFailure",
            "RateLimitError",
            "ServiceUnavailable"
        ]
        
        return (error_context.recovery_attempts < self.max_retries and
                any(err in error_context.error_type for err in transient_errors))
    
    def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement exponential backoff retry"""
        start_time = time.time()
        
        # Calculate delay
        attempt = error_context.recovery_attempts
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        logger.info(f"Retry recovery: attempt {attempt + 1}, delay {delay}s")
        time.sleep(delay)
        
        recovery_time = time.time() - start_time
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY,
            error_resolved=False,  # Will be determined by re-execution
            recovery_time_seconds=recovery_time,
            additional_context={"retry_attempt": attempt + 1, "delay_used": delay}
        )


class CircuitBreakerRecoveryHandler(ErrorRecoveryHandler):
    """Circuit breaker pattern for failing operations"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts: Dict[str, int] = {}
        self.circuit_open_times: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Can handle repeated failures to the same operation"""
        operation_key = f"{error_context.component}.{error_context.operation}"
        
        with self._lock:
            failure_count = self.failure_counts.get(operation_key, 0)
            return failure_count >= self.failure_threshold
    
    def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Open circuit breaker and provide fallback"""
        start_time = time.time()
        operation_key = f"{error_context.component}.{error_context.operation}"
        
        with self._lock:
            self.circuit_open_times[operation_key] = datetime.now()
            logger.warning(f"Circuit breaker OPEN for {operation_key}")
        
        recovery_time = time.time() - start_time
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
            error_resolved=True,
            fallback_value=self._get_fallback_value(error_context),
            recovery_time_seconds=recovery_time,
            additional_context={"circuit_state": "OPEN", "operation": operation_key}
        )
    
    def _get_fallback_value(self, error_context: ErrorContext) -> Any:
        """Provide appropriate fallback value based on context"""
        if "list" in error_context.operation.lower():
            return []
        elif "dict" in error_context.operation.lower():
            return {}
        elif "count" in error_context.operation.lower():
            return 0
        elif "status" in error_context.operation.lower():
            return "UNAVAILABLE"
        else:
            return None
    
    def is_circuit_closed(self, operation_key: str) -> bool:
        """Check if circuit breaker should be closed"""
        with self._lock:
            open_time = self.circuit_open_times.get(operation_key)
            if open_time:
                if datetime.now() - open_time > timedelta(seconds=self.recovery_timeout):
                    # Reset circuit
                    self.failure_counts[operation_key] = 0
                    del self.circuit_open_times[operation_key]
                    logger.info(f"Circuit breaker CLOSED for {operation_key}")
                    return True
                return False
            return True


class GracefulDegradationHandler(ErrorRecoveryHandler):
    """Handles errors by degrading functionality gracefully"""
    
    def __init__(self):
        self.degraded_features: Dict[str, datetime] = {}
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Can handle non-critical errors"""
        return error_context.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
    
    def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Degrade functionality while maintaining core operations"""
        start_time = time.time()
        
        component_key = error_context.component
        self.degraded_features[component_key] = datetime.now()
        
        fallback_value = self._provide_degraded_functionality(error_context)
        
        logger.warning(f"Graceful degradation activated for {component_key}")
        
        recovery_time = time.time() - start_time
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
            error_resolved=True,
            fallback_value=fallback_value,
            recovery_time_seconds=recovery_time,
            additional_context={
                "degraded_component": component_key,
                "degradation_level": "partial"
            }
        )
    
    def _provide_degraded_functionality(self, error_context: ErrorContext) -> Any:
        """Provide minimal functionality when full feature fails"""
        component = error_context.component.lower()
        
        if "sentiment" in component:
            return {"sentiment": "neutral", "confidence": 0.0, "status": "degraded"}
        elif "learning" in component:
            return {"insights": [], "confidence": 0.0, "status": "degraded"}
        elif "metrics" in component:
            return {"status": "degraded", "data": {}}
        else:
            return {"status": "degraded", "message": "Reduced functionality"}


class ErrorRecoverySystem:
    """Centralized error recovery and resilience system"""
    
    def __init__(self, persist_errors: bool = True):
        self.handlers: List[ErrorRecoveryHandler] = []
        self.error_history: List[ErrorContext] = []
        self.recovery_history: List[RecoveryResult] = []
        self.persist_errors = persist_errors
        self.error_log_path = Path("docs/status/error_recovery.json")
        self._error_counter = 0
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default recovery handlers"""
        self.handlers.extend([
            RetryRecoveryHandler(),
            CircuitBreakerRecoveryHandler(), 
            GracefulDegradationHandler()
        ])
    
    def register_handler(self, handler: ErrorRecoveryHandler):
        """Register a custom error recovery handler"""
        self.handlers.append(handler)
        logger.info(f"Registered recovery handler: {handler.__class__.__name__}")
    
    def handle_error(self, 
                    error: Exception,
                    component: str,
                    operation: str,
                    context_data: Optional[Dict[str, Any]] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> RecoveryResult:
        """Handle an error and attempt recovery"""
        
        # Create error context
        error_context = ErrorContext(
            error_id=self._generate_error_id(),
            timestamp=datetime.now(),
            error_type=error.__class__.__name__,
            error_message=str(error),
            severity=severity,
            component=component,
            operation=operation,
            context_data=context_data or {},
            stack_trace=traceback.format_exc()
        )
        
        self.error_history.append(error_context)
        logger.error(f"Error in {component}.{operation}: {error}")
        
        # Try recovery handlers
        for handler in self.handlers:
            if handler.can_handle(error_context):
                try:
                    logger.info(f"Attempting recovery with {handler.__class__.__name__}")
                    recovery_result = handler.recover(error_context)
                    
                    error_context.recovery_attempts += 1
                    error_context.recovery_strategy = recovery_result.strategy_used
                    
                    self.recovery_history.append(recovery_result)
                    
                    if self.persist_errors:
                        self._persist_error_data()
                    
                    logger.info(f"Recovery {'successful' if recovery_result.success else 'failed'} "
                               f"using {recovery_result.strategy_used.value}")
                    
                    return recovery_result
                    
                except Exception as recovery_error:
                    logger.error(f"Recovery handler failed: {recovery_error}")
                    continue
        
        # No handler could recover - escalate
        logger.critical(f"No recovery possible for error {error_context.error_id}")
        
        escalation_result = RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ESCALATE,
            error_resolved=False,
            additional_context={
                "error_id": error_context.error_id,
                "escalation_required": True
            }
        )
        
        self.recovery_history.append(escalation_result)
        
        if self.persist_errors:
            self._persist_error_data()
        
        return escalation_result
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        self._error_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ERR_{timestamp}_{self._error_counter:04d}"
    
    def _persist_error_data(self):
        """Persist error and recovery data to disk"""
        try:
            self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "errors": [asdict(error) for error in self.error_history[-100:]],  # Last 100 errors
                "recoveries": [asdict(recovery) for recovery in self.recovery_history[-100:]],
                "summary": self.get_error_summary(),
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.error_log_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to persist error data: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error patterns and recovery effectiveness"""
        if not self.error_history:
            return {"status": "no_errors"}
        
        # Error frequency by component
        component_errors = {}
        for error in self.error_history:
            component = error.component
            component_errors[component] = component_errors.get(component, 0) + 1
        
        # Recovery success rate
        successful_recoveries = sum(1 for r in self.recovery_history if r.success)
        recovery_rate = successful_recoveries / len(self.recovery_history) if self.recovery_history else 0
        
        # Most common error types
        error_types = {}
        for error in self.error_history:
            error_type = error.error_type
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "total_recoveries": len(self.recovery_history),
            "recovery_success_rate": recovery_rate,
            "errors_by_component": component_errors,
            "most_common_errors": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
            "critical_errors": sum(1 for e in self.error_history if e.severity == ErrorSeverity.CRITICAL)
        }


# Global error recovery system
error_recovery = ErrorRecoverySystem()


def resilient_operation(component: str, 
                       operation: str,
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       context_data: Optional[Dict[str, Any]] = None):
    """Decorator for resilient operation execution"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    recovery_result = error_recovery.handle_error(
                        error=e,
                        component=component,
                        operation=operation,
                        context_data=context_data,
                        severity=severity
                    )
                    
                    if recovery_result.success and recovery_result.strategy_used == RecoveryStrategy.RETRY:
                        continue  # Retry the operation
                    elif recovery_result.fallback_value is not None:
                        return recovery_result.fallback_value
                    else:
                        raise  # Re-raise if no recovery possible
            
            # All attempts exhausted
            raise RuntimeError(f"Operation {component}.{operation} failed after {max_attempts} attempts")
        
        return wrapper
    return decorator


def handle_critical_error(error: Exception, context: str = "unknown") -> None:
    """Handle critical errors that require immediate attention"""
    error_recovery.handle_error(
        error=error,
        component="system",
        operation=context,
        severity=ErrorSeverity.CRITICAL
    )
    
    # Additional critical error handling
    logger.critical(f"CRITICAL ERROR in {context}: {error}")
    
    # Could trigger alerts, notifications, etc.


def get_system_health() -> Dict[str, Any]:
    """Get overall system health based on error patterns"""
    summary = error_recovery.get_error_summary()
    
    if summary.get("status") == "no_errors":
        health_score = 100
    else:
        # Calculate health score based on recent error patterns
        total_errors = summary.get("total_errors", 0)
        critical_errors = summary.get("critical_errors", 0)
        recovery_rate = summary.get("recovery_success_rate", 0)
        
        # Simple health scoring algorithm
        health_score = max(0, 100 - (total_errors * 2) - (critical_errors * 10) + (recovery_rate * 20))
    
    return {
        "health_score": health_score,
        "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "critical",
        "error_summary": summary,
        "recommendations": _get_health_recommendations(summary)
    }


def _get_health_recommendations(summary: Dict[str, Any]) -> List[str]:
    """Get recommendations based on error patterns"""
    recommendations = []
    
    if summary.get("critical_errors", 0) > 0:
        recommendations.append("Address critical errors immediately")
    
    if summary.get("recovery_success_rate", 1) < 0.8:
        recommendations.append("Improve error recovery strategies")
    
    errors_by_component = summary.get("errors_by_component", {})
    if errors_by_component:
        problematic_component = max(errors_by_component.items(), key=lambda x: x[1])[0]
        recommendations.append(f"Focus on {problematic_component} component reliability")
    
    return recommendations