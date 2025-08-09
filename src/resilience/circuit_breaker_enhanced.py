#!/usr/bin/env python3
"""
Enhanced Circuit Breaker Pattern for Robust ADO Operations
Implements advanced resilience patterns with adaptive thresholds
"""

import time
import logging
import threading
from typing import Callable, Any, Optional, Dict, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import functools
import statistics
from collections import deque


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls fail immediately
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    average_response_time: float = 0.0
    state_transitions: int = 0


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, message: str, metrics: CircuitMetrics):
        super().__init__(message)
        self.metrics = metrics


class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with adaptive behavior and monitoring"""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
        adaptive_threshold: bool = True
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.adaptive_threshold = adaptive_threshold
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"circuit_breaker_{name}")
        
        # Adaptive behavior components
        self.response_times = deque(maxlen=100)
        self.error_rates = deque(maxlen=20)  # Track error rates over time
        self.last_state_change = time.time()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            self._update_state()
            
            if self.state == CircuitState.OPEN:
                self._record_failure()
                raise CircuitBreakerException(
                    f"Circuit breaker '{self.name}' is OPEN",
                    self.metrics
                )
                
        # Execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            self._record_success(response_time)
            return result
            
        except self.expected_exception as e:
            response_time = time.time() - start_time
            self._record_failure(response_time)
            raise e
            
    def _update_state(self):
        """Update circuit breaker state based on current conditions"""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            if current_time - self.last_state_change >= self.timeout:
                self._transition_to_half_open()
                
        elif self.state == CircuitState.HALF_OPEN:
            if self.metrics.consecutive_failures >= 1:
                self._transition_to_open()
            elif self.metrics.successful_calls >= self.success_threshold:
                self._transition_to_closed()
                
        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open()
                
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on failure patterns"""
        # Basic threshold check
        if self.metrics.consecutive_failures >= self._get_dynamic_threshold():
            return True
            
        # Adaptive checks
        if self.adaptive_threshold:
            # Check error rate over recent calls
            recent_calls = min(50, self.metrics.total_calls)
            if recent_calls >= 10:
                error_rate = self.metrics.failed_calls / self.metrics.total_calls
                if error_rate > 0.5:  # 50% error rate
                    return True
                    
            # Check response time degradation
            if len(self.response_times) >= 10:
                recent_avg = statistics.mean(list(self.response_times)[-10:])
                overall_avg = statistics.mean(self.response_times)
                if recent_avg > overall_avg * 2:  # Response time doubled
                    return True
                    
        return False
        
    def _get_dynamic_threshold(self) -> int:
        """Get dynamic failure threshold based on historical data"""
        if not self.adaptive_threshold:
            return self.failure_threshold
            
        # Adjust threshold based on recent error patterns
        if len(self.error_rates) >= 5:
            recent_avg_error_rate = statistics.mean(list(self.error_rates)[-5:])
            if recent_avg_error_rate > 0.3:
                return max(2, self.failure_threshold - 1)  # Lower threshold
            elif recent_avg_error_rate < 0.1:
                return self.failure_threshold + 2  # Higher threshold
                
        return self.failure_threshold
        
    def _record_success(self, response_time: float = 0.0):
        """Record successful call"""
        with self.lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()
            
            if response_time > 0:
                self.response_times.append(response_time)
                self._update_average_response_time()
                
            # Update error rate tracking
            self._update_error_rate()
            
            self.logger.debug(f"Circuit '{self.name}' recorded success")
            
    def _record_failure(self, response_time: float = 0.0):
        """Record failed call"""
        with self.lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.last_failure_time = time.time()
            
            if response_time > 0:
                self.response_times.append(response_time)
                self._update_average_response_time()
                
            # Update error rate tracking
            self._update_error_rate()
            
            self.logger.warning(
                f"Circuit '{self.name}' recorded failure. "
                f"Consecutive failures: {self.metrics.consecutive_failures}"
            )
            
    def _update_average_response_time(self):
        """Update average response time metric"""
        if self.response_times:
            self.metrics.average_response_time = statistics.mean(self.response_times)
            
    def _update_error_rate(self):
        """Update error rate tracking"""
        if self.metrics.total_calls > 0:
            current_error_rate = self.metrics.failed_calls / self.metrics.total_calls
            self.error_rates.append(current_error_rate)
            
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        self.state = CircuitState.OPEN
        self.last_state_change = time.time()
        self.metrics.state_transitions += 1
        self.logger.error(f"Circuit '{self.name}' transitioned to OPEN")
        
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self.metrics.state_transitions += 1
        self.logger.info(f"Circuit '{self.name}' transitioned to HALF_OPEN")
        
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.metrics.state_transitions += 1
        self.metrics.consecutive_failures = 0
        self.logger.info(f"Circuit '{self.name}' transitioned to CLOSED")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        with self.lock:
            success_rate = 0.0
            if self.metrics.total_calls > 0:
                success_rate = self.metrics.successful_calls / self.metrics.total_calls
                
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.metrics.total_calls,
                'successful_calls': self.metrics.successful_calls,
                'failed_calls': self.metrics.failed_calls,
                'success_rate': success_rate,
                'consecutive_failures': self.metrics.consecutive_failures,
                'average_response_time': self.metrics.average_response_time,
                'state_transitions': self.metrics.state_transitions,
                'last_failure_time': self.metrics.last_failure_time,
                'last_success_time': self.metrics.last_success_time,
                'current_threshold': self._get_dynamic_threshold()
            }
            
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.metrics = CircuitMetrics()
            self.response_times.clear()
            self.error_rates.clear()
            self.last_state_change = time.time()
            self.logger.info(f"Circuit '{self.name}' has been reset")
            
    def force_open(self):
        """Force circuit breaker to OPEN state"""
        with self.lock:
            self._transition_to_open()
            self.logger.warning(f"Circuit '{self.name}' was forced OPEN")
            
    def force_close(self):
        """Force circuit breaker to CLOSED state"""
        with self.lock:
            self._transition_to_closed()
            self.logger.info(f"Circuit '{self.name}' was forced CLOSED")


def circuit_breaker(
    name: str = None,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    expected_exception: type = Exception,
    recovery_timeout: float = 30.0,
    success_threshold: int = 3,
    adaptive_threshold: bool = True
):
    """Decorator for applying circuit breaker pattern to functions"""
    def decorator(func: Callable) -> Callable:
        cb_name = name or f"{func.__module__}.{func.__name__}"
        breaker = EnhancedCircuitBreaker(
            name=cb_name,
            failure_threshold=failure_threshold,
            timeout=timeout,
            expected_exception=expected_exception,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            adaptive_threshold=adaptive_threshold
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
            
        # Attach circuit breaker to function for external access
        wrapper.circuit_breaker = breaker
        return wrapper
        
    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self.breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self.lock = threading.RLock()
        
    def register(self, breaker: EnhancedCircuitBreaker) -> None:
        """Register a circuit breaker"""
        with self.lock:
            self.breakers[breaker.name] = breaker
            
    def get_breaker(self, name: str) -> Optional[EnhancedCircuitBreaker]:
        """Get circuit breaker by name"""
        return self.breakers.get(name)
        
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered circuit breakers"""
        with self.lock:
            return {name: breaker.get_metrics() 
                   for name, breaker in self.breakers.items()}
                   
    def reset_all(self) -> None:
        """Reset all circuit breakers"""
        with self.lock:
            for breaker in self.breakers.values():
                breaker.reset()
                
    def get_unhealthy_breakers(self) -> List[str]:
        """Get list of circuit breakers in OPEN state"""
        unhealthy = []
        for name, breaker in self.breakers.items():
            if breaker.state == CircuitState.OPEN:
                unhealthy.append(name)
        return unhealthy


# Global registry instance
registry = CircuitBreakerRegistry()


def main():
    """CLI entry point for circuit breaker monitoring"""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python circuit_breaker_enhanced.py <command>")
        print("Commands: metrics, status, reset")
        return
        
    command = sys.argv[1]
    
    if command == "metrics":
        metrics = registry.get_all_metrics()
        print(json.dumps(metrics, indent=2, default=str))
        
    elif command == "status":
        unhealthy = registry.get_unhealthy_breakers()
        if unhealthy:
            print(f"⚠️  Unhealthy circuit breakers: {', '.join(unhealthy)}")
        else:
            print("✅ All circuit breakers are healthy")
            
    elif command == "reset":
        registry.reset_all()
        print("🔄 All circuit breakers have been reset")
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
