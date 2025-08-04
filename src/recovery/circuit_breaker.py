#!/usr/bin/env python3
"""
Circuit breaker pattern for resilient execution
"""

import time
import logging
from typing import Callable, Any, Dict, Optional
from enum import Enum
from dataclasses import dataclass
from threading import Lock


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    timeout: float = 60.0  # seconds
    recovery_timeout: float = 30.0  # seconds  
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker for resilient operations"""
    
    def __init__(self, config: CircuitConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = Lock()
        self.logger = logging.getLogger("circuit_breaker")
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN - failing fast")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.config.timeout
    
    def _on_success(self):
        """Handle successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info("Circuit breaker CLOSED - operation successful")
        self.failure_count = 0
        
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures")
    
    def get_stats(self) -> Dict:
        """Get circuit breaker statistics"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "is_available": self.state != CircuitState.OPEN
        }


class RetryStrategy:
    """Configurable retry strategy with exponential backoff"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.logger = logging.getLogger("retry_strategy")
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Function succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_attempts - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
        
        # All attempts failed
        self.logger.error(f"All {self.max_attempts} attempts failed")
        raise last_exception


class RecoveryManager:
    """Manages error recovery and resilience patterns"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_strategies: Dict[str, RetryStrategy] = {}
        self.logger = logging.getLogger("recovery_manager")
    
    def register_circuit_breaker(self, name: str, config: CircuitConfig) -> CircuitBreaker:
        """Register a new circuit breaker"""
        breaker = CircuitBreaker(config)
        self.circuit_breakers[name] = breaker
        self.logger.info(f"Registered circuit breaker: {name}")
        return breaker
    
    def register_retry_strategy(self, name: str, max_attempts: int = 3, 
                              base_delay: float = 1.0) -> RetryStrategy:
        """Register a new retry strategy"""
        strategy = RetryStrategy(max_attempts=max_attempts, base_delay=base_delay)
        self.retry_strategies[name] = strategy
        self.logger.info(f"Registered retry strategy: {name}")
        return strategy
    
    def execute_with_resilience(self, operation_name: str, func: Callable, 
                              *args, **kwargs) -> Any:
        """Execute function with both circuit breaker and retry"""
        breaker = self.circuit_breakers.get(operation_name)
        retry_strategy = self.retry_strategies.get(operation_name)
        
        if breaker and retry_strategy:
            # Use both patterns
            return retry_strategy.execute(breaker.call, func, *args, **kwargs)
        elif breaker:
            # Use only circuit breaker
            return breaker.call(func, *args, **kwargs)
        elif retry_strategy:
            # Use only retry
            return retry_strategy.execute(func, *args, **kwargs)
        else:
            # No resilience patterns
            return func(*args, **kwargs)
    
    def get_health_status(self) -> Dict:
        """Get health status of all resilience components"""
        status = {
            "circuit_breakers": {},
            "timestamp": time.time()
        }
        
        for name, breaker in self.circuit_breakers.items():
            status["circuit_breakers"][name] = breaker.get_stats()
        
        return status
    
    def reset_circuit_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker"""
        if name in self.circuit_breakers:
            breaker = self.circuit_breakers[name]
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.last_failure_time = None
            self.logger.info(f"Circuit breaker {name} manually reset")
            return True
        return False