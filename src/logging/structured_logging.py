#!/usr/bin/env python3
"""
Advanced Structured Logging System
Comprehensive logging with structured data, log correlation, and distributed tracing support
"""

import json
import logging
import time
import os
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback
import inspect
from functools import wraps
import uuid
from contextvars import ContextVar
from collections import deque
import gzip
import shutil
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


# Context variables for distributed tracing
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar('span_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


@dataclass
class LogContext:
    """Structured log context"""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    environment: Optional[str] = None
    version: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    cpu_percent: Optional[float] = None


@dataclass
class SecurityContext:
    """Security-related context for logging"""
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    auth_method: Optional[str] = None
    permissions: Optional[List[str]] = None
    sensitive_operation: bool = False


class StructuredLogger:
    """Advanced structured logger with correlation and performance tracking"""
    
    def __init__(self, name: str, log_dir: Path = None, 
                 max_file_size: int = 100_000_000,  # 100MB
                 backup_count: int = 10,
                 enable_console: bool = True,
                 enable_json_format: bool = True):
        
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup log directory
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.log_dir = log_dir
        self.enable_json_format = enable_json_format
        
        # Create formatters
        self.json_formatter = StructuredFormatter()
        self.console_formatter = ConsoleFormatter()
        
        # Setup file handlers
        self._setup_file_handlers(max_file_size, backup_count)
        
        # Setup console handler
        if enable_console:
            self._setup_console_handler()
        
        # Performance tracking
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.metrics_lock = threading.Lock()
        
        # Log correlation
        self.correlation_store: deque = deque(maxlen=10000)
        
        # Context
        self.default_context = LogContext(
            environment=os.getenv('ENVIRONMENT', 'development'),
            version=os.getenv('VERSION', '1.0.0')
        )
    
    def _setup_file_handlers(self, max_file_size: int, backup_count: int):
        """Setup file-based log handlers"""
        
        # Main application log (rotating by size)
        app_log_file = self.log_dir / f"{self.name}.log"
        app_handler = RotatingFileHandler(
            app_log_file, 
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        app_handler.setFormatter(self.json_formatter if self.enable_json_format else self.console_formatter)
        app_handler.setLevel(logging.INFO)
        self.logger.addHandler(app_handler)
        
        # Error log (errors and above)
        error_log_file = self.log_dir / f"{self.name}_error.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setFormatter(self.json_formatter if self.enable_json_format else self.console_formatter)
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
        
        # Debug log (all levels, daily rotation)
        debug_log_file = self.log_dir / f"{self.name}_debug.log"
        debug_handler = TimedRotatingFileHandler(
            debug_log_file,
            when='midnight',
            interval=1,
            backupCount=7,  # Keep 7 days
            utc=True
        )
        debug_handler.setFormatter(self.json_formatter if self.enable_json_format else self.console_formatter)
        debug_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(debug_handler)
        
        # Performance log (for performance metrics)
        perf_log_file = self.log_dir / f"{self.name}_performance.log"
        perf_handler = TimedRotatingFileHandler(
            perf_log_file,
            when='hourly',
            interval=1,
            backupCount=48,  # Keep 48 hours
            utc=True
        )
        perf_handler.setFormatter(self.json_formatter if self.enable_json_format else self.console_formatter)
        perf_handler.addFilter(PerformanceLogFilter())
        perf_handler.setLevel(logging.INFO)
        self.logger.addHandler(perf_handler)
    
    def _setup_console_handler(self):
        """Setup console handler"""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
    
    def _get_current_context(self) -> LogContext:
        """Get current logging context from context vars"""
        context = LogContext(
            trace_id=trace_id_var.get(),
            span_id=span_id_var.get(),
            user_id=user_id_var.get(),
            environment=self.default_context.environment,
            version=self.default_context.version
        )
        return context
    
    def _create_log_record(self, level: str, message: str, 
                          context: Optional[LogContext] = None,
                          extra: Optional[Dict[str, Any]] = None,
                          exception: Optional[Exception] = None,
                          performance: Optional[PerformanceMetrics] = None,
                          security: Optional[SecurityContext] = None) -> Dict[str, Any]:
        """Create structured log record"""
        
        # Get caller information
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller (skip logging methods)
            caller_frame = frame.f_back.f_back.f_back
            caller_info = {
                'file': os.path.basename(caller_frame.f_code.co_filename),
                'function': caller_frame.f_code.co_name,
                'line': caller_frame.f_lineno
            }
        except (AttributeError, TypeError):
            caller_info = {'file': 'unknown', 'function': 'unknown', 'line': 0}
        finally:
            del frame
        
        # Base record
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            'caller': caller_info,
            'thread_id': threading.get_ident(),
            'process_id': os.getpid()
        }
        
        # Add context
        if context is None:
            context = self._get_current_context()
        
        if context:
            context_dict = {k: v for k, v in asdict(context).items() if v is not None}
            if context_dict:
                record['context'] = context_dict
        
        # Add extra fields
        if extra:
            record['extra'] = extra
        
        # Add exception information
        if exception:
            record['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        # Add performance metrics
        if performance:
            perf_dict = {k: v for k, v in asdict(performance).items() if v is not None}
            if perf_dict:
                record['performance'] = perf_dict
                record['log_type'] = 'performance'
        
        # Add security context
        if security:
            security_dict = {k: v for k, v in asdict(security).items() if v is not None}
            if security_dict:
                record['security'] = security_dict
                if security.sensitive_operation:
                    record['sensitive'] = True
        
        return record
    
    def debug(self, message: str, context: Optional[LogContext] = None,
              extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message"""
        record = self._create_log_record('DEBUG', message, context, extra, **kwargs)
        self.logger.debug(record)
        self._store_correlation(record)
    
    def info(self, message: str, context: Optional[LogContext] = None,
             extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message"""
        record = self._create_log_record('INFO', message, context, extra, **kwargs)
        self.logger.info(record)
        self._store_correlation(record)
    
    def warning(self, message: str, context: Optional[LogContext] = None,
                extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message"""
        record = self._create_log_record('WARNING', message, context, extra, **kwargs)
        self.logger.warning(record)
        self._store_correlation(record)
    
    def error(self, message: str, context: Optional[LogContext] = None,
              extra: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None,
              **kwargs):
        """Log error message"""
        record = self._create_log_record('ERROR', message, context, extra, exception, **kwargs)
        self.logger.error(record)
        self._store_correlation(record)
    
    def critical(self, message: str, context: Optional[LogContext] = None,
                 extra: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None,
                 **kwargs):
        """Log critical message"""
        record = self._create_log_record('CRITICAL', message, context, extra, exception, **kwargs)
        self.logger.critical(record)
        self._store_correlation(record)
    
    def performance(self, operation: str, metrics: PerformanceMetrics,
                   context: Optional[LogContext] = None,
                   extra: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        message = f"Performance metrics for {operation}"
        record = self._create_log_record('INFO', message, context, extra, performance=metrics)
        record['log_type'] = 'performance'
        self.logger.info(record)
        self._store_correlation(record)
    
    def security(self, message: str, security_context: SecurityContext,
                context: Optional[LogContext] = None,
                extra: Optional[Dict[str, Any]] = None):
        """Log security-related event"""
        record = self._create_log_record('WARNING', message, context, extra, security=security_context)
        record['log_type'] = 'security'
        self.logger.warning(record)
        self._store_correlation(record)
    
    def _store_correlation(self, record: Dict[str, Any]):
        """Store log record for correlation"""
        if 'context' in record and 'trace_id' in record['context']:
            self.correlation_store.append({
                'timestamp': record['timestamp'],
                'trace_id': record['context']['trace_id'],
                'level': record['level'],
                'message': record['message'],
                'logger': record['logger']
            })
    
    def get_correlated_logs(self, trace_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs correlated by trace ID"""
        correlated = [
            log for log in list(self.correlation_store)
            if log.get('trace_id') == trace_id
        ]
        return sorted(correlated, key=lambda x: x['timestamp'])[-limit:]
    
    def start_performance_tracking(self, operation_id: str) -> str:
        """Start tracking performance for an operation"""
        import psutil
        
        metrics = PerformanceMetrics(
            start_time=time.time(),
            memory_before_mb=psutil.Process().memory_info().rss / (1024 * 1024)
        )
        
        with self.metrics_lock:
            self.performance_metrics[operation_id] = metrics
        
        return operation_id
    
    def end_performance_tracking(self, operation_id: str) -> Optional[PerformanceMetrics]:
        """End tracking performance for an operation"""
        with self.metrics_lock:
            metrics = self.performance_metrics.pop(operation_id, None)
        
        if metrics:
            import psutil
            end_time = time.time()
            metrics.end_time = end_time
            metrics.duration_ms = (end_time - metrics.start_time) * 1000
            metrics.memory_after_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Log the performance metrics
            self.performance(operation_id, metrics)
        
        return metrics


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        if hasattr(record, 'getMessage'):
            # Standard logging record
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread_id': record.thread,
                'process_id': record.process
            }
            
            if record.exc_info:
                log_data['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': self.formatException(record.exc_info)
                }
        else:
            # Already structured record
            log_data = record
        
        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        if hasattr(record, 'getMessage'):
            # Standard logging record
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
            level_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            
            base_msg = f"{timestamp} | {level_color}{record.levelname:8}{reset_color} | {record.name} | {record.getMessage()}"
            
            if record.exc_info:
                base_msg += f"\n{self.formatException(record.exc_info)}"
            
            return base_msg
        else:
            # Structured record
            if isinstance(record, dict):
                timestamp = record.get('timestamp', datetime.now().isoformat())
                level = record.get('level', 'INFO')
                logger_name = record.get('logger', 'unknown')
                message = record.get('message', '')
                
                level_color = self.COLORS.get(level, '')
                reset_color = self.COLORS['RESET']
                
                base_msg = f"{timestamp} | {level_color}{level:8}{reset_color} | {logger_name} | {message}"
                
                # Add context if available
                if 'context' in record and record['context']:
                    context_items = []
                    for key, value in record['context'].items():
                        if value:
                            context_items.append(f"{key}={value}")
                    if context_items:
                        base_msg += f" | Context: {', '.join(context_items)}"
                
                # Add performance info
                if 'performance' in record:
                    perf = record['performance']
                    if 'duration_ms' in perf:
                        base_msg += f" | Duration: {perf['duration_ms']:.2f}ms"
                
                return base_msg
            else:
                return str(record)


class PerformanceLogFilter(logging.Filter):
    """Filter to only allow performance-related logs"""
    
    def filter(self, record):
        if isinstance(record.msg, dict):
            return record.msg.get('log_type') == 'performance'
        return False


class DistributedTracing:
    """Distributed tracing context manager"""
    
    def __init__(self, operation_name: str, trace_id: Optional[str] = None,
                 parent_span_id: Optional[str] = None, user_id: Optional[str] = None):
        self.operation_name = operation_name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.user_id = user_id
        self.start_time = None
        
        # Context tokens for restoration
        self.trace_token = None
        self.span_token = None
        self.user_token = None
    
    def __enter__(self):
        self.start_time = time.time()
        
        # Set context variables
        self.trace_token = trace_id_var.set(self.trace_id)
        self.span_token = span_id_var.set(self.span_id)
        if self.user_id:
            self.user_token = user_id_var.set(self.user_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Reset context variables
        trace_id_var.reset(self.trace_token)
        span_id_var.reset(self.span_token)
        if self.user_token:
            user_id_var.reset(self.user_token)
        
        # Log span completion
        logger = get_logger('distributed_tracing')
        context = LogContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            operation=self.operation_name
        )
        
        extra = {
            'span_duration_ms': duration * 1000,
            'parent_span_id': self.parent_span_id,
            'span_type': 'operation'
        }
        
        if exc_type:
            logger.error(
                f"Span {self.operation_name} failed",
                context=context,
                extra=extra,
                exception=exc_val
            )
        else:
            logger.info(
                f"Span {self.operation_name} completed",
                context=context,
                extra=extra
            )


def get_logger(name: str, **kwargs) -> StructuredLogger:
    """Get or create a structured logger"""
    return StructuredLogger(name, **kwargs)


def trace_operation(operation_name: Optional[str] = None):
    """Decorator for tracing operations"""
    def decorator(func):
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with DistributedTracing(op_name):
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with DistributedTracing(op_name):
                return await func(*args, **kwargs)
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def log_performance(func):
    """Decorator for logging function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        operation_id = f"{func.__name__}_{int(time.time())}"
        
        logger.start_performance_tracking(operation_id)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            logger.end_performance_tracking(operation_id)
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        operation_id = f"{func.__name__}_{int(time.time())}"
        
        logger.start_performance_tracking(operation_id)
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            logger.end_performance_tracking(operation_id)
    
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


# Global logger instances
_loggers: Dict[str, StructuredLogger] = {}
_logger_lock = threading.Lock()


def setup_logging(log_level: str = "INFO", log_dir: Optional[Path] = None,
                  enable_console: bool = True, enable_json: bool = True):
    """Setup global logging configuration"""
    
    # Create logs directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set global log level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Create main application logger
    main_logger = StructuredLogger(
        "terragon_sdlc",
        log_dir=log_dir,
        enable_console=enable_console,
        enable_json_format=enable_json
    )
    
    with _logger_lock:
        _loggers["terragon_sdlc"] = main_logger
    
    return main_logger


if __name__ == "__main__":
    # Example usage
    import time
    
    # Setup logging
    setup_logging("DEBUG", enable_console=True)
    logger = get_logger("example")
    
    # Basic logging
    logger.info("Application started")
    
    # Logging with context
    context = LogContext(
        user_id="user123",
        operation="data_processing",
        component="example_module"
    )
    logger.info("Processing data", context=context, extra={"record_count": 1000})
    
    # Error logging
    try:
        raise ValueError("Example error")
    except ValueError as e:
        logger.error("Data processing failed", context=context, exception=e)
    
    # Performance logging
    with logger.start_performance_tracking("data_processing") as perf_id:
        time.sleep(0.1)  # Simulate work
    
    # Distributed tracing
    with DistributedTracing("user_operation", user_id="user123"):
        logger.info("User operation started")
        time.sleep(0.05)
        logger.info("User operation completed")
    
    # Security logging
    security_ctx = SecurityContext(
        client_ip="192.168.1.100",
        user_agent="Mozilla/5.0",
        auth_method="oauth2",
        sensitive_operation=True
    )
    logger.security("Sensitive operation performed", security_ctx)
    
    print("Logging examples completed. Check logs/ directory for output files.")