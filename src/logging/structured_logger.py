#!/usr/bin/env python3
"""
Structured Logging and Audit Trail System
Provides comprehensive logging with structured JSON output, audit trails, and security event tracking
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import uuid


class LogLevel(Enum):
    """Enhanced log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 55  # Special security event level


class EventType(Enum):
    """Types of events for audit logging"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_EVENT = "performance_event"
    ERROR_EVENT = "error_event"


@dataclass
class LogContext:
    """Structured logging context"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    transaction_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass 
class AuditEvent:
    """Audit event structure"""
    event_id: str
    timestamp: str
    event_type: EventType
    actor: Optional[str]
    action: str
    resource: Optional[str]
    outcome: str  # success/failure/partial
    metadata: Dict[str, Any]
    risk_score: int = 0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class StructuredLogger:
    """Enhanced structured logger with audit trails"""
    
    def __init__(self, 
                 name: str,
                 log_dir: str = "logs",
                 enable_audit: bool = True,
                 enable_security_logging: bool = True,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 10):
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.enable_audit = enable_audit
        self.enable_security_logging = enable_security_logging
        self.context = LogContext()
        self.lock = threading.RLock()
        
        # Set up main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(max_file_size, backup_count)
        
        # Set up audit logger
        if enable_audit:
            self.audit_logger = logging.getLogger(f"{name}_audit")
            if not self.audit_logger.handlers:
                self._setup_audit_handlers(max_file_size, backup_count)
        
        # Set up security logger
        if enable_security_logging:
            self.security_logger = logging.getLogger(f"{name}_security")
            if not self.security_logger.handlers:
                self._setup_security_handlers(max_file_size, backup_count)
    
    def _setup_handlers(self, max_file_size: int, backup_count: int):
        """Setup main logging handlers"""
        formatter = StructuredFormatter()
        
        # Console handler with color support
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColorFormatter() if sys.stdout.isatty() else formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def _setup_audit_handlers(self, max_file_size: int, backup_count: int):
        """Setup audit logging handlers"""
        formatter = StructuredFormatter()
        
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_audit.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(formatter)
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def _setup_security_handlers(self, max_file_size: int, backup_count: int):
        """Setup security logging handlers"""
        formatter = StructuredFormatter()
        
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_security.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        security_handler.setLevel(logging.INFO)
        security_handler.setFormatter(formatter)
        self.security_logger.addHandler(security_handler)
        self.security_logger.setLevel(logging.INFO)
    
    def set_context(self, **kwargs):
        """Set logging context"""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.context, key):
                    setattr(self.context, key, value)
    
    def clear_context(self):
        """Clear logging context"""
        with self.lock:
            self.context = LogContext()
    
    def _create_log_record(self, level: str, message: str, extra: Optional[Dict] = None) -> Dict:
        """Create structured log record"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            "thread_id": threading.get_ident(),
            "process_id": os.getpid()
        }
        
        # Add context
        if self.context:
            context_dict = {k: v for k, v in asdict(self.context).items() if v is not None}
            if context_dict:
                record["context"] = context_dict
        
        # Add extra fields
        if extra:
            record.update(extra)
        
        # Add stack trace for errors
        if level in ["ERROR", "CRITICAL"]:
            record["stack_trace"] = traceback.format_stack()
        
        return record
    
    def trace(self, message: str, **kwargs):
        """Log trace message"""
        record = self._create_log_record("TRACE", message, kwargs)
        self.logger.log(LogLevel.TRACE.value, json.dumps(record))
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        record = self._create_log_record("DEBUG", message, kwargs)
        self.logger.debug(json.dumps(record))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        record = self._create_log_record("INFO", message, kwargs)
        self.logger.info(json.dumps(record))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        record = self._create_log_record("WARNING", message, kwargs)
        self.logger.warning(json.dumps(record))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        extra = kwargs.copy()
        if exception:
            extra.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "exception_traceback": traceback.format_exception(type(exception), exception, exception.__traceback__)
            })
        
        record = self._create_log_record("ERROR", message, extra)
        self.logger.error(json.dumps(record))
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        extra = kwargs.copy()
        if exception:
            extra.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "exception_traceback": traceback.format_exception(type(exception), exception, exception.__traceback__)
            })
        
        record = self._create_log_record("CRITICAL", message, extra)
        self.logger.critical(json.dumps(record))
    
    def security(self, message: str, risk_score: int = 0, **kwargs):
        """Log security event"""
        if not self.enable_security_logging:
            return
        
        extra = kwargs.copy()
        extra.update({
            "risk_score": risk_score,
            "security_event": True
        })
        
        record = self._create_log_record("SECURITY", message, extra)
        self.security_logger.log(LogLevel.SECURITY.value, json.dumps(record))
    
    def audit(self, event: AuditEvent):
        """Log audit event"""
        if not self.enable_audit:
            return
        
        record = {
            "timestamp": event.timestamp,
            "level": "AUDIT",
            "logger": f"{self.name}_audit",
            **asdict(event)
        }
        
        self.audit_logger.info(json.dumps(record))
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        extra = kwargs.copy()
        extra.update({
            "operation": operation,
            "duration_ms": duration * 1000,
            "performance_event": True
        })
        
        record = self._create_log_record("INFO", f"Performance: {operation}", extra)
        self.logger.info(json.dumps(record))


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging"""
    
    def format(self, record):
        # If the record is already JSON, return as-is
        if hasattr(record, 'getMessage') and record.getMessage().startswith('{'):
            return record.getMessage()
        
        # Otherwise, create structured record
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class ColorFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'SECURITY': '\033[95m',  # Bright Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Parse JSON if it's structured
        try:
            if record.getMessage().startswith('{'):
                data = json.loads(record.getMessage())
                level = data.get('level', record.levelname)
                message = data.get('message', '')
                timestamp = data.get('timestamp', datetime.now().isoformat())
                
                color = self.COLORS.get(level, '')
                reset = self.COLORS['RESET']
                
                return f"{color}[{timestamp[:19]}] {level:8} | {message}{reset}"
        except:
            pass
        
        # Fallback to standard formatting
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        return f"{color}[{datetime.fromtimestamp(record.created).isoformat()[:19]}] {record.levelname:8} | {record.getMessage()}{reset}"


# Singleton logger instances
_loggers: Dict[str, StructuredLogger] = {}
_lock = threading.Lock()


def get_logger(name: str, **kwargs) -> StructuredLogger:
    """Get or create a structured logger instance"""
    with _lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(name, **kwargs)
        return _loggers[name]


# Decorator for automatic performance logging
def log_performance(logger_name: str = "performance"):
    """Decorator to automatically log function performance"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.performance(
                    f"{func.__module__}.{func.__name__}",
                    duration,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {duration:.3f}s",
                    exception=e,
                    function=f"{func.__module__}.{func.__name__}"
                )
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo usage
    logger = get_logger("demo")
    
    logger.info("Application started")
    logger.debug("Debug information", user_id="test123")
    
    # Audit event
    audit_event = AuditEvent(
        event_id="",
        timestamp="",
        event_type=EventType.AUTHENTICATION,
        actor="user123",
        action="login",
        resource="api",
        outcome="success",
        metadata={"ip": "127.0.0.1"},
        risk_score=1
    )
    logger.audit(audit_event)
    
    # Security event
    logger.security("Suspicious login attempt", risk_score=8, ip="192.168.1.100")
    
    print("Demo logging completed. Check logs/ directory for output.")