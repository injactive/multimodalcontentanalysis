"""
Structured logging utility for the Multi-Modal Content Analysis API.

This module provides structured logging with JSON output, correlation IDs,
and performance monitoring capabilities.

Author: Christian Kruschel
Version: 0.0.1
"""

import logging
import sys
import time
from typing import Any, Dict, Optional
from datetime import datetime
import json
import uuid
from functools import wraps

import structlog
from structlog.stdlib import LoggerFactory


class CorrelationIDProcessor:
    """Processor to add correlation IDs to log records."""
    
    def __init__(self):
        self._correlation_id: Optional[str] = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current context."""
        self._correlation_id = correlation_id
    
    def clear_correlation_id(self):
        """Clear correlation ID."""
        self._correlation_id = None
    
    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID to event dict."""
        if self._correlation_id:
            event_dict['correlation_id'] = self._correlation_id
        return event_dict


class PerformanceProcessor:
    """Processor to add performance metrics to log records."""
    
    def __init__(self):
        self._start_times: Dict[str, float] = {}
    
    def start_timer(self, timer_name: str):
        """Start a performance timer."""
        self._start_times[timer_name] = time.time()
    
    def end_timer(self, timer_name: str) -> Optional[float]:
        """End a performance timer and return duration."""
        if timer_name in self._start_times:
            duration = time.time() - self._start_times[timer_name]
            del self._start_times[timer_name]
            return duration
        return None
    
    def __call__(self, logger, method_name, event_dict):
        """Add performance metrics to event dict."""
        # Add timestamp
        event_dict['timestamp'] = datetime.utcnow().isoformat()
        
        # Add process info if available
        event_dict['process_id'] = id(self)
        
        return event_dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("json" or "text")
        log_file: Optional log file path
    """
    # Configure structlog
    correlation_processor = CorrelationIDProcessor()
    performance_processor = PerformanceProcessor()
    
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        correlation_processor,
        performance_processor,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if format_type == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())  # Always use JSON for files
        root_logger.addHandler(file_handler)
    
    # Store processors for later use
    root_logger._correlation_processor = correlation_processor
    root_logger._performance_processor = performance_processor


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def with_correlation_id(correlation_id: Optional[str] = None):
    """
    Decorator to add correlation ID to all log messages in a function.
    
    Args:
        correlation_id: Optional correlation ID (generates one if not provided)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cid = correlation_id or str(uuid.uuid4())
            
            # Get correlation processor
            root_logger = logging.getLogger()
            if hasattr(root_logger, '_correlation_processor'):
                root_logger._correlation_processor.set_correlation_id(cid)
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                if hasattr(root_logger, '_correlation_processor'):
                    root_logger._correlation_processor.clear_correlation_id()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cid = correlation_id or str(uuid.uuid4())
            
            # Get correlation processor
            root_logger = logging.getLogger()
            if hasattr(root_logger, '_correlation_processor'):
                root_logger._correlation_processor.set_correlation_id(cid)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if hasattr(root_logger, '_correlation_processor'):
                    root_logger._correlation_processor.clear_correlation_id()
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_performance_logging(operation_name: str):
    """
    Decorator to add performance logging to functions.
    
    Args:
        operation_name: Name of the operation for logging
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Start timer
            root_logger = logging.getLogger()
            if hasattr(root_logger, '_performance_processor'):
                root_logger._performance_processor.start_timer(operation_name)
            
            start_time = time.time()
            
            try:
                logger.info(f"Starting {operation_name}")
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(
                    f"Completed {operation_name}",
                    duration=duration,
                    status="success"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed {operation_name}",
                    duration=duration,
                    status="error",
                    error=str(e)
                )
                raise
            finally:
                # End timer
                if hasattr(root_logger, '_performance_processor'):
                    root_logger._performance_processor.end_timer(operation_name)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Start timer
            root_logger = logging.getLogger()
            if hasattr(root_logger, '_performance_processor'):
                root_logger._performance_processor.start_timer(operation_name)
            
            start_time = time.time()
            
            try:
                logger.info(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(
                    f"Completed {operation_name}",
                    duration=duration,
                    status="success"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed {operation_name}",
                    duration=duration,
                    status="error",
                    error=str(e)
                )
                raise
            finally:
                # End timer
                if hasattr(root_logger, '_performance_processor'):
                    root_logger._performance_processor.end_timer(operation_name)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Initialize logging with default settings
setup_logging()

