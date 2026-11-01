import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging
def setup_logging():
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Create handlers
    # File handler for all logs
    all_logs_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'all.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    all_logs_handler.setLevel(logging.DEBUG)
    all_logs_handler.setFormatter(file_formatter)

    # File handler for errors
    error_logs_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'error.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_logs_handler.setLevel(logging.ERROR)
    error_logs_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(all_logs_handler)
    root_logger.addHandler(error_logs_handler)
    root_logger.addHandler(console_handler)

    # Create logger for performance monitoring
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'performance.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    perf_handler.setFormatter(file_formatter)
    perf_logger.addHandler(perf_handler)

    return root_logger, perf_logger

# Initialize loggers
logger, perf_logger = setup_logging() 