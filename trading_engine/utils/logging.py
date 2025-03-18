"""
Logging utilities for the trading engine.
"""
import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(config):
    """
    Set up logging for the trading engine.
    
    Args:
        config (dict): Logging configuration
    """
    # Get configuration values with defaults
    log_level = config.get("level", "INFO").upper()
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("file", "logs/trading_engine.log")
    max_size = config.get("max_size", 10 * 1024 * 1024)  # 10 MB
    backup_count = config.get("backup_count", 5)
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress logs from third-party libraries
    for module in ["matplotlib", "alpaca_trade_api", "urllib3", "requests"]:
        logging.getLogger(module).setLevel(logging.WARNING)


def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)