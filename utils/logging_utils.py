"""
Logging configuration utilities.
"""

import logging
import os
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    suppress_warnings: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        suppress_warnings: Whether to suppress warnings (default: False)
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)
    
    # Suppress warnings if requested
    if suppress_warnings:
        import warnings
        warnings.simplefilter('ignore')
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('datasets').setLevel(logging.ERROR)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level {logging.getLevelName(level)}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

