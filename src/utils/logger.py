"""
Logging utilities for StructureExtractor-Pretrain.
Handles setting up logging configuration.
"""

import logging
import os
from typing import Optional
import sys


def setup_logger(name: str, 
                 log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 use_stdout: bool = True) -> logging.Logger:
    """
    Set up logger with specified configuration.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_stdout: Whether to log to stdout
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add stdout handler if requested
    if use_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    
    # Add file handler if requested
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_default_logger(config: dict) -> logging.Logger:
    """
    Set up default logger based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger instance
    """
    infrastructure_config = config.get('infrastructure', {})
    logging_config = infrastructure_config.get('logging', {})
    
    log_level = logging_config.get('level', 'INFO')
    log_dir = logging_config.get('log_dir', './logs')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'structure_extractor.log')
    
    return setup_logger('StructureExtractor', log_level, log_file)