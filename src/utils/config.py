"""
Configuration utilities for StructureExtractor-Pretrain.
Handles loading and validating configuration files.
"""

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['model', 'data', 'training', 'evaluation', 'infrastructure']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate model configuration
    model_config = config.get('model', {})
    required_model_keys = ['name', 'hidden_dim', 'num_layers']
    for key in required_model_keys:
        if key not in model_config:
            logger.error(f"Missing required model configuration key: {key}")
            return False
    
    # Validate data configuration
    data_config = config.get('data', {})
    required_data_keys = ['dataset_name', 'data_dir']
    for key in required_data_keys:
        if key not in data_config:
            logger.error(f"Missing required data configuration key: {key}")
            return False
    
    # Validate training configuration
    training_config = config.get('training', {})
    required_training_keys = ['batch_size', 'learning_rate', 'num_epochs']
    for key in required_training_keys:
        if key not in training_config:
            logger.error(f"Missing required training configuration key: {key}")
            return False
    
    logger.info("Configuration validation passed")
    return True