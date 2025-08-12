"""
Main training script for StructureExtractor-Pretrain.
Handles command-line arguments and starts the training process.
"""

import argparse
import os
import sys
import torch
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.extractor import StructureExtractor
from src.data.dataset import ReDocREDDataset
from src.train.trainer import PretrainTrainer
from src.utils.config import load_config, validate_config
from src.utils.logger import setup_default_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train StructureExtractor model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data file"
    )
    
    parser.add_argument(
        "--dev-data",
        type=str,
        default=None,
        help="Path to validation data file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Configuration validation failed")
    
    # Set up logging
    logger = setup_default_logger(config)
    logger.info("Starting StructureExtractor pretraining")
    
    # Update paths in config if output directory is specified
    if args.output_dir:
        config['paths']['model_dir'] = os.path.join(args.output_dir, 'checkpoints')
        config['paths']['log_dir'] = os.path.join(args.output_dir, 'logs')
        config['infrastructure']['logging']['log_dir'] = os.path.join(args.output_dir, 'logs')
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior if requested
    if config.get('deterministic', True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create model
    logger.info("Creating StructureExtractor model")
    model = StructureExtractor(config)
    
    # Load datasets
    logger.info("Loading training data")
    train_dataset = ReDocREDDataset(
        data_path=args.train_data,
        max_seq_length=config['data'].get('max_seq_length', 512),
        max_entities=config['data'].get('max_entities', 100),
        max_relations=config['data'].get('max_relations', 50)
    )
    
    dev_dataset = None
    if args.dev_data:
        logger.info("Loading validation data")
        dev_dataset = ReDocREDDataset(
            data_path=args.dev_data,
            max_seq_length=config['data'].get('max_seq_length', 512),
            max_entities=config['data'].get('max_entities', 100),
            max_relations=config['data'].get('max_relations', 50)
        )
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = PretrainTrainer(
        model=model,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        config=config
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()