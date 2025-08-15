"""
Main training and evaluation script for StructureExtractor-Pretrain.
Handles the complete training and evaluation pipeline.
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
from src.evaluate.evaluate_adapted import evaluate_model
from src.utils.config import load_config, validate_config
from src.utils.logger import setup_default_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate StructureExtractor model")
    
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
        "--test-data",
        type=str,
        default=None,
        help="Path to test data file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints, logs, and results"
    )
    
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate the model (requires --checkpoint)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation"
    )
    
    return parser.parse_args()


def train_model(args, config: Dict[str, Any]):
    """Train the model."""
    # Set up logging
    logger = setup_default_logger(config)
    logger.info("Starting StructureExtractor training")
    
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
    
    return model


def evaluate_model_from_checkpoint(args, config: Dict[str, Any]):
    """Evaluate model from checkpoint."""
    # Set up logging
    logger = setup_default_logger(config)
    logger.info("Starting StructureExtractor evaluation")
    
    # Update paths in config if output directory is specified
    if args.output_dir:
        config['paths']['log_dir'] = os.path.join(args.output_dir, 'logs')
        config['infrastructure']['logging']['log_dir'] = os.path.join(args.output_dir, 'logs')
    
    # Create model
    logger.info("Creating StructureExtractor model")
    model = StructureExtractor(config)
    
    # Load model checkpoint
    logger.info(f"Loading model checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load test dataset
    logger.info("Loading test data")
    test_dataset = ReDocREDDataset(
        data_path=args.test_data,
        max_seq_length=config['data'].get('max_seq_length', 512),
        max_entities=config['data'].get('max_entities', 100),
        max_relations=config['data'].get('max_relations', 50)
    )
    
    # Evaluate model
    logger.info("Evaluating model")
    results = evaluate_model(
        model=model,
        dataset=test_dataset,
        config=config,
        batch_size=config['evaluation'].get('batch_size', 1)
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save results if output directory is specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            for metric, value in results.items():
                f.write(f"{metric}: {value:.4f}\n")
        logger.info(f"Evaluation results saved to {results_file}")
    
    logger.info("Evaluation completed successfully")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Configuration validation failed")
    
    # Handle evaluation only mode
    if args.evaluate_only:
        if not args.checkpoint or not args.test_data:
            raise ValueError("--checkpoint and --test-data are required for evaluation")
        evaluate_model_from_checkpoint(args, config)
        return
    
    # Train model
    model = train_model(args, config)
    
    # Evaluate on test set if provided
    if args.test_data:
        logger = setup_default_logger(config)
        logger.info("Loading test data for final evaluation")
        
        # Load test dataset
        test_dataset = ReDocREDDataset(
            data_path=args.test_data,
            max_seq_length=config['data'].get('max_seq_length', 512),
            max_entities=config['data'].get('max_entities', 100),
            max_relations=config['data'].get('max_relations', 50)
        )
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Evaluate model
        logger.info("Evaluating model on test set")
        results = evaluate_model(
            model=model,
            dataset=test_dataset,
            config=config,
            batch_size=config['evaluation'].get('batch_size', 1)
        )
        
        # Print results
        logger.info("Test Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save results if output directory is specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            results_file = os.path.join(args.output_dir, 'test_results.txt')
            with open(results_file, 'w') as f:
                for metric, value in results.items():
                    f.write(f"{metric}: {value:.4f}\n")
            logger.info(f"Test results saved to {results_file}")


if __name__ == "__main__":
    main()