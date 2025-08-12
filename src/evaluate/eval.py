"""
Evaluation script for StructureExtractor-Pretrain.
Handles model evaluation on test dataset.
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
from src.evaluate.evaluate import evaluate_model
from src.utils.config import load_config, validate_config
from src.utils.logger import setup_default_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate StructureExtractor model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Configuration validation failed")
    
    # Set up logging
    logger = setup_default_logger(config)
    logger.info("Starting StructureExtractor evaluation")
    
    # Update paths in config if output directory is specified
    if args.output_dir:
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
        batch_size=config['evaluation'].get('batch_size', 16)
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


if __name__ == "__main__":
    main()