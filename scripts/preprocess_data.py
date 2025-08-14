"""
Data preprocessing script for StructureExtractor-Pretrain.
Preprocesses ReDocRED data into chunks for efficient training.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import logging
from typing import Dict, List, Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ReDocREDDataset
from src.utils.logger import setup_default_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess ReDocRED data into chunks")
    
    parser.add_argument(
        "--config",
        type=str,
        default="../config/default_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--input-data",
        type=str,
        default="../data/train_revised.json",
        help="Path to input data file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../output",
        help="Output directory for preprocessed chunks"
    )
    
    parser.add_argument(
        "--split-docs",
        action="store_true",
        help="Whether to split documents into chunks during preprocessing"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess_data_to_chunks(input_path: str, 
                             output_dir: str, 
                             config: Dict[str, Any],
                             split_docs: bool = True) -> None:
    """
    Preprocess data into chunks and save to disk.
    
    Args:
        input_path: Path to input data file
        output_dir: Directory to save preprocessed chunks
        config: Configuration dictionary
        split_docs: Whether to split documents into chunks
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset with lazy loading
    dataset = ReDocREDDataset(
        data_path=input_path,
        max_seq_length=config['data'].get('max_seq_length', 512),
        max_entities=config['data'].get('max_entities', 100),
        max_relations=config['data'].get('max_relations', 50),
        chunk_size=config['data'].get('chunk_size', 3),
        overlap_size=config['data'].get('overlap_size', 1),
        max_documents=config['data'].get('max_documents', None)
    )
    # # Debug info
    # print(f"[DEBUG] Dataset type: {type(dataset)}")
    # print(f"[DEBUG] Dataset length: {len(dataset)}")
    # print(f"[DEBUG] Dataset first item keys: {dataset[0].keys() if len(dataset) > 0 else 'Empty dataset'}")
    # print(f"[DEBUG] First item content sample: {dataset[0] if len(dataset) > 0 else None}")

    # Process and save chunks
    total_chunks = len(dataset)
    chunk_size = 1000  # Save chunks in batches
    chunk_files = []
    import  time
    
    print(f"Processing {total_chunks} chunks...")
    
    for i in range(0, total_chunks, chunk_size):
        batch_start_time = time.time()
        print(f"正在处理第{i}到第{i+chunk_size}个chunk")
        # Process a batch of chunks
        batch_chunks = []
        batch_end = min(i + chunk_size, total_chunks)
        
        for j in range(i, batch_end):
            item_start = time.time()

            try:
                chunk_data = dataset[j]
                batch_chunks.append(chunk_data)
            except Exception as e:
                print(f"Error processing chunk {j}: {e}")
                continue
            item_load_time = time.time() - item_start

        # Save batch to file
        save_start = time.time()
        batch_file = os.path.join(output_dir, f"chunks_{i//chunk_size:04d}.json")
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_chunks, f, ensure_ascii=False, indent=2)
        
        chunk_files.append(batch_file)
        save_time = time.time() - save_start
        print(f"[DEBUG] 保存 {batch_file} 耗时: {save_time:.4f} 秒")

        print(f"Saved chunks {i} to {batch_end-1} to {batch_file}")
    
    # Save chunk index
    index_file = os.path.join(output_dir, "chunk_index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({
            'chunk_files': chunk_files,
            'total_chunks': total_chunks,
            'chunk_size': chunk_size
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Preprocessing completed. Index saved to {index_file}")


def main():
    """Main preprocessing function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_default_logger(config)
    logger.info("Starting data preprocessing")
    
    # Preprocess data
    preprocess_data_to_chunks(
        input_path=args.input_data,
        output_dir=args.output_dir,
        config=config,
        split_docs=args.split_docs
    )
    
    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()