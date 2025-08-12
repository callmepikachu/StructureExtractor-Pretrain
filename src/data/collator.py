"""
Data collator for StructureExtractor-Pretrain.
Handles batching and preprocessing of ReDocRED data.
"""

import torch
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class DataCollator:
    """
    Data collator for ReDocRED dataset.
    Handles dynamic padding and batch creation.
    """
    
    tokenizer_name: str = "bert-base-uncased"
    max_seq_length: int = 512
    pad_to_multiple_of: int = 8
    
    def __post_init__(self):
        """Initialize tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Batched dictionary of tensors
        """
        # Extract text inputs
        texts = [example['text'] for example in examples]
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of if self.pad_to_multiple_of else None
        )
        
        # Prepare batch dictionary
        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        
        # Add entity information
        batch["entities"] = [example["entities"] for example in examples]
        
        # Add relation information
        batch["relations"] = [example["relations"] for example in examples]
        
        # Add document and chunk information
        batch["doc_ids"] = [example["doc_id"] for example in examples]
        batch["chunk_indices"] = torch.tensor([example["chunk_idx"] for example in examples])
        
        return batch


# Convenience function for backward compatibility
def get_data_collator(config: Dict[str, Any]) -> DataCollator:
    """
    Get data collator based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataCollator instance
    """
    data_config = config.get("data", {})
    
    return DataCollator(
        tokenizer_name="bert-base-uncased",
        max_seq_length=data_config.get("max_seq_length", 512),
        pad_to_multiple_of=8
    )