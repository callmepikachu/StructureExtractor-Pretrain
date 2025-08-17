"""
Data module for StructureExtractor-Pretrain.
"""

# Expose the main dataset class
from .dataset import ReDocREDDataset
from .optimized_dataset import MemoryEfficientReDocREDDataset, OptimizedReDocREDDataset

__all__ = [
    "ReDocREDDataset",
    "MemoryEfficientReDocREDDataset",
    "OptimizedReDocREDDataset",
]