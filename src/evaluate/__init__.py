"""
Evaluation module for StructureExtractor-Pretrain.
"""

# Expose the main evaluation function
from .evaluate_adapted import evaluate_model

__all__ = [
    "evaluate_model",
]