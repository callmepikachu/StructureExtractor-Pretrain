"""
Evaluation utilities for StructureExtractor-Pretrain.
Implements model evaluation metrics and evaluation procedures.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

logger = logging.getLogger(__name__)


def evaluate_model(model, 
                   dataset, 
                   config: Dict[str, Any],
                   batch_size: int = 16) -> Dict[str, float]:
    """
    Evaluate model on dataset.
    
    Args:
        model: StructureExtractor model to evaluate
        dataset: Dataset to evaluate on
        config: Configuration dictionary
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    evaluation_config = config.get('evaluation', {})
    metrics = evaluation_config.get('metrics', ['precision', 'recall', 'f1', 'accuracy'])
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_evaluation_collate_fn
    )
    
    # Initialize tracking variables
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch['texts'][0])  # Simplified - process one text at a time
            
            # Compute loss (placeholder)
            loss = _compute_evaluation_loss(outputs, batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions and labels (simplified)
            # In practice, this would depend on the specific task
            predictions = outputs.get('entity_logits', torch.empty(0))
            labels = torch.zeros_like(predictions)  # Placeholder
            
            if predictions.numel() > 0:
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    results = {}
    
    # Average loss
    results['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute specified metrics
    if all_predictions and all_labels:
        predictions_array = np.array(all_predictions)
        labels_array = np.array(all_labels)
        
        # For classification tasks, convert to class predictions
        if predictions_array.ndim > 1:
            pred_classes = np.argmax(predictions_array, axis=1)
            label_classes = np.argmax(labels_array, axis=1) if labels_array.ndim > 1 else labels_array
        else:
            # For binary classification or regression, use threshold
            pred_classes = (predictions_array > 0.5).astype(int)
            label_classes = (labels_array > 0.5).astype(int)
        
        # Compute metrics
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(label_classes, pred_classes)
        
        if 'precision' in metrics or 'recall' in metrics or 'f1' in metrics:
            precision, recall, f1, _ = precision_recall_fscore_support(
                label_classes, pred_classes, average='weighted'
            )
            
            if 'precision' in metrics:
                results['precision'] = precision
            if 'recall' in metrics:
                results['recall'] = recall
            if 'f1' in metrics:
                results['f1'] = f1
    
    return results


def _evaluation_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function for evaluation."""
    # This is a simple implementation - can be enhanced based on needs
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'entities': [item['entities'] for item in batch],
        'relations': [item['relations'] for item in batch],
        'texts': [item['text'] for item in batch]
    }


def _compute_evaluation_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
    """
    Compute evaluation loss.
    
    Args:
        outputs: Model outputs
        batch: Batch data
        
    Returns:
        Computed loss tensor
    """
    # Extract model outputs
    entity_logits = outputs.get('entity_logits', torch.empty(0))
    relation_logits = outputs.get('relation_logits', torch.empty(0))
    entities = outputs.get('entities', [])
    relations = outputs.get('relations', [])
    
    # Initialize loss
    device = entity_logits.device if entity_logits.numel() > 0 else torch.device('cpu')
    loss = torch.tensor(0.0, device=device)
    
    # Entity extraction loss
    if entity_logits.numel() > 0 and 'entities' in batch and batch['entities']:
        # Create entity targets from batch data
        entity_targets = _create_entity_targets(entities, batch['entities'], device)
        if entity_targets is not None and entity_targets.numel() > 0:
            # Use cross-entropy loss for entity classification
            if entity_logits.shape[0] == entity_targets.shape[0]:
                loss = loss + torch.nn.functional.cross_entropy(entity_logits, entity_targets)
    
    # Relation extraction loss
    if relation_logits.numel() > 0 and 'relations' in batch and batch['relations']:
        # Create relation targets from batch data
        relation_targets = _create_relation_targets(relations, batch['relations'], device)
        if relation_targets is not None and relation_targets.numel() > 0:
            # Use cross-entropy loss for relation classification
            if relation_logits.shape[0] == relation_targets.shape[0]:
                loss = loss + torch.nn.functional.cross_entropy(relation_logits, relation_targets)
    
    return loss if loss.requires_grad else loss.detach().requires_grad_(True)


def _create_entity_targets(predicted_entities: List[Dict], batch_entities: List[List[Dict]], device: torch.device) -> Optional[torch.Tensor]:
    """
    Create entity targets from batch data.
    
    Args:
        predicted_entities: Predicted entities from model
        batch_entities: Ground truth entities from batch
        device: Device to put the tensor on
        
    Returns:
        Entity target tensor or None if no targets
    """
    if not predicted_entities or not batch_entities:
        return None
        
    # Flatten batch entities
    flat_batch_entities = [entity for doc_entities in batch_entities for entity in doc_entities]
    
    # For simplicity, we'll create a simple matching based on text
    # In practice, this would be more sophisticated
    targets = []
    for pred_entity in predicted_entities:
        pred_text = pred_entity.get('text', '').lower()
        matched = False
        
        for batch_entity in flat_batch_entities:
            batch_text = batch_entity.get('text', '').lower()
            if pred_text == batch_text:
                # Use entity type as target class (simplified)
                entity_type = batch_entity.get('type', 'ENTITY')
                # Convert to index (placeholder - in practice, use a proper mapping)
                target_idx = hash(entity_type) % 100  # Assuming 100 classes
                targets.append(target_idx)
                matched = True
                break
        
        if not matched:
            targets.append(0)  # Default class for unmatched entities
    
    if targets:
        return torch.tensor(targets, device=device)
    return None


def _create_relation_targets(predicted_relations: List[Dict], batch_relations: List[List[Dict]], device: torch.device) -> Optional[torch.Tensor]:
    """
    Create relation targets from batch data.
    
    Args:
        predicted_relations: Predicted relations from model
        batch_relations: Ground truth relations from batch
        device: Device to put the tensor on
        
    Returns:
        Relation target tensor or None if no targets
    """
    if not predicted_relations or not batch_relations:
        return None
        
    # Flatten batch relations
    flat_batch_relations = [relation for doc_relations in batch_relations for relation in doc_relations]
    
    # For simplicity, we'll create a simple matching based on head-tail pairs
    # In practice, this would be more sophisticated
    targets = []
    for pred_relation in predicted_relations:
        pred_head = pred_relation.get('head', '')
        pred_tail = pred_relation.get('tail', '')
        matched = False
        
        for batch_relation in flat_batch_relations:
            batch_head = batch_relation.get('head', '')
            batch_tail = batch_relation.get('tail', '')
            
            # Match based on head and tail entity IDs
            if pred_head == batch_head and pred_tail == batch_tail:
                # Use relation type as target class (simplified)
                relation_type = batch_relation.get('type', 'RELATED_TO')
                # Convert to index (placeholder - in practice, use a proper mapping)
                target_idx = hash(relation_type) % 50  # Assuming 50 relation types
                targets.append(target_idx)
                matched = True
                break
        
        if not matched:
            targets.append(0)  # Default class for unmatched relations
    
    if targets:
        return torch.tensor(targets, device=device)
    return None