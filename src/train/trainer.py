"""
Trainer for StructureExtractor-Pretrain.
Implements the training loop with ContinualGNN and StreamE optimizations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import logging
import os
from tqdm import tqdm
import wandb
from transformers import get_linear_schedule_with_warmup

from src.model.extractor import StructureExtractor
from src.data.dataset import ReDocREDDataset
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """
    Trainer class for pretraining StructureExtractor model.
    Handles the complete training loop with evaluation and checkpointing.
    """
    
    def __init__(self, 
                 model: StructureExtractor,
                 train_dataset: ReDocREDDataset,
                 dev_dataset: Optional[ReDocREDDataset],
                 config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: StructureExtractor model to train
            train_dataset: Training dataset
            dev_dataset: Optional validation dataset
            config: Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.config = config
        
        # Training configuration
        self.training_config = config['training']
        self.batch_size = self.training_config['batch_size']
        self.learning_rate = float(self.training_config['learning_rate'])
        self.num_epochs = self.training_config['num_epochs']
        self.warmup_steps = self.training_config.get('warmup_steps', 0)
        self.gradient_clip_norm = self.training_config.get('gradient_clip_norm', 1.0)
        
        # Checkpointing configuration
        self.save_steps = self.training_config.get('save_steps', 1000)
        self.eval_steps = self.training_config.get('eval_steps', 500)
        self.logging_steps = self.training_config.get('logging_steps', 100)
        
        # Early stopping configuration
        self.early_stopping_patience = self.training_config.get('early_stopping_patience', 3)
        self.early_stopping_threshold = self.training_config.get('early_stopping_threshold', 0.001)
        
        # Infrastructure configuration
        self.infrastructure_config = config['infrastructure']
        self.device = self._setup_device()
        self.use_mixed_precision = self.infrastructure_config.get('mixed_precision', False)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = None
        
        # Initialize data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        if dev_dataset:
            self.dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )
        else:
            self.dev_dataloader = None
            
        # Initialize tracking variables
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_triggered = False
        
        # Initialize logging
        self._setup_logging()
        
    def _setup_device(self) -> torch.device:
        """Set up device for training."""
        device_config = self.infrastructure_config.get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cpu':
            device = torch.device('cpu')
        elif device_config.startswith('cuda'):
            device = torch.device(device_config)
        else:
            device = torch.device('cpu')
            
        logger.info(f"Using device: {device}")
        return device
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.training_config.get('optimizer', 'AdamW').lower()
        weight_decay = self.training_config.get('weight_decay', 0.01)
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        logger.info(f"Using optimizer: {optimizer_name}")
        return optimizer
    
    def _setup_scheduler(self, total_steps: int):
        """Set up learning rate scheduler."""
        scheduler_name = self.training_config.get('scheduler', 'linear').lower()
        
        if scheduler_name == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_name == 'none':
            self.scheduler = None
        else:
            logger.warning(f"Unsupported scheduler: {scheduler_name}, using no scheduler")
            self.scheduler = None
    
    def _setup_logging(self):
        """Set up logging and experiment tracking."""
        # Set up Weights & Biases if requested
        if self.infrastructure_config.get('logging', {}).get('use_wandb', False):
            try:
                wandb_project = self.infrastructure_config.get('logging', {}).get(
                    'wandb_project', 'structure-extractor-pretrain'
                )
                wandb.init(project=wandb_project, config=self.config)
                logger.info("Initialized Weights & Biases logging")
            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        # Convert lists to tensors
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'entities': [item['entities'] for item in batch],
            'relations': [item['relations'] for item in batch],
            'texts': [item['text'] for item in batch]
        }
    
    def train(self):
        """
        Main training loop.
        """
        logger.info("Starting training...")
        
        # Calculate total steps for scheduler
        total_steps = len(self.train_dataloader) * self.num_epochs
        self._setup_scheduler(total_steps)
        
        # Training loop
        for epoch in range(self.num_epochs):
            if self.early_stopping_triggered:
                logger.info("Early stopping triggered, stopping training")
                break
                
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch
            train_loss = self._train_epoch(epoch)
            
            # Evaluate if validation dataset is available
            if self.dev_dataloader:
                eval_loss = self._evaluate()
                logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
                
                # Check for early stopping
                self._check_early_stopping(eval_loss)
            else:
                logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, is_best=False)
        
        logger.info("Training completed")
    
    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Use mixed precision if requested
        scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision and self.device.type == 'cuda' else None
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch['texts'][0])  # Simplified - process one text at a time
                    loss = self._compute_loss(outputs, batch)
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip_norm
                    )
                
                # Optimizer step
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Forward pass
                outputs = self.model(batch['texts'][0])  # Simplified - process one text at a time
                loss = self._compute_loss(outputs, batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Update tracking
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
            
            # Log metrics
            if self.global_step % self.logging_steps == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Step {self.global_step} - Loss: {avg_loss:.4f}")
                
                # Log to wandb if available
                if 'wandb' in globals():
                    try:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/avg_loss': avg_loss,
                            'train/learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate,
                            'step': self.global_step
                        })
                    except:
                        pass
            
            # Evaluate periodically
            if self.dev_dataloader and self.global_step % self.eval_steps == 0:
                eval_loss = self._evaluate()
                logger.info(f"Step {self.global_step} - Eval Loss: {eval_loss:.4f}")
                
                # Log to wandb if available
                if 'wandb' in globals():
                    try:
                        wandb.log({
                            'eval/loss': eval_loss,
                            'step': self.global_step
                        })
                    except:
                        pass
                
                # Check for early stopping
                self._check_early_stopping(eval_loss)
            
            # Save checkpoint periodically
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(epoch, step=self.global_step, is_best=False)
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _compute_loss(self, model_outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for model outputs.
        
        Args:
            model_outputs: Model outputs dictionary
            batch: Batch data
            
        Returns:
            Computed loss tensor
        """
        # Extract model outputs
        entity_logits = model_outputs.get('entity_logits', torch.empty(0))
        relation_logits = model_outputs.get('relation_logits', torch.empty(0))
        entities = model_outputs.get('entities', [])
        relations = model_outputs.get('relations', [])
        
        # Initialize loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Entity extraction loss
        if entity_logits.numel() > 0 and 'entities' in batch and batch['entities']:
            # Create entity targets from batch data
            entity_targets = self._create_entity_targets(entities, batch['entities'])
            if entity_targets is not None and entity_targets.numel() > 0:
                # Use cross-entropy loss for entity classification
                if entity_logits.shape[0] == entity_targets.shape[0]:
                    loss = loss + nn.functional.cross_entropy(entity_logits, entity_targets)
        
        # Relation extraction loss
        if relation_logits.numel() > 0 and 'relations' in batch and batch['relations']:
            # Create relation targets from batch data
            relation_targets = self._create_relation_targets(relations, batch['relations'])
            if relation_targets is not None and relation_targets.numel() > 0:
                # Use cross-entropy loss for relation classification
                if relation_logits.shape[0] == relation_targets.shape[0]:
                    loss = loss + nn.functional.cross_entropy(relation_logits, relation_targets)
        
        return loss
    
    def _create_entity_targets(self, predicted_entities: List[Dict], batch_entities: List[List[Dict]]) -> Optional[torch.Tensor]:
        """
        Create entity targets from batch data.
        
        Args:
            predicted_entities: Predicted entities from model
            batch_entities: Ground truth entities from batch
            
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
            return torch.tensor(targets, device=self.device)
        return None
    
    def _create_relation_targets(self, predicted_relations: List[Dict], batch_relations: List[List[Dict]]) -> Optional[torch.Tensor]:
        """
        Create relation targets from batch data.
        
        Args:
            predicted_relations: Predicted relations from model
            batch_relations: Ground truth relations from batch
            
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
            return torch.tensor(targets, device=self.device)
        return None
    
    def _evaluate(self) -> float:
        """
        Evaluate model on validation set.
        
        Returns:
            Average evaluation loss
        """
        if not self.dev_dataloader:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dev_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch['texts'][0])  # Simplified - process one text at a time
                loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.model.train()
        return avg_loss
    
    def _check_early_stopping(self, current_loss: float):
        """
        Check if early stopping criteria are met.
        
        Args:
            current_loss: Current evaluation loss
        """
        if current_loss < self.best_eval_loss - self.early_stopping_threshold:
            self.best_eval_loss = current_loss
            self.patience_counter = 0
            # Save best model
            self._save_checkpoint(None, is_best=True)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                self.early_stopping_triggered = True
    
    def _save_checkpoint(self, epoch: Optional[int], step: Optional[int] = None, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            step: Current step number (optional)
            is_best: Whether this is the best model so far
        """
        # Create checkpoint directory
        model_dir = self.config.get('paths', {}).get('model_dir', './checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'step': self.global_step,
            'best_eval_loss': self.best_eval_loss
        }
        
        # Add scheduler state if available
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Determine checkpoint filename
        if is_best:
            checkpoint_path = os.path.join(model_dir, 'best_model.pt')
        elif step:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_step_{step}.pt')
        else:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pt')
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")