import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

class LossFunction(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2
    ):
        """
        Initialize loss function
        
        Args:
            alpha: Weight for classification loss
            beta: Weight for temporal consistency loss
            gamma: Weight for attention regularization
        """
        super(LossFunction, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Temporal consistency loss
        self.temporal_loss = nn.MSELoss()
        
        # Attention regularization
        self.attention_reg = nn.L1Loss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            attention_weights: Attention weights
            temporal_features: Temporal features
            
        Returns:
            Tuple of (total loss, loss components)
        """
        # Classification loss
        cls_loss = self.classification_loss(predictions, targets)
        
        # Temporal consistency loss
        temp_loss = self.temporal_loss(
            temporal_features[1:],
            temporal_features[:-1]
        )
        
        # Attention regularization
        attn_reg = self.attention_reg(
            attention_weights,
            torch.ones_like(attention_weights) / attention_weights.size(1)
        )
        
        # Total loss
        total_loss = (
            self.alpha * cls_loss +
            self.beta * temp_loss +
            self.gamma * attn_reg
        )
        
        # Loss components
        loss_components = {
            'classification_loss': cls_loss.item(),
            'temporal_loss': temp_loss.item(),
            'attention_reg': attn_reg.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize model trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = LossFunction()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        metrics = {
            'classification_loss': 0,
            'temporal_loss': 0,
            'attention_reg': 0
        }
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions, attention_weights = self.model(inputs)
                
                # Compute loss
                loss, loss_components = self.criterion(
                    predictions,
                    targets,
                    attention_weights,
                    self.model.feature_extractor.lstm_out
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                for k, v in loss_components.items():
                    metrics[k] += v
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': total_loss / (batch_idx + 1)
                })
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(self.train_loader)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        metrics = {
            'classification_loss': 0,
            'temporal_loss': 0,
            'attention_reg': 0
        }
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch_idx, (inputs, targets) in enumerate(pbar):
                    # Move data to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    predictions, attention_weights = self.model(inputs)
                    
                    # Compute loss
                    loss, loss_components = self.criterion(
                        predictions,
                        targets,
                        attention_weights,
                        self.model.feature_extractor.lstm_out
                    )
                    
                    # Update metrics
                    total_loss += loss.item()
                    for k, v in loss_components.items():
                        metrics[k] += v
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'avg_loss': total_loss / (batch_idx + 1)
                    })
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(self.val_loader)
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train model
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_classification_loss': [],
            'val_classification_loss': [],
            'train_temporal_loss': [],
            'val_temporal_loss': [],
            'train_attention_reg': [],
            'val_attention_reg': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch + 1}/{num_epochs}')
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            
            # Save metrics
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['train_classification_loss'].append(train_metrics['classification_loss'])
            history['val_classification_loss'].append(val_metrics['classification_loss'])
            history['train_temporal_loss'].append(train_metrics['temporal_loss'])
            history['val_temporal_loss'].append(val_metrics['temporal_loss'])
            history['train_attention_reg'].append(train_metrics['attention_reg'])
            history['val_attention_reg'].append(val_metrics['attention_reg'])
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info('Early stopping triggered')
                break
        
        return history
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False
    ):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path]
    ):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'] 