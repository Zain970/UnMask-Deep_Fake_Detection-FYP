import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from datetime import datetime

class FineTuner:
    """
    A class for fine-tuning the UnMask model with transfer learning and domain adaptation capabilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Union[str, Path],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        freeze_layers: Optional[List[str]] = None,
        domain_adaptation: bool = False
    ):
        """
        Initialize the fine-tuning pipeline.
        
        Args:
            model: The model to fine-tune
            device: Device to run the model on
            output_dir: Directory to save fine-tuned models and logs
            learning_rate: Learning rate for fine-tuning
            weight_decay: Weight decay for optimizer
            freeze_layers: List of layer names to freeze during fine-tuning
            domain_adaptation: Whether to use domain adaptation
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_layers = freeze_layers or []
        self.domain_adaptation = domain_adaptation
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'fine_tuning.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Domain adaptation components
        if domain_adaptation:
            self.domain_classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            ).to(device)
            self.domain_optimizer = optim.Adam(
                self.domain_classifier.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Freeze specified layers
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in self.freeze_layers):
                param.requires_grad = False
        
        # Get trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
    
    def _domain_adaptation_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute domain adaptation loss using Maximum Mean Discrepancy (MMD).
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            
        Returns:
            Domain adaptation loss
        """
        # Compute MMD loss
        source_mean = source_features.mean(0)
        target_mean = target_features.mean(0)
        mmd_loss = torch.norm(source_mean - target_mean, p=2)
        
        return mmd_loss
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        source_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            source_loader: DataLoader for source domain data (for domain adaptation)
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Domain adaptation
            if self.domain_adaptation and source_loader is not None:
                try:
                    source_inputs, _ = next(iter(source_loader))
                    source_inputs = source_inputs.to(self.device)
                    
                    # Get features
                    source_features = self.model.get_features(source_inputs)
                    target_features = self.model.get_features(inputs)
                    
                    # Compute domain loss
                    domain_loss = self._domain_adaptation_loss(
                        source_features,
                        target_features
                    )
                    
                    # Combine losses
                    loss = loss + 0.1 * domain_loss
                except StopIteration:
                    source_loader = iter(source_loader)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Compute metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def _validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        source_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: int = 5
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            source_loader: DataLoader for source domain data
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary of training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Train
            train_metrics = self._train_epoch(train_loader, source_loader)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self.logger.info(
                f'Epoch {epoch + 1}/{epochs} - '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Train Acc: {train_metrics["accuracy"]:.2f}%, '
                f'Val Loss: {val_metrics["loss"]:.4f}, '
                f'Val Acc: {val_metrics["accuracy"]:.2f}%'
            )
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info('Early stopping triggered')
                    break
        
        return history
    
    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.domain_adaptation:
            checkpoint['domain_classifier_state_dict'] = self.domain_classifier.state_dict()
        
        torch.save(
            checkpoint,
            self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        )
        
        # Save training info
        info = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'freeze_layers': self.freeze_layers,
            'domain_adaptation': self.domain_adaptation
        }
        
        with open(self.output_dir / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=4)
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path]
    ) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.domain_adaptation and 'domain_classifier_state_dict' in checkpoint:
            self.domain_classifier.load_state_dict(
                checkpoint['domain_classifier_state_dict']
            )
        
        self.logger.info(f'Loaded checkpoint from {checkpoint_path}') 