import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
import json

class AdversarialRobustness:
    """
    A class for implementing adversarial robustness techniques including
    adversarial training, defense mechanisms, and robustness testing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Union[str, Path],
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_steps: int = 10,
        defense_type: str = 'adversarial_training'
    ):
        """
        Initialize the adversarial robustness pipeline.
        
        Args:
            model: The model to make robust
            device: Device to run the model on
            output_dir: Directory to save robust models and logs
            epsilon: Maximum perturbation size
            alpha: Step size for PGD attack
            num_steps: Number of steps for PGD attack
            defense_type: Type of defense mechanism to use
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.defense_type = defense_type
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'adversarial_robustness.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # Initialize defense mechanisms
        self._setup_defense()
    
    def _setup_defense(self) -> None:
        """Setup defense mechanisms based on defense type."""
        if self.defense_type == 'adversarial_training':
            self.defense = self._adversarial_training
        elif self.defense_type == 'gradient_masking':
            self.defense = self._gradient_masking
        elif self.defense_type == 'feature_squeezing':
            self.defense = self._feature_squeezing
        else:
            raise ValueError(f'Unknown defense type: {self.defense_type}')
    
    def _pgd_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False
    ) -> torch.Tensor:
        """
        Perform PGD attack on the input images.
        
        Args:
            images: Input images
            labels: True labels
            targeted: Whether to perform targeted attack
            
        Returns:
            Adversarial examples
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Initialize perturbation
        delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        
        for _ in range(self.num_steps):
            delta.requires_grad = True
            
            # Forward pass
            outputs = self.model(images + delta)
            loss = F.cross_entropy(outputs, labels)
            
            if targeted:
                loss = -loss
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            grad = delta.grad.detach()
            delta = delta + self.alpha * grad.sign()
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = delta.detach()
        
        return images + delta
    
    def _adversarial_training(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform adversarial training.
        
        Args:
            images: Input images
            labels: True labels
            
        Returns:
            Tuple of (adversarial examples, clean examples)
        """
        # Generate adversarial examples
        adv_images = self._pgd_attack(images, labels)
        
        # Mix clean and adversarial examples
        mixed_images = torch.cat([images, adv_images], dim=0)
        mixed_labels = torch.cat([labels, labels], dim=0)
        
        return mixed_images, mixed_labels
    
    def _gradient_masking(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gradient masking defense.
        
        Args:
            images: Input images
            
        Returns:
            Defended images
        """
        # Apply random noise
        noise = torch.randn_like(images) * 0.01
        defended_images = images + noise
        
        # Apply smoothing
        defended_images = F.avg_pool2d(defended_images, kernel_size=3, stride=1, padding=1)
        
        return defended_images
    
    def _feature_squeezing(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply feature squeezing defense.
        
        Args:
            images: Input images
            
        Returns:
            Defended images
        """
        # Reduce color depth
        defended_images = torch.round(images * 255 / 32) * 32 / 255
        
        # Apply median filtering
        defended_images = F.avg_pool2d(defended_images, kernel_size=3, stride=1, padding=1)
        
        return defended_images
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train for one epoch with adversarial robustness.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for training
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Apply defense mechanism
            defended_images, defended_labels = self.defense(images, labels)
            
            # Forward pass
            outputs = self.model(defended_images)
            loss = F.cross_entropy(outputs, defended_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += defended_labels.size(0)
            correct += predicted.eq(defended_labels).sum().item()
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def _test_robustness(
        self,
        test_loader: DataLoader,
        attack_type: str = 'pgd'
    ) -> Dict[str, float]:
        """
        Test model robustness against adversarial attacks.
        
        Args:
            test_loader: DataLoader for test data
            attack_type: Type of attack to test against
            
        Returns:
            Dictionary of robustness metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Generate adversarial examples
            if attack_type == 'pgd':
                adv_images = self._pgd_attack(images, labels)
            else:
                raise ValueError(f'Unknown attack type: {attack_type}')
            
            # Apply defense mechanism
            defended_images = self.defense(adv_images, labels)[0]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(defended_images)
                loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        metrics = {
            'loss': total_loss / len(test_loader),
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def train_robust_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.001
    ) -> Dict[str, List[float]]:
        """
        Train a robust model using adversarial training.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            learning_rate: Learning rate for training
            
        Returns:
            Dictionary of training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Train
            train_metrics = self._train_epoch(train_loader, optimizer)
            
            # Validate
            val_metrics = self._test_robustness(val_loader)
            
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
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics)
        
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
            'metrics': metrics,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'num_steps': self.num_steps,
            'defense_type': self.defense_type
        }
        
        torch.save(
            checkpoint,
            self.output_dir / f'robust_model_epoch_{epoch}.pt'
        )
        
        # Save training info
        info = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'num_steps': self.num_steps,
            'defense_type': self.defense_type
        }
        
        with open(self.output_dir / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=4)
    
    def load_robust_model(
        self,
        checkpoint_path: Union[str, Path]
    ) -> None:
        """
        Load robust model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.alpha = checkpoint['alpha']
        self.num_steps = checkpoint['num_steps']
        self.defense_type = checkpoint['defense_type']
        
        self.logger.info(f'Loaded robust model from {checkpoint_path}') 