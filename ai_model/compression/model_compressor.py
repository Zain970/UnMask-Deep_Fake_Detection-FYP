import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import torch.nn.utils.prune as prune
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub

class ModelCompressor:
    """
    A class for implementing model compression techniques including
    knowledge distillation, model pruning, and quantization.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: torch.device,
        output_dir: Union[str, Path],
        temperature: float = 2.0,
        alpha: float = 0.5,
        pruning_amount: float = 0.3,
        quantization_bits: int = 8
    ):
        """
        Initialize the model compression pipeline.
        
        Args:
            teacher_model: The teacher model for knowledge distillation
            student_model: The student model to compress
            device: Device to run the models on
            output_dir: Directory to save compressed models and logs
            temperature: Temperature for knowledge distillation
            alpha: Weight for distillation loss
            pruning_amount: Amount of weights to prune
            quantization_bits: Number of bits for quantization
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temperature = temperature
        self.alpha = alpha
        self.pruning_amount = pruning_amount
        self.quantization_bits = quantization_bits
        
        # Move models to device
        self.teacher_model.to(device)
        self.student_model.to(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'model_compression.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
    
    def _knowledge_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_outputs: Outputs from student model
            teacher_outputs: Outputs from teacher model
            labels: True labels
            
        Returns:
            Combined loss
        """
        # Hard loss (cross entropy with true labels)
        hard_loss = nn.CrossEntropyLoss()(student_outputs, labels)
        
        # Soft loss (KL divergence with teacher outputs)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            nn.LogSoftmax(dim=1)(student_outputs / self.temperature),
            nn.Softmax(dim=1)(teacher_outputs / self.temperature)
        ) * (self.temperature ** 2)
        
        # Combine losses
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
    
    def distill(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.001
    ) -> Dict[str, List[float]]:
        """
        Perform knowledge distillation.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            learning_rate: Learning rate for training
            
        Returns:
            Dictionary of training history
        """
        optimizer = optim.Adam(self.student_model.parameters(), lr=learning_rate)
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Train
            self.student_model.train()
            self.teacher_model.eval()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs)
                
                student_outputs = self.student_model(inputs)
                
                # Compute loss
                loss = self._knowledge_distillation_loss(
                    student_outputs,
                    teacher_outputs,
                    labels
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = student_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Validate
            val_loss, val_accuracy = self._validate(val_loader)
            
            # Log metrics
            train_accuracy = 100. * correct / total
            self.logger.info(
                f'Epoch {epoch + 1}/{epochs} - '
                f'Train Loss: {total_loss / len(train_loader):.4f}, '
                f'Train Acc: {train_accuracy:.2f}%, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_accuracy:.2f}%'
            )
            
            # Update history
            history['train_loss'].append(total_loss / len(train_loader))
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_loss, val_accuracy)
        
        return history
    
    def _validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Validate the student model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        self.student_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.student_model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def prune_model(self) -> None:
        """
        Prune the student model.
        """
        # Get parameters to prune
        parameters_to_prune = []
        for name, module in self.student_model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.pruning_amount
        )
        
        # Make pruning permanent
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        self.logger.info(f'Pruned {self.pruning_amount * 100:.1f}% of weights')
    
    def quantize_model(self) -> None:
        """
        Quantize the student model.
        """
        # Add quantization stubs
        self.student_model.quant = QuantStub()
        self.student_model.dequant = DeQuantStub()
        
        # Prepare model for quantization
        self.student_model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(self.student_model, inplace=True)
        
        # Calibrate model
        self.student_model.eval()
        with torch.no_grad():
            for _ in range(100):  # Use a subset of data for calibration
                inputs = torch.randn(1, 3, 224, 224).to(self.device)
                _ = self.student_model(inputs)
        
        # Convert to quantized model
        quantization.convert(self.student_model, inplace=True)
        
        self.logger.info(f'Quantized model to {self.quantization_bits} bits')
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_accuracy: float
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            val_accuracy: Validation accuracy
        """
        checkpoint = {
            'epoch': epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'pruning_amount': self.pruning_amount,
            'quantization_bits': self.quantization_bits
        }
        
        torch.save(
            checkpoint,
            self.output_dir / f'compressed_model_epoch_{epoch}.pt'
        )
        
        # Save compression info
        info = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'pruning_amount': self.pruning_amount,
            'quantization_bits': self.quantization_bits
        }
        
        with open(self.output_dir / 'compression_info.json', 'w') as f:
            json.dump(info, f, indent=4)
    
    def load_compressed_model(
        self,
        checkpoint_path: Union[str, Path]
    ) -> None:
        """
        Load compressed model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.temperature = checkpoint['temperature']
        self.alpha = checkpoint['alpha']
        self.pruning_amount = checkpoint['pruning_amount']
        self.quantization_bits = checkpoint['quantization_bits']
        
        self.logger.info(f'Loaded compressed model from {checkpoint_path}') 