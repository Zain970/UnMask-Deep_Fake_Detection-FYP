import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
from datetime import datetime

class ValidationMetrics:
    def __init__(
        self,
        num_classes: int = 2,
        average: str = 'binary'
    ):
        """
        Initialize validation metrics
        
        Args:
            num_classes: Number of classes
            average: Averaging method for multi-class metrics
        """
        self.num_classes = num_classes
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with batch results
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            probabilities: Class probabilities
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute metrics
        
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities) if self.probabilities else None
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average=self.average),
            'recall': recall_score(targets, predictions, average=self.average),
            'f1': f1_score(targets, predictions, average=self.average)
        }
        
        if probabilities is not None:
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            metrics['auc'] = auc(fpr, tpr)
        
        return metrics
    
    def get_confusion_matrix(
        self,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get confusion matrix
        
        Args:
            normalize: Whether to normalize the matrix
            
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(self.targets, self.predictions)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm

class ValidationVisualizer:
    def __init__(
        self,
        output_dir: str = 'validation_results'
    ):
        """
        Initialize validation visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = 'Confusion Matrix'
    ):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        title: str = 'ROC Curve'
    ):
        """
        Plot ROC curve
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()
    
    def plot_metrics(
        self,
        metrics: Dict[str, float],
        title: str = 'Validation Metrics'
    ):
        """
        Plot metrics bar chart
        
        Args:
            metrics: Dictionary of metrics
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics.png')
        plt.close()

class ModelValidator:
    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'validation_results'
    ):
        """
        Initialize model validator
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            device: Device to validate on
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        
        # Initialize metrics
        self.metrics = ValidationMetrics()
        
        # Initialize visualizer
        self.visualizer = ValidationVisualizer(output_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            class_names: List of class names
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Update metrics
                self.metrics.update(predictions, targets, probabilities)
        
        # Compute metrics
        metrics = self.metrics.compute()
        
        # Get confusion matrix
        cm = self.metrics.get_confusion_matrix()
        
        # Plot results
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.metrics.num_classes)]
        
        self.visualizer.plot_confusion_matrix(cm, class_names)
        
        if 'auc' in metrics:
            fpr, tpr, _ = roc_curve(
                self.metrics.targets,
                np.array(self.metrics.probabilities)[:, 1]
            )
            self.visualizer.plot_roc_curve(fpr, tpr, metrics['auc'])
        
        self.visualizer.plot_metrics(metrics)
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _save_results(
        self,
        metrics: Dict[str, float]
    ):
        """
        Save validation results
        
        Args:
            metrics: Dictionary of metrics
        """
        # Save metrics
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(self.visualizer.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log metrics
        self.logger.info('Validation Results:')
        for metric, value in metrics.items():
            self.logger.info(f'{metric}: {value:.4f}')
    
    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and targets
        
        Returns:
            Tuple of (predictions, targets)
        """
        return np.array(self.metrics.predictions), np.array(self.metrics.targets)
    
    def get_probabilities(self) -> Optional[np.ndarray]:
        """
        Get class probabilities
        
        Returns:
            Class probabilities if available
        """
        return np.array(self.metrics.probabilities) if self.metrics.probabilities else None 