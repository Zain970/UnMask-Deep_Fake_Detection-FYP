import torch
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
    auc,
    precision_recall_curve,
    average_precision_score
)
from pathlib import Path
import json
import logging
from datetime import datetime

class MetricsEvaluator:
    def __init__(
        self,
        num_classes: int = 2,
        average: str = 'binary',
        output_dir: str = 'evaluation_results'
    ):
        """
        Initialize metrics evaluator
        
        Args:
            num_classes: Number of classes
            average: Averaging method for multi-class metrics
            output_dir: Directory to save evaluation results
        """
        self.num_classes = num_classes
        self.average = average
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
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
    
    def reset(self):
        """Reset metrics storage"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def compute_accuracy_metrics(self) -> Dict[str, float]:
        """
        Compute accuracy-based metrics
        
        Returns:
            Dictionary of accuracy metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average=self.average),
            'recall': recall_score(targets, predictions, average=self.average),
            'f1': f1_score(targets, predictions, average=self.average)
        }
        
        return metrics
    
    def compute_confusion_matrix(
        self,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute confusion matrix
        
        Args:
            normalize: Whether to normalize the matrix
            
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(self.targets, self.predictions)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm
    
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
    
    def compute_roc_metrics(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute ROC curve metrics
        
        Returns:
            Tuple of (fpr, tpr, auc_score)
        """
        if not self.probabilities:
            raise ValueError('No probabilities available for ROC analysis')
        
        probabilities = np.array(self.probabilities)
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(self.targets, probabilities[:, 1])
            auc_score = auc(fpr, tpr)
        else:
            # One-vs-rest ROC curve for multi-class
            fpr = dict()
            tpr = dict()
            auc_score = dict()
            
            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(self.targets == i, probabilities[:, i])
                auc_score[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve
            fpr["micro"], tpr["micro"], _ = roc_curve(
                np.array(self.targets).ravel(),
                probabilities.ravel()
            )
            auc_score["micro"] = auc(fpr["micro"], tpr["micro"])
        
        return fpr, tpr, auc_score
    
    def plot_roc_curve(
        self,
        fpr: Union[np.ndarray, Dict[int, np.ndarray]],
        tpr: Union[np.ndarray, Dict[int, np.ndarray]],
        auc_score: Union[float, Dict[int, float]],
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
        
        if self.num_classes == 2:
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        else:
            for i in range(self.num_classes):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    label=f'Class {i} (AUC = {auc_score[i]:.3f})'
                )
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label=f'Micro-average (AUC = {auc_score["micro"]:.3f})',
                linestyle='--'
            )
        
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
    
    def compute_precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve
        
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        if not self.probabilities:
            raise ValueError('No probabilities available for precision-recall analysis')
        
        probabilities = np.array(self.probabilities)
        if self.num_classes == 2:
            precision, recall, thresholds = precision_recall_curve(
                self.targets,
                probabilities[:, 1]
            )
        else:
            # One-vs-rest precision-recall curve for multi-class
            precision = dict()
            recall = dict()
            thresholds = dict()
            
            for i in range(self.num_classes):
                precision[i], recall[i], thresholds[i] = precision_recall_curve(
                    self.targets == i,
                    probabilities[:, i]
                )
        
        return precision, recall, thresholds
    
    def plot_precision_recall_curve(
        self,
        precision: Union[np.ndarray, Dict[int, np.ndarray]],
        recall: Union[np.ndarray, Dict[int, np.ndarray]],
        title: str = 'Precision-Recall Curve'
    ):
        """
        Plot precision-recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        
        if self.num_classes == 2:
            plt.plot(recall, precision, label='Precision-Recall curve')
            ap_score = average_precision_score(
                self.targets,
                np.array(self.probabilities)[:, 1]
            )
            plt.title(f'{title} (AP = {ap_score:.3f})')
        else:
            for i in range(self.num_classes):
                ap_score = average_precision_score(
                    self.targets == i,
                    np.array(self.probabilities)[:, i]
                )
                plt.plot(
                    recall[i],
                    precision[i],
                    label=f'Class {i} (AP = {ap_score:.3f})'
                )
            plt.title(title)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall_curve.png')
        plt.close()
    
    def evaluate(
        self,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Run complete evaluation
        
        Args:
            class_names: List of class names
            
        Returns:
            Dictionary of metrics
        """
        # Compute accuracy metrics
        metrics = self.compute_accuracy_metrics()
        
        # Compute and plot confusion matrix
        cm = self.compute_confusion_matrix()
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        self.plot_confusion_matrix(cm, class_names)
        
        # Compute and plot ROC curve if probabilities are available
        if self.probabilities:
            fpr, tpr, auc_score = self.compute_roc_metrics()
            self.plot_roc_curve(fpr, tpr, auc_score)
            
            if self.num_classes == 2:
                metrics['auc'] = auc_score
            else:
                metrics['auc_micro'] = auc_score['micro']
                for i in range(self.num_classes):
                    metrics[f'auc_class_{i}'] = auc_score[i]
        
        # Compute and plot precision-recall curve
        if self.probabilities:
            precision, recall, _ = self.compute_precision_recall_curve()
            self.plot_precision_recall_curve(precision, recall)
            
            if self.num_classes == 2:
                metrics['ap'] = average_precision_score(
                    self.targets,
                    np.array(self.probabilities)[:, 1]
                )
            else:
                for i in range(self.num_classes):
                    metrics[f'ap_class_{i}'] = average_precision_score(
                        self.targets == i,
                        np.array(self.probabilities)[:, i]
                    )
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _save_metrics(
        self,
        metrics: Dict[str, float]
    ):
        """
        Save evaluation metrics
        
        Args:
            metrics: Dictionary of metrics
        """
        # Save metrics
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log metrics
        self.logger.info('Evaluation Results:')
        for metric, value in metrics.items():
            self.logger.info(f'{metric}: {value:.4f}')
    
    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and targets
        
        Returns:
            Tuple of (predictions, targets)
        """
        return np.array(self.predictions), np.array(self.targets)
    
    def get_probabilities(self) -> Optional[np.ndarray]:
        """
        Get class probabilities
        
        Returns:
            Class probabilities if available
        """
        return np.array(self.probabilities) if self.probabilities else None 