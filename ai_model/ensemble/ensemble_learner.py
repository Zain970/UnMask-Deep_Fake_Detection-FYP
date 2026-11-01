import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from collections import defaultdict
import torch.nn.functional as F
from scipy.stats import mode

class EnsembleLearner:
    def __init__(
        self,
        models: List[nn.Module],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'ensemble_results'
    ):
        """
        Initialize ensemble learner
        
        Args:
            models: List of models to ensemble
            device: Device to run inference on
            output_dir: Directory to save results
        """
        self.models = [model.to(device) for model in models]
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set models to eval mode
        for model in self.models:
            model.eval()
    
    def predict(
        self,
        x: torch.Tensor,
        voting_method: str = 'soft',
        confidence_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make ensemble prediction
        
        Args:
            x: Input tensor
            voting_method: Voting method ('soft', 'hard', 'weighted')
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (predictions, confidences)
        """
        # Get predictions from all models
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(x)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                predictions.append(preds)
                confidences.append(probs)
        
        # Stack predictions and confidences
        predictions = torch.stack(predictions)
        confidences = torch.stack(confidences)
        
        # Apply voting mechanism
        if voting_method == 'soft':
            final_preds, final_confs = self._soft_voting(predictions, confidences)
        elif voting_method == 'hard':
            final_preds, final_confs = self._hard_voting(predictions, confidences)
        elif voting_method == 'weighted':
            final_preds, final_confs = self._weighted_voting(predictions, confidences)
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")
        
        # Apply confidence threshold
        mask = final_confs >= confidence_threshold
        final_preds = final_preds * mask + (-1) * (~mask)  # -1 for low confidence
        
        return final_preds, final_confs
    
    def _soft_voting(
        self,
        predictions: torch.Tensor,
        confidences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply soft voting
        
        Args:
            predictions: Stacked predictions
            confidences: Stacked confidences
            
        Returns:
            Tuple of (final predictions, final confidences)
        """
        # Average confidences
        avg_confs = torch.mean(confidences, dim=0)
        final_preds = torch.argmax(avg_confs, dim=1)
        final_confs = torch.max(avg_confs, dim=1)[0]
        
        return final_preds, final_confs
    
    def _hard_voting(
        self,
        predictions: torch.Tensor,
        confidences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hard voting
        
        Args:
            predictions: Stacked predictions
            confidences: Stacked confidences
            
        Returns:
            Tuple of (final predictions, final confidences)
        """
        # Convert to numpy for mode calculation
        preds_np = predictions.cpu().numpy()
        
        # Calculate mode for each sample
        final_preds = []
        final_confs = []
        
        for i in range(preds_np.shape[1]):
            pred, count = mode(preds_np[:, i])
            final_preds.append(pred[0])
            
            # Calculate confidence as ratio of models agreeing
            confidence = count[0] / preds_np.shape[0]
            final_confs.append(confidence)
        
        # Convert back to tensors
        final_preds = torch.tensor(final_preds, device=predictions.device)
        final_confs = torch.tensor(final_confs, device=confidences.device)
        
        return final_preds, final_confs
    
    def _weighted_voting(
        self,
        predictions: torch.Tensor,
        confidences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply weighted voting
        
        Args:
            predictions: Stacked predictions
            confidences: Stacked confidences
            
        Returns:
            Tuple of (final predictions, final confidences)
        """
        # Calculate model weights based on confidence
        weights = torch.mean(confidences, dim=1)  # Average confidence per model
        
        # Normalize weights
        weights = weights / torch.sum(weights)
        
        # Weighted average of confidences
        weighted_confs = torch.sum(confidences * weights.view(-1, 1, 1), dim=0)
        final_preds = torch.argmax(weighted_confs, dim=1)
        final_confs = torch.max(weighted_confs, dim=1)[0]
        
        return final_preds, final_confs
    
    def evaluate_ensemble(
        self,
        test_loader: torch.utils.data.DataLoader,
        voting_method: str = 'soft',
        confidence_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance
        
        Args:
            test_loader: Test data loader
            voting_method: Voting method
            confidence_threshold: Confidence threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_preds = []
        all_confs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                preds, confs = self.predict(
                    x,
                    voting_method=voting_method,
                    confidence_threshold=confidence_threshold
                )
                
                all_preds.append(preds)
                all_confs.append(confs)
                all_labels.append(y)
        
        # Concatenate results
        all_preds = torch.cat(all_preds)
        all_confs = torch.cat(all_confs)
        all_labels = torch.cat(all_labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': self._calculate_accuracy(all_preds, all_labels),
            'confidence': torch.mean(all_confs).item(),
            'confidence_std': torch.std(all_confs).item(),
            'low_confidence_ratio': torch.mean((all_confs < confidence_threshold).float()).item()
        }
        
        # Log results
        self._log_evaluation_results(metrics, voting_method, confidence_threshold)
        
        return metrics
    
    def _calculate_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Calculate accuracy
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            
        Returns:
            Accuracy score
        """
        # Only consider high confidence predictions
        mask = predictions != -1
        if torch.sum(mask) == 0:
            return 0.0
        
        correct = (predictions[mask] == labels[mask]).float()
        return torch.mean(correct).item()
    
    def _log_evaluation_results(
        self,
        metrics: Dict[str, float],
        voting_method: str,
        confidence_threshold: float
    ):
        """
        Log evaluation results
        
        Args:
            metrics: Evaluation metrics
            voting_method: Voting method used
            confidence_threshold: Confidence threshold used
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'voting_method': voting_method,
            'confidence_threshold': confidence_threshold,
            'metrics': metrics
        }
        
        # Save results
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log results
        self.logger.info('Ensemble Evaluation Results:')
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
    
    def save_ensemble(
        self,
        filename: str = 'ensemble.pt'
    ):
        """
        Save ensemble models
        
        Args:
            filename: Output filename
        """
        # Save model states
        states = [model.state_dict() for model in self.models]
        torch.save(states, self.output_dir / filename)
        
        # Save ensemble info
        info = {
            'timestamp': datetime.now().isoformat(),
            'num_models': len(self.models),
            'model_types': [type(model).__name__ for model in self.models]
        }
        
        with open(self.output_dir / 'ensemble_info.json', 'w') as f:
            json.dump(info, f, indent=4)
        
        self.logger.info(f'Ensemble saved to {self.output_dir / filename}')
    
    def load_ensemble(
        self,
        filename: str = 'ensemble.pt'
    ):
        """
        Load ensemble models
        
        Args:
            filename: Input filename
        """
        # Load model states
        states = torch.load(self.output_dir / filename)
        
        # Load states into models
        for model, state in zip(self.models, states):
            model.load_state_dict(state)
        
        self.logger.info(f'Ensemble loaded from {self.output_dir / filename}') 