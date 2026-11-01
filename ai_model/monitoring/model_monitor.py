import torch
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import logging
from scipy import stats
import psutil
import gc

class ModelMonitor:
    def __init__(
        self,
        model: torch.nn.Module,
        output_dir: Union[str, Path],
        performance_threshold: float = 0.95,
        drift_threshold: float = 0.1,
        health_check_interval: int = 3600,  # 1 hour
        max_history_size: int = 1000
    ):
        """
        Initialize the model monitor.
        
        Args:
            model: The model to monitor
            output_dir: Directory to save monitoring data
            performance_threshold: Minimum acceptable performance
            drift_threshold: Maximum allowed data drift
            health_check_interval: Time between health checks (seconds)
            max_history_size: Maximum number of historical records to keep
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        self.health_check_interval = health_check_interval
        self.max_history_size = max_history_size
        
        # Initialize monitoring data
        self.performance_history: List[Dict] = []
        self.drift_history: List[Dict] = []
        self.health_history: List[Dict] = []
        
        # Set up logging
        self.logger = logging.getLogger('ModelMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_dir / 'monitoring.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Initialize last health check time
        self.last_health_check = time.time()
    
    def track_performance(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> None:
        """
        Track model performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            timestamp: Optional timestamp for the metrics
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add metrics to history
        record = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        self.performance_history.append(record)
        
        # Trim history if needed
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
        
        # Check for performance degradation
        if 'accuracy' in metrics and metrics['accuracy'] < self.performance_threshold:
            self.logger.warning(
                f"Performance degradation detected: accuracy {metrics['accuracy']} "
                f"below threshold {self.performance_threshold}"
            )
        
        # Save performance history
        self._save_history('performance_history.json', self.performance_history)
    
    def detect_drift(
        self,
        current_data: torch.Tensor,
        reference_data: torch.Tensor,
        timestamp: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Detect data drift between current and reference data.
        
        Args:
            current_data: Current batch of data
            reference_data: Reference batch of data
            timestamp: Optional timestamp for the drift detection
        
        Returns:
            Dictionary of drift metrics
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Convert to numpy for statistical analysis
        current_np = current_data.cpu().numpy()
        reference_np = reference_data.cpu().numpy()
        
        # Calculate drift metrics
        drift_metrics = {
            'ks_statistic': self._calculate_ks_statistic(current_np, reference_np),
            'wasserstein_distance': self._calculate_wasserstein_distance(current_np, reference_np),
            'mean_difference': np.mean(current_np) - np.mean(reference_np),
            'std_difference': np.std(current_np) - np.std(reference_np)
        }
        
        # Add to drift history
        record = {
            'timestamp': timestamp,
            'metrics': drift_metrics
        }
        self.drift_history.append(record)
        
        # Trim history if needed
        if len(self.drift_history) > self.max_history_size:
            self.drift_history = self.drift_history[-self.max_history_size:]
        
        # Check for significant drift
        if drift_metrics['ks_statistic'] > self.drift_threshold:
            self.logger.warning(
                f"Data drift detected: KS statistic {drift_metrics['ks_statistic']} "
                f"above threshold {self.drift_threshold}"
            )
        
        # Save drift history
        self._save_history('drift_history.json', self.drift_history)
        
        return drift_metrics
    
    def check_health(self) -> Dict[str, Union[bool, float]]:
        """
        Perform comprehensive model health check.
        
        Returns:
            Dictionary of health check results
        """
        current_time = time.time()
        
        # Check if enough time has passed since last health check
        if current_time - self.last_health_check < self.health_check_interval:
            return self.health_history[-1]['results'] if self.health_history else {}
        
        # Perform health checks
        health_results = {
            'memory_usage': self._check_memory_usage(),
            'gpu_available': torch.cuda.is_available(),
            'model_weights_valid': self._check_model_weights(),
            'input_output_consistency': self._check_input_output_consistency(),
            'performance_stability': self._check_performance_stability(),
            'data_drift': self._check_data_drift()
        }
        
        # Add to health history
        record = {
            'timestamp': current_time,
            'results': health_results
        }
        self.health_history.append(record)
        
        # Trim history if needed
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
        
        # Save health history
        self._save_history('health_history.json', self.health_history)
        
        # Update last health check time
        self.last_health_check = current_time
        
        return health_results
    
    def _calculate_ks_statistic(
        self,
        current: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Calculate Kolmogorov-Smirnov statistic between two distributions."""
        return stats.ks_2samp(current.flatten(), reference.flatten()).statistic
    
    def _calculate_wasserstein_distance(
        self,
        current: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Calculate Wasserstein distance between two distributions."""
        return stats.wasserstein_distance(current.flatten(), reference.flatten())
    
    def _check_memory_usage(self) -> float:
        """Check current memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        else:
            process = psutil.Process()
            return process.memory_percent() / 100
    
    def _check_model_weights(self) -> bool:
        """Check if model weights are valid."""
        try:
            for param in self.model.parameters():
                if not torch.isfinite(param).all():
                    return False
            return True
        except Exception:
            return False
    
    def _check_input_output_consistency(self) -> bool:
        """Check if model produces consistent outputs for same input."""
        try:
            test_input = torch.randn(1, 3, 224, 224)
            output1 = self.model(test_input)
            output2 = self.model(test_input)
            return torch.allclose(output1, output2)
        except Exception:
            return False
    
    def _check_performance_stability(self) -> bool:
        """Check if model performance is stable."""
        if len(self.performance_history) < 2:
            return True
        
        recent_metrics = self.performance_history[-10:]
        if not recent_metrics:
            return True
        
        accuracies = [record['metrics'].get('accuracy', 0) for record in recent_metrics]
        return np.std(accuracies) < 0.1
    
    def _check_data_drift(self) -> bool:
        """Check if data drift is within acceptable limits."""
        if len(self.drift_history) < 2:
            return True
        
        recent_drift = self.drift_history[-1]['metrics']
        return recent_drift['ks_statistic'] < self.drift_threshold
    
    def _save_history(self, filename: str, history: List[Dict]) -> None:
        """Save monitoring history to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_monitoring_summary(self) -> Dict:
        """Get a summary of all monitoring data."""
        return {
            'performance': {
                'current': self.performance_history[-1] if self.performance_history else None,
                'trend': self._calculate_trend(self.performance_history)
            },
            'drift': {
                'current': self.drift_history[-1] if self.drift_history else None,
                'trend': self._calculate_trend(self.drift_history)
            },
            'health': {
                'current': self.health_history[-1] if self.health_history else None,
                'status': self._get_health_status()
            }
        }
    
    def _calculate_trend(self, history: List[Dict]) -> str:
        """Calculate trend from history data."""
        if len(history) < 2:
            return 'stable'
        
        recent = history[-5:]
        if not recent:
            return 'stable'
        
        # Calculate trend based on the metric values
        values = [record['metrics'].get('accuracy', 0) for record in recent]
        if len(values) < 2:
            return 'stable'
        
        slope = np.polyfit(range(len(values)), values, 1)[0]
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'degrading'
        else:
            return 'stable'
    
    def _get_health_status(self) -> str:
        """Get overall health status."""
        if not self.health_history:
            return 'unknown'
        
        current_health = self.health_history[-1]['results']
        
        # Check critical health indicators
        if not current_health['model_weights_valid']:
            return 'critical'
        if not current_health['input_output_consistency']:
            return 'warning'
        if current_health['memory_usage'] > 0.9:
            return 'warning'
        if not current_health['performance_stability']:
            return 'warning'
        if not current_health['data_drift']:
            return 'warning'
        
        return 'healthy' 