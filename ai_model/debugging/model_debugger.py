import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.autograd.profiler as profiler
import torch.nn.utils.prune as prune
from collections import defaultdict

class ModelDebugger:
    """
    A class for debugging deep learning models, including error analysis,
    performance profiling, and debugging utilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Union[str, Path]
    ):
        """
        Initialize the model debugger.
        
        Args:
            model: The model to debug
            device: Device to run the model on
            output_dir: Directory to save debugging results
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'model_debugging.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # Initialize metrics storage
        self.metrics = {
            'error_analysis': defaultdict(list),
            'performance_metrics': defaultdict(list),
            'layer_metrics': defaultdict(list)
        }
    
    def analyze_errors(
        self,
        test_loader: DataLoader,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze model errors on test data.
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary of error analysis results
        """
        self.model.eval()
        error_analysis = {
            'misclassified_samples': [],
            'confidence_distribution': [],
            'class_errors': defaultdict(int),
            'error_patterns': defaultdict(list)
        }
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if i >= num_samples:
                    break
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                
                # Track misclassified samples
                for j in range(len(labels)):
                    if predictions[j] != labels[j]:
                        error_analysis['misclassified_samples'].append({
                            'sample_idx': i * len(labels) + j,
                            'true_label': labels[j].item(),
                            'predicted_label': predictions[j].item(),
                            'confidence': probabilities[j][predictions[j]].item()
                        })
                        
                        # Track class-specific errors
                        error_analysis['class_errors'][labels[j].item()] += 1
                        
                        # Track error patterns
                        error_analysis['error_patterns'][(labels[j].item(), predictions[j].item())].append(
                            i * len(labels) + j
                        )
                
                # Track confidence distribution
                error_analysis['confidence_distribution'].extend(
                    probabilities.max(dim=1)[0].cpu().numpy().tolist()
                )
        
        # Save error analysis results
        self._save_error_analysis(error_analysis)
        
        return error_analysis
    
    def profile_performance(
        self,
        test_loader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, Any]:
        """
        Profile model performance.
        
        Args:
            test_loader: DataLoader for test data
            num_batches: Number of batches to profile
            
        Returns:
            Dictionary of performance metrics
        """
        self.model.eval()
        performance_metrics = {
            'inference_times': [],
            'memory_usage': [],
            'layer_times': defaultdict(list),
            'layer_memory': defaultdict(list)
        }
        
        process = psutil.Process()
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= num_batches:
                    break
                
                inputs = inputs.to(self.device)
                
                # Profile inference time
                start_time = time.time()
                outputs = self.model(inputs)
                end_time = time.time()
                
                # Track inference time
                performance_metrics['inference_times'].append(end_time - start_time)
                
                # Track memory usage
                performance_metrics['memory_usage'].append(
                    process.memory_info().rss / 1024 / 1024  # Convert to MB
                )
                
                # Profile layer-wise performance
                with profiler.profile(record_shapes=True) as prof:
                    with profiler.record_function("model_inference"):
                        _ = self.model(inputs)
                
                # Process profiling results
                for evt in prof.key_averages():
                    if evt.key.startswith("model_inference"):
                        continue
                    performance_metrics['layer_times'][evt.key].append(evt.cpu_time_total)
                    performance_metrics['layer_memory'][evt.key].append(evt.cpu_memory_usage)
        
        # Save performance metrics
        self._save_performance_metrics(performance_metrics)
        
        return performance_metrics
    
    def analyze_layer_behavior(
        self,
        test_loader: DataLoader,
        num_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze behavior of individual layers.
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary of layer analysis results
        """
        self.model.eval()
        layer_analysis = {
            'activations': defaultdict(list),
            'gradients': defaultdict(list),
            'weights': defaultdict(list)
        }
        
        # Register hooks for layer analysis
        hooks = []
        
        def forward_hook(module, input, output):
            layer_analysis['activations'][module.__class__.__name__].append(
                output.detach().cpu().numpy()
            )
        
        def backward_hook(module, grad_input, grad_output):
            layer_analysis['gradients'][module.__class__.__name__].append(
                grad_output[0].detach().cpu().numpy()
            )
        
        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(forward_hook))
                hooks.append(module.register_backward_hook(backward_hook))
                
                # Store weights
                layer_analysis['weights'][module.__class__.__name__].append(
                    module.weight.detach().cpu().numpy()
                )
        
        # Run analysis
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= num_samples:
                    break
                
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Save layer analysis results
        self._save_layer_analysis(layer_analysis)
        
        return layer_analysis
    
    def _save_error_analysis(
        self,
        error_analysis: Dict[str, Any]
    ) -> None:
        """
        Save error analysis results.
        
        Args:
            error_analysis: Dictionary of error analysis results
        """
        # Save raw results
        with open(self.output_dir / 'error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=4)
        
        # Generate and save plots
        self._plot_error_analysis(error_analysis)
    
    def _save_performance_metrics(
        self,
        performance_metrics: Dict[str, Any]
    ) -> None:
        """
        Save performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # Save raw results
        with open(self.output_dir / 'performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=4)
        
        # Generate and save plots
        self._plot_performance_metrics(performance_metrics)
    
    def _save_layer_analysis(
        self,
        layer_analysis: Dict[str, Any]
    ) -> None:
        """
        Save layer analysis results.
        
        Args:
            layer_analysis: Dictionary of layer analysis results
        """
        # Save raw results
        with open(self.output_dir / 'layer_analysis.json', 'w') as f:
            json.dump(layer_analysis, f, indent=4)
        
        # Generate and save plots
        self._plot_layer_analysis(layer_analysis)
    
    def _plot_error_analysis(
        self,
        error_analysis: Dict[str, Any]
    ) -> None:
        """
        Generate plots for error analysis.
        
        Args:
            error_analysis: Dictionary of error analysis results
        """
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(error_analysis['confidence_distribution'], bins=20)
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.savefig(plots_dir / 'confidence_distribution.png')
        plt.close()
        
        # Plot class error distribution
        plt.figure(figsize=(10, 6))
        plt.bar(
            list(error_analysis['class_errors'].keys()),
            list(error_analysis['class_errors'].values())
        )
        plt.xlabel('Class')
        plt.ylabel('Error Count')
        plt.title('Class Error Distribution')
        plt.savefig(plots_dir / 'class_errors.png')
        plt.close()
    
    def _plot_performance_metrics(
        self,
        performance_metrics: Dict[str, Any]
    ) -> None:
        """
        Generate plots for performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot inference times
        plt.figure(figsize=(10, 6))
        plt.plot(performance_metrics['inference_times'])
        plt.xlabel('Batch')
        plt.ylabel('Inference Time (s)')
        plt.title('Inference Time per Batch')
        plt.savefig(plots_dir / 'inference_times.png')
        plt.close()
        
        # Plot memory usage
        plt.figure(figsize=(10, 6))
        plt.plot(performance_metrics['memory_usage'])
        plt.xlabel('Batch')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage per Batch')
        plt.savefig(plots_dir / 'memory_usage.png')
        plt.close()
        
        # Plot layer-wise performance
        plt.figure(figsize=(15, 8))
        for layer, times in performance_metrics['layer_times'].items():
            plt.plot(times, label=layer)
        plt.xlabel('Batch')
        plt.ylabel('Time (ms)')
        plt.title('Layer-wise Performance')
        plt.legend()
        plt.savefig(plots_dir / 'layer_performance.png')
        plt.close()
    
    def _plot_layer_analysis(
        self,
        layer_analysis: Dict[str, Any]
    ) -> None:
        """
        Generate plots for layer analysis.
        
        Args:
            layer_analysis: Dictionary of layer analysis results
        """
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot activation distributions
        plt.figure(figsize=(15, 8))
        for layer, activations in layer_analysis['activations'].items():
            plt.hist(
                np.concatenate([a.flatten() for a in activations]),
                bins=50,
                alpha=0.5,
                label=layer
            )
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        plt.title('Layer Activation Distributions')
        plt.legend()
        plt.savefig(plots_dir / 'activation_distributions.png')
        plt.close()
        
        # Plot weight distributions
        plt.figure(figsize=(15, 8))
        for layer, weights in layer_analysis['weights'].items():
            plt.hist(
                np.concatenate([w.flatten() for w in weights]),
                bins=50,
                alpha=0.5,
                label=layer
            )
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        plt.title('Layer Weight Distributions')
        plt.legend()
        plt.savefig(plots_dir / 'weight_distributions.png')
        plt.close() 