import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import time
import psutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ModelComparator:
    """
    A class for comparing different models in terms of performance and resource usage.
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: torch.device,
        output_dir: Union[str, Path],
        batch_sizes: List[int] = [1, 8, 16, 32],
        num_runs: int = 10
    ):
        """
        Initialize the model comparison framework.
        
        Args:
            models: Dictionary of models to compare
            device: Device to run the models on
            output_dir: Directory to save comparison results
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs for each test
        """
        self.models = models
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_sizes = batch_sizes
        self.num_runs = num_runs
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / 'model_comparison.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # Initialize results storage
        self.results = {
            'performance': {},
            'resource_usage': {},
            'benchmarks': {}
        }
    
    def _measure_inference_time(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> float:
        """
        Measure inference time for a single forward pass.
        
        Args:
            model: Model to test
            inputs: Input tensor
            
        Returns:
            Average inference time in milliseconds
        """
        model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(self.num_runs):
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return np.mean(times)
    
    def _measure_memory_usage(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measure memory usage during inference.
        
        Args:
            model: Model to test
            inputs: Input tensor
            
        Returns:
            Dictionary of memory usage metrics
        """
        model.eval()
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        with torch.no_grad():
            _ = model(inputs)
        
        # Get peak memory usage
        peak_memory = process.memory_info().peak_rss / 1024 / 1024  # Convert to MB
        
        return {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - initial_memory
        }
    
    def _measure_accuracy(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> float:
        """
        Measure model accuracy on test data.
        
        Args:
            model: Model to test
            test_loader: DataLoader for test data
            
        Returns:
            Accuracy percentage
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total
    
    def _run_benchmark(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        batch_size: int
    ) -> Dict[str, float]:
        """
        Run a complete benchmark for a model.
        
        Args:
            model: Model to benchmark
            test_loader: DataLoader for test data
            batch_size: Batch size to use
            
        Returns:
            Dictionary of benchmark results
        """
        # Get a batch of data
        inputs, _ = next(iter(test_loader))
        inputs = inputs[:batch_size].to(self.device)
        
        # Measure performance
        inference_time = self._measure_inference_time(model, inputs)
        memory_usage = self._measure_memory_usage(model, inputs)
        accuracy = self._measure_accuracy(model, test_loader)
        
        return {
            'inference_time_ms': inference_time,
            'accuracy': accuracy,
            **memory_usage
        }
    
    def compare_models(
        self,
        test_loader: DataLoader
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare all models across different batch sizes.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of comparison results
        """
        for model_name, model in self.models.items():
            self.logger.info(f'Testing model: {model_name}')
            model.to(self.device)
            
            self.results['performance'][model_name] = {}
            self.results['resource_usage'][model_name] = {}
            self.results['benchmarks'][model_name] = {}
            
            for batch_size in self.batch_sizes:
                self.logger.info(f'Testing batch size: {batch_size}')
                
                # Run benchmark
                benchmark_results = self._run_benchmark(
                    model,
                    test_loader,
                    batch_size
                )
                
                # Store results
                self.results['benchmarks'][model_name][str(batch_size)] = benchmark_results
                
                # Log results
                self.logger.info(
                    f'Results for {model_name} (batch size {batch_size}):\n'
                    f'Inference Time: {benchmark_results["inference_time_ms"]:.2f} ms\n'
                    f'Accuracy: {benchmark_results["accuracy"]:.2f}%\n'
                    f'Memory Usage: {benchmark_results["peak_memory_mb"]:.2f} MB'
                )
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self) -> None:
        """Save comparison results to files."""
        # Save raw results
        with open(self.output_dir / 'comparison_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Generate and save plots
        self._generate_plots()
    
    def _generate_plots(self) -> None:
        """Generate comparison plots."""
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot inference time vs batch size
        plt.figure(figsize=(10, 6))
        for model_name in self.models.keys():
            batch_sizes = []
            inference_times = []
            for batch_size in self.batch_sizes:
                batch_sizes.append(batch_size)
                inference_times.append(
                    self.results['benchmarks'][model_name][str(batch_size)]['inference_time_ms']
                )
            plt.plot(batch_sizes, inference_times, label=model_name)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time vs Batch Size')
        plt.legend()
        plt.savefig(plots_dir / 'inference_time.png')
        plt.close()
        
        # Plot memory usage vs batch size
        plt.figure(figsize=(10, 6))
        for model_name in self.models.keys():
            batch_sizes = []
            memory_usage = []
            for batch_size in self.batch_sizes:
                batch_sizes.append(batch_size)
                memory_usage.append(
                    self.results['benchmarks'][model_name][str(batch_size)]['peak_memory_mb']
                )
            plt.plot(batch_sizes, memory_usage, label=model_name)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.title('Memory Usage vs Batch Size')
        plt.legend()
        plt.savefig(plots_dir / 'memory_usage.png')
        plt.close()
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        model_names = list(self.models.keys())
        accuracies = [
            self.results['benchmarks'][name][str(self.batch_sizes[0])]['accuracy']
            for name in model_names
        ]
        
        plt.bar(model_names, accuracies)
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy Comparison')
        plt.savefig(plots_dir / 'accuracy_comparison.png')
        plt.close()
    
    def get_best_model(
        self,
        metric: str = 'accuracy',
        batch_size: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Get the best model based on a specific metric.
        
        Args:
            metric: Metric to compare ('accuracy', 'inference_time_ms', 'peak_memory_mb')
            batch_size: Specific batch size to consider (if None, uses first batch size)
            
        Returns:
            Tuple of (best model name, metric value)
        """
        if batch_size is None:
            batch_size = self.batch_sizes[0]
        
        best_model = None
        best_value = float('-inf') if metric == 'accuracy' else float('inf')
        
        for model_name in self.models.keys():
            value = self.results['benchmarks'][model_name][str(batch_size)][metric]
            
            if (metric == 'accuracy' and value > best_value) or \
               (metric != 'accuracy' and value < best_value):
                best_value = value
                best_model = model_name
        
        return best_model, best_value 