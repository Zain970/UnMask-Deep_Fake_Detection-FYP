import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import time
import logging
from torch.utils.data import DataLoader
from ..data.data_loader import DataLoader as UnMaskDataLoader
from ..model.base_model import BaseModel
from ..debugging.model_debugger import ModelDebugger

class FinalOptimizer:
    def __init__(
        self,
        model: BaseModel,
        data_loader: UnMaskDataLoader,
        output_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.device = device
        self.debugger = ModelDebugger(model, output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.output_dir / "optimization.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def fine_tune_hyperparameters(
        self,
        param_grid: Dict[str, List[float]],
        num_epochs: int = 10,
        num_folds: int = 5
    ) -> Dict[str, float]:
        """Fine-tune hyperparameters using grid search with cross-validation."""
        self.logger.info("Starting hyperparameter fine-tuning")
        
        best_params = {}
        best_score = float('-inf')
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            self.logger.info(f"Testing parameters: {params}")
            
            # Cross-validation
            fold_scores = []
            for fold in range(num_folds):
                # Split data for this fold
                train_loader, val_loader = self._get_fold_data(fold, num_folds)
                
                # Train with current parameters
                score = self._train_with_params(params, train_loader, val_loader, num_epochs)
                fold_scores.append(score)
            
            # Calculate average score
            avg_score = np.mean(fold_scores)
            self.logger.info(f"Average score: {avg_score}")
            
            # Update best parameters if needed
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best score: {best_score}")
        
        # Save best parameters
        with open(self.output_dir / "best_params.json", "w") as f:
            json.dump(best_params, f)
        
        return best_params
    
    def optimize_inference_speed(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Optimize inference speed by testing different batch sizes and configurations."""
        self.logger.info("Starting inference speed optimization")
        
        results = {}
        
        for batch_size in batch_sizes:
            self.logger.info(f"Testing batch size: {batch_size}")
            
            # Create data loader with current batch size
            test_loader = DataLoader(
                self.data_loader.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Warmup
            self._run_inference_batch(test_loader, num_warmup)
            
            # Measure performance
            times = []
            for _ in range(num_iterations):
                start_time = time.time()
                self._run_inference_batch(test_loader, 1)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate metrics
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            results[f"batch_{batch_size}"] = {
                "avg_time": avg_time,
                "throughput": throughput,
                "p95_time": np.percentile(times, 95),
                "p99_time": np.percentile(times, 99)
            }
            
            self.logger.info(f"Batch size {batch_size}: {results[f'batch_{batch_size}']}")
        
        # Save results
        with open(self.output_dir / "inference_speed.json", "w") as f:
            json.dump(results, f)
        
        return results
    
    def run_final_performance_tests(
        self,
        test_data: str,
        batch_size: int = 32,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Run comprehensive performance tests on the optimized model."""
        self.logger.info("Starting final performance tests")
        
        # Initialize test data loader
        test_loader = DataLoader(
            self.data_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Run tests
        results = {
            "accuracy": self._test_accuracy(test_loader),
            "inference_speed": self._test_inference_speed(test_loader, num_iterations),
            "memory_usage": self._test_memory_usage(test_loader),
            "robustness": self._test_robustness(test_loader)
        }
        
        # Save results
        with open(self.output_dir / "final_performance.json", "w") as f:
            json.dump(results, f)
        
        return results
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[float]]) -> List[Dict[str, float]]:
        """Generate all combinations of parameters from the grid."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _get_fold_data(self, fold: int, num_folds: int) -> Tuple[DataLoader, DataLoader]:
        """Split data into training and validation sets for cross-validation."""
        dataset = self.data_loader.dataset
        fold_size = len(dataset) // num_folds
        
        # Calculate indices for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        
        # Create indices for train and val sets
        indices = list(range(len(dataset)))
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]
        
        # Create data loaders
        train_loader = DataLoader(
            dataset,
            batch_size=self.data_loader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_indices),
            num_workers=4
        )
        val_loader = DataLoader(
            dataset,
            batch_size=self.data_loader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def _train_with_params(
        self,
        params: Dict[str, float],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> float:
        """Train model with given parameters and return validation score."""
        # Create optimizer with current parameters
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=params.get("learning_rate", 0.001),
            weight_decay=params.get("weight_decay", 0.0001)
        )
        
        # Train for specified epochs
        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch["images"].to(self.device))
                loss = nn.CrossEntropyLoss()(outputs, batch["labels"].to(self.device))
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation set
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(batch["images"].to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total += batch["labels"].size(0)
                correct += (predicted == batch["labels"].to(self.device)).sum().item()
        
        return correct / total
    
    def _run_inference_batch(self, data_loader: DataLoader, num_batches: int) -> None:
        """Run inference on specified number of batches."""
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                self.model(batch["images"].to(self.device))
    
    def _test_accuracy(self, test_loader: DataLoader) -> float:
        """Test model accuracy on test set."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model(batch["images"].to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total += batch["labels"].size(0)
                correct += (predicted == batch["labels"].to(self.device)).sum().item()
        return correct / total
    
    def _test_inference_speed(self, test_loader: DataLoader, num_iterations: int) -> Dict[str, float]:
        """Test inference speed with different batch sizes."""
        times = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                for batch in test_loader:
                    self.model(batch["images"].to(self.device))
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            "avg_time": np.mean(times),
            "p95_time": np.percentile(times, 95),
            "p99_time": np.percentile(times, 99)
        }
    
    def _test_memory_usage(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test memory usage during inference."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                self.model(batch["images"].to(self.device))
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "initial_memory": initial_memory,
            "peak_memory": peak_memory,
            "memory_increase": peak_memory - initial_memory
        }
    
    def _test_robustness(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test model robustness against various perturbations."""
        from ..robustness.adversarial_robustness import AdversarialRobustness
        
        robustness = AdversarialRobustness(self.model, str(self.output_dir))
        results = robustness.evaluate_defenses(test_loader)
        
        return {
            "clean_accuracy": results["clean_accuracy"],
            "adversarial_accuracy": results["adversarial_accuracy"],
            "robustness_score": results["robustness_score"]
        } 