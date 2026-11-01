import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import shutil
import tempfile
from ai_model.optimization.final_optimizer import FinalOptimizer
from ai_model.model.base_model import BaseModel
from ai_model.data.data_loader import DataLoader

class TestFinalOptimizer(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create dummy model
        self.model = BaseModel()
        
        # Create dummy dataset
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224),  # images
            torch.randint(0, 2, (100,))     # labels
        )
        
        # Create data loader
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        # Initialize optimizer
        self.optimizer = FinalOptimizer(
            model=self.model,
            data_loader=self.data_loader,
            output_dir=str(self.test_dir)
        )
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_hyperparameter_tuning(self):
        # Define parameter grid
        param_grid = {
            "learning_rate": [0.001, 0.0001],
            "weight_decay": [0.0001, 0.00001]
        }
        
        # Run hyperparameter tuning
        best_params = self.optimizer.fine_tune_hyperparameters(
            param_grid=param_grid,
            num_epochs=2,
            num_folds=2
        )
        
        # Check results
        self.assertIn("learning_rate", best_params)
        self.assertIn("weight_decay", best_params)
        self.assertIsInstance(best_params["learning_rate"], float)
        self.assertIsInstance(best_params["weight_decay"], float)
        
        # Check if results file exists
        self.assertTrue((self.test_dir / "best_params.json").exists())
    
    def test_inference_speed_optimization(self):
        # Run inference speed optimization
        results = self.optimizer.optimize_inference_speed(
            batch_sizes=[1, 2],
            num_warmup=1,
            num_iterations=2
        )
        
        # Check results
        self.assertIn("batch_1", results)
        self.assertIn("batch_2", results)
        self.assertIn("avg_time", results["batch_1"])
        self.assertIn("throughput", results["batch_1"])
        
        # Check if results file exists
        self.assertTrue((self.test_dir / "inference_speed.json").exists())
    
    def test_final_performance_tests(self):
        # Run final performance tests
        results = self.optimizer.run_final_performance_tests(
            test_data="dummy",
            batch_size=2,
            num_iterations=2
        )
        
        # Check results
        self.assertIn("accuracy", results)
        self.assertIn("inference_speed", results)
        self.assertIn("memory_usage", results)
        self.assertIn("robustness", results)
        
        # Check if results file exists
        self.assertTrue((self.test_dir / "final_performance.json").exists())
    
    def test_parameter_combinations(self):
        # Test parameter combination generation
        param_grid = {
            "param1": [1, 2],
            "param2": [3, 4]
        }
        
        combinations = self.optimizer._generate_param_combinations(param_grid)
        
        # Check number of combinations
        self.assertEqual(len(combinations), 4)
        
        # Check combination values
        expected_combinations = [
            {"param1": 1, "param2": 3},
            {"param1": 1, "param2": 4},
            {"param1": 2, "param2": 3},
            {"param1": 2, "param2": 4}
        ]
        
        for combo in combinations:
            self.assertIn(combo, expected_combinations)
    
    def test_fold_data_generation(self):
        # Test fold data generation
        train_loader, val_loader = self.optimizer._get_fold_data(0, 2)
        
        # Check data loader types
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        
        # Check dataset sizes
        self.assertEqual(len(train_loader.dataset) + len(val_loader.dataset), len(self.dataset))
    
    def test_training_with_params(self):
        # Test training with specific parameters
        params = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        }
        
        train_loader, val_loader = self.optimizer._get_fold_data(0, 2)
        score = self.optimizer._train_with_params(params, train_loader, val_loader, 1)
        
        # Check score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_inference_batch(self):
        # Test inference batch processing
        test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False
        )
        
        # Should not raise any exceptions
        self.optimizer._run_inference_batch(test_loader, 1)
    
    def test_accuracy_testing(self):
        # Test accuracy testing
        test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False
        )
        
        accuracy = self.optimizer._test_accuracy(test_loader)
        
        # Check accuracy
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_memory_usage_testing(self):
        # Test memory usage testing
        test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False
        )
        
        memory_usage = self.optimizer._test_memory_usage(test_loader)
        
        # Check memory usage metrics
        self.assertIn("initial_memory", memory_usage)
        self.assertIn("peak_memory", memory_usage)
        self.assertIn("memory_increase", memory_usage)
        
        # Check values
        self.assertGreaterEqual(memory_usage["peak_memory"], memory_usage["initial_memory"])
        self.assertGreaterEqual(memory_usage["memory_increase"], 0)

if __name__ == "__main__":
    unittest.main() 