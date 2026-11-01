import unittest
import torch
import numpy as np
import time
import psutil
import gc
from pathlib import Path
import sys
import os
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ai_model.models.base_model import BaseModel
from ai_model.preprocessing.image_preprocessing import ImagePreprocessor
from ai_model.temporal.temporal_analysis import FrameSequenceProcessor
from ai_model.ensemble.ensemble_learner import EnsembleLearner
from ai_model.interpretability.model_interpreter import ModelInterpreter
from ai_model.training.trainer import ModelTrainer
from ai_model.validation.validator import ModelValidator
from ai_model.optimization.model_optimizer import ModelOptimizer
from ai_model.deployment.model_deployer import ModelDeployer

class TestModelPerformance(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel()
        self.batch_sizes = [1, 4, 8, 16]
        self.input_shape = (1, 3, 224, 224)
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
        
        # Set up CUDA if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    def measure_memory_usage(self):
        if self.device == 'cuda':
            return torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024**2  # MB
    
    def test_inference_speed(self):
        results = {}
        for batch_size in self.batch_sizes:
            # Create input tensor
            x = torch.randn(batch_size, *self.input_shape[1:]).to(self.device)
            
            # Warm up
            for _ in range(5):
                _ = self.model(x)
            
            # Measure inference time
            start_time = time.time()
            for _ in range(10):
                _ = self.model(x)
            end_time = time.time()
            
            # Calculate average time per inference
            avg_time = (end_time - start_time) / 10
            results[batch_size] = avg_time
            
            # Log results
            print(f"Batch size {batch_size}: {avg_time:.4f} seconds per inference")
        
        # Verify performance improvement with batch size
        self.assertTrue(results[16] < results[1] * 16)
    
    def test_memory_usage(self):
        results = {}
        for batch_size in self.batch_sizes:
            # Clear memory
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Measure initial memory
            initial_memory = self.measure_memory_usage()
            
            # Create input tensor
            x = torch.randn(batch_size, *self.input_shape[1:]).to(self.device)
            
            # Run inference
            _ = self.model(x)
            
            # Measure final memory
            final_memory = self.measure_memory_usage()
            
            # Calculate memory increase
            memory_increase = final_memory - initial_memory
            results[batch_size] = memory_increase
            
            # Log results
            print(f"Batch size {batch_size}: {memory_increase:.2f} MB memory increase")
        
        # Verify memory usage scales reasonably with batch size
        self.assertTrue(results[16] < results[1] * 16)
    
    def test_training_performance(self):
        trainer = ModelTrainer(self.model)
        batch_size = 4
        
        # Create dummy data
        train_data = torch.randn(100, *self.input_shape[1:]).to(self.device)
        train_labels = torch.randint(0, 2, (100,)).to(self.device)
        val_data = torch.randn(20, *self.input_shape[1:]).to(self.device)
        val_labels = torch.randint(0, 2, (20,)).to(self.device)
        
        # Measure training time
        start_time = time.time()
        trainer.train(
            train_data,
            train_labels,
            val_data,
            val_labels,
            epochs=2,
            batch_size=batch_size
        )
        end_time = time.time()
        
        # Log results
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")
        
        # Verify training time is reasonable
        self.assertTrue(training_time < 300)  # Less than 5 minutes
    
    def test_optimization_impact(self):
        optimizer = ModelOptimizer(self.model)
        batch_size = 4
        
        # Measure original model performance
        x = torch.randn(batch_size, *self.input_shape[1:]).to(self.device)
        
        # Warm up
        for _ in range(5):
            _ = self.model(x)
        
        # Measure original inference time
        start_time = time.time()
        for _ in range(10):
            _ = self.model(x)
        original_time = (time.time() - start_time) / 10
        
        # Optimize model
        optimized_model = optimizer.optimize_model(
            prune_method='l1_unstructured',
            quantize_backend='fbgemm'
        )
        
        # Measure optimized model performance
        start_time = time.time()
        for _ in range(10):
            _ = optimized_model(x)
        optimized_time = (time.time() - start_time) / 10
        
        # Log results
        print(f"Original model: {original_time:.4f} seconds per inference")
        print(f"Optimized model: {optimized_time:.4f} seconds per inference")
        
        # Verify optimization improved performance
        self.assertTrue(optimized_time < original_time)
    
    def test_deployment_performance(self):
        deployer = ModelDeployer(self.model)
        batch_size = 4
        
        # Convert to ONNX
        onnx_path = self.test_dir / 'model.onnx'
        deployer.convert_to_onnx(onnx_path)
        
        # Optimize with TensorRT
        trt_path = self.test_dir / 'model.engine'
        deployer.optimize_tensorrt(onnx_path, trt_path)
        
        # Test performance
        performance = deployer.test_performance()
        
        # Log results
        print(f"ONNX latency: {performance['onnx_latency']:.4f} seconds")
        print(f"TensorRT latency: {performance['trt_latency']:.4f} seconds")
        
        # Verify TensorRT optimization
        self.assertTrue(performance['trt_latency'] < performance['onnx_latency'])
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if self.device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main() 