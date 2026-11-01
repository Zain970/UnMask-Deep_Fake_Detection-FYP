import unittest
import torch
import numpy as np
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

class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel()
        self.trainer = ModelTrainer(self.model)
        self.batch_size = 4
        self.input_shape = (1, 3, 224, 224)
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
    
    def test_full_training_cycle(self):
        # Create dummy data
        train_data = torch.randn(100, *self.input_shape[1:])
        train_labels = torch.randint(0, 2, (100,))
        val_data = torch.randn(20, *self.input_shape[1:])
        val_labels = torch.randint(0, 2, (20,))
        
        # Train model
        self.trainer.train(
            train_data,
            train_labels,
            val_data,
            val_labels,
            epochs=2,
            batch_size=self.batch_size
        )
        
        # Verify training results
        self.assertTrue(hasattr(self.trainer, 'best_metrics'))
        self.assertTrue('val_loss' in self.trainer.best_metrics)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)

class TestValidationPipeline(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel()
        self.validator = ModelValidator(self.model)
        self.batch_size = 4
        self.input_shape = (1, 3, 224, 224)
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
    
    def test_full_validation_cycle(self):
        # Create dummy data
        val_data = torch.randn(20, *self.input_shape[1:])
        val_labels = torch.randint(0, 2, (20,))
        
        # Validate model
        metrics = self.validator.validate(val_data, val_labels)
        
        # Verify validation results
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('precision' in metrics)
        self.assertTrue('recall' in metrics)
        self.assertTrue('f1' in metrics)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)

class TestOptimizationPipeline(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel()
        self.optimizer = ModelOptimizer(self.model)
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
    
    def test_full_optimization_cycle(self):
        # Optimize model
        optimized_model = self.optimizer.optimize_model(
            prune_method='l1_unstructured',
            quantize_backend='fbgemm'
        )
        
        # Verify optimization results
        self.assertIsInstance(optimized_model, torch.nn.Module)
        self.assertTrue(hasattr(self.optimizer, 'optimization_metrics'))
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)

class TestDeploymentPipeline(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel()
        self.deployer = ModelDeployer(self.model)
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
    
    def test_full_deployment_cycle(self):
        # Convert to ONNX
        onnx_path = self.test_dir / 'model.onnx'
        self.deployer.convert_to_onnx(onnx_path)
        self.assertTrue(onnx_path.exists())
        
        # Optimize with TensorRT
        trt_path = self.test_dir / 'model.engine'
        self.deployer.optimize_tensorrt(onnx_path, trt_path)
        self.assertTrue(trt_path.exists())
        
        # Test performance
        performance = self.deployer.test_performance()
        self.assertTrue('onnx_latency' in performance)
        self.assertTrue('trt_latency' in performance)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel()
        self.preprocessor = ImagePreprocessor()
        self.processor = FrameSequenceProcessor()
        self.ensemble = EnsembleLearner([self.model])
        self.interpreter = ModelInterpreter(self.model)
        self.batch_size = 4
        self.input_shape = (1, 3, 224, 224)
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
    
    def test_full_pipeline(self):
        # Create dummy video frames
        frames = torch.randn(32, *self.input_shape[1:])
        
        # Preprocess frames
        processed_frames = self.preprocessor.preprocess_batch(frames)
        
        # Create sequences
        sequences = self.processor.create_sequences(processed_frames)
        
        # Make predictions
        predictions, confidences = self.ensemble.predict(sequences[0])
        
        # Interpret results
        heatmap = self.interpreter.compute_grad_cam(sequences[0])
        
        # Verify pipeline results
        self.assertEqual(processed_frames.shape, frames.shape)
        self.assertIsInstance(sequences, list)
        self.assertEqual(predictions.shape, (self.batch_size,))
        self.assertEqual(confidences.shape, (self.batch_size, 2))
        self.assertEqual(heatmap.shape, self.input_shape[2:])
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main() 