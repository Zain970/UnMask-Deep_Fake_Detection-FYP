import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ai_model.models.base_model import BaseModel
from ai_model.preprocessing.image_preprocessing import ImagePreprocessor
from ai_model.temporal.temporal_analysis import FrameSequenceProcessor
from ai_model.ensemble.ensemble_learner import EnsembleLearner
from ai_model.interpretability.model_interpreter import ModelInterpreter

class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel()
        self.input_shape = (1, 3, 224, 224)
        self.batch_size = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_model_initialization(self):
        self.assertIsInstance(self.model, nn.Module)
        self.assertTrue(hasattr(self.model, 'backbone'))
        self.assertTrue(hasattr(self.model, 'attention'))
    
    def test_forward_pass(self):
        x = torch.randn(self.batch_size, *self.input_shape[1:])
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 2))
    
    def test_model_saving(self):
        save_path = Path('test_model.pt')
        self.model.save_model(save_path)
        self.assertTrue(save_path.exists())
        save_path.unlink()
    
    def test_model_loading(self):
        save_path = Path('test_model.pt')
        self.model.save_model(save_path)
        new_model = BaseModel()
        new_model.load_model(save_path)
        self.assertTrue(torch.allclose(
            self.model.state_dict()['backbone.conv1.weight'],
            new_model.state_dict()['backbone.conv1.weight']
        ))
        save_path.unlink()

class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        self.image_shape = (3, 224, 224)
        self.batch_size = 4
    
    def test_single_image_preprocessing(self):
        image = torch.randn(*self.image_shape)
        processed = self.preprocessor.preprocess_image(image)
        self.assertEqual(processed.shape, self.image_shape)
    
    def test_batch_preprocessing(self):
        batch = torch.randn(self.batch_size, *self.image_shape)
        processed = self.preprocessor.preprocess_batch(batch)
        self.assertEqual(processed.shape, (self.batch_size, *self.image_shape))
    
    def test_face_detection(self):
        image = torch.randn(*self.image_shape)
        faces = self.preprocessor.detect_faces(image)
        self.assertIsInstance(faces, list)
    
    def test_face_alignment(self):
        image = torch.randn(*self.image_shape)
        bbox = [0, 0, 100, 100]
        aligned = self.preprocessor.align_face(image, bbox)
        self.assertEqual(aligned.shape, self.image_shape)

class TestFrameSequenceProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FrameSequenceProcessor()
        self.frame_shape = (3, 224, 224)
        self.sequence_length = 16
        self.batch_size = 4
    
    def test_sequence_creation(self):
        frames = torch.randn(self.sequence_length, *self.frame_shape)
        sequences = self.processor.create_sequences(frames)
        self.assertIsInstance(sequences, list)
        self.assertTrue(all(s.shape == (self.processor.sequence_length, *self.frame_shape) for s in sequences))
    
    def test_batch_processing(self):
        sequences = [torch.randn(self.processor.sequence_length, *self.frame_shape) for _ in range(self.batch_size)]
        processed = self.processor.process_batch(sequences)
        self.assertEqual(processed.shape, (self.batch_size, self.processor.sequence_length, *self.frame_shape))

class TestEnsembleLearner(unittest.TestCase):
    def setUp(self):
        self.models = [BaseModel() for _ in range(3)]
        self.ensemble = EnsembleLearner(self.models)
        self.input_shape = (1, 3, 224, 224)
        self.batch_size = 4
    
    def test_ensemble_prediction(self):
        x = torch.randn(self.batch_size, *self.input_shape[1:])
        predictions, confidences = self.ensemble.predict(x)
        self.assertEqual(predictions.shape, (self.batch_size,))
        self.assertEqual(confidences.shape, (self.batch_size, 2))
    
    def test_voting_mechanisms(self):
        x = torch.randn(self.batch_size, *self.input_shape[1:])
        for method in ['soft', 'hard', 'weighted']:
            predictions, confidences = self.ensemble.predict(x, voting_method=method)
            self.assertEqual(predictions.shape, (self.batch_size,))
            self.assertEqual(confidences.shape, (self.batch_size, 2))

class TestModelInterpreter(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel()
        self.interpreter = ModelInterpreter(self.model)
        self.input_shape = (1, 3, 224, 224)
    
    def test_grad_cam(self):
        x = torch.randn(*self.input_shape)
        heatmap = self.interpreter.compute_grad_cam(x)
        self.assertEqual(heatmap.shape, self.input_shape[2:])
    
    def test_feature_visualization(self):
        x = torch.randn(*self.input_shape)
        visualization = self.interpreter.visualize_features(x, 'backbone.conv1')
        self.assertIsInstance(visualization, np.ndarray)
    
    def test_attention_visualization(self):
        x = torch.randn(*self.input_shape)
        visualization = self.interpreter.visualize_attention(x, 'attention')
        self.assertIsInstance(visualization, np.ndarray)

if __name__ == '__main__':
    unittest.main() 