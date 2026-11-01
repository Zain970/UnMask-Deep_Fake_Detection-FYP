import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np

class FrameSequenceProcessor:
    def __init__(
        self,
        sequence_length: int = 16,
        stride: int = 1,
        overlap: float = 0.5
    ):
        """
        Initialize frame sequence processor
        
        Args:
            sequence_length: Number of frames in each sequence
            stride: Stride between sequences
            overlap: Overlap ratio between sequences
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.overlap = overlap
    
    def process_frames(
        self,
        frames: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Process frames into sequences
        
        Args:
            frames: Input frames tensor of shape (batch_size, channels, height, width)
            
        Returns:
            List of frame sequences
        """
        sequences = []
        num_frames = frames.size(0)
        
        # Calculate overlap in frames
        overlap_frames = int(self.sequence_length * self.overlap)
        
        # Process frames into sequences
        for i in range(0, num_frames - self.sequence_length + 1, self.stride):
            sequence = frames[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return sequences

class TemporalFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2
    ):
        """
        Initialize temporal feature extractor
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            num_layers: Number of LSTM layers
        """
        super(TemporalFeatureExtractor, self).__init__()
        
        # 3D convolution for spatial-temporal feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract temporal features from input sequence
        
        Args:
            x: Input sequence tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Tuple of (temporal features, attention weights)
        """
        # Apply 3D convolution
        x = self.conv3d(x)
        
        # Reshape for LSTM
        batch_size, channels, seq_len, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, seq_len, -1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)
        
        return attended_features, attention_weights

class SequenceModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        """
        Initialize sequence model
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(SequenceModel, self).__init__()
        
        # Feature extractor
        self.feature_extractor = TemporalFeatureExtractor(
            input_channels=input_channels,
            hidden_channels=hidden_channels
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input sequence
        
        Args:
            x: Input sequence tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Tuple of (predictions, attention weights)
        """
        # Extract features
        features, attention_weights = self.feature_extractor(x)
        
        # Classify
        predictions = self.classifier(features)
        
        return predictions, attention_weights
    
    def get_config(self) -> Dict:
        """
        Get model configuration
        
        Returns:
            Model configuration dictionary
        """
        return {
            'input_channels': self.feature_extractor.conv3d[0].in_channels,
            'hidden_channels': self.feature_extractor.conv3d[0].out_channels,
            'num_classes': self.classifier[-1].out_features
        }
    
    def save_model(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None
    ):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_model(
        cls,
        path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Tuple['SequenceModel', Dict]:
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
            
        Returns:
            Loaded model and checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved configuration
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        return model, checkpoint 