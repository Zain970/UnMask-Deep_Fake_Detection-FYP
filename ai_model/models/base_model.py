import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple, Dict
import math

class AttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16
    ):
        """
        Initialize attention block
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction ratio
        """
        super(AttentionBlock, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        x_spatial = x_channel * spatial_weights
        
        return x_spatial

class BaseModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 2,
        use_attention: bool = True,
        attention_reduction: int = 16,
        dropout_rate: float = 0.5
    ):
        """
        Initialize base model
        
        Args:
            backbone_name: Name of the backbone model
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes
            use_attention: Whether to use attention mechanism
            attention_reduction: Channel reduction ratio for attention
            dropout_rate: Dropout rate
        """
        super(BaseModel, self).__init__()
        
        # Load backbone
        self.backbone = self._load_backbone(backbone_name, pretrained)
        self.backbone_name = backbone_name
        
        # Get feature dimensions
        if 'resnet' in backbone_name:
            self.feature_dim = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        elif 'vgg' in backbone_name:
            self.feature_dim = 512
            self.backbone = nn.Sequential(*list(self.backbone.features.children()))
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        # Initialize attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(
                self.feature_dim,
                reduction_ratio=attention_reduction
            )
        
        # Initialize classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _load_backbone(
        self,
        backbone_name: str,
        pretrained: bool
    ) -> nn.Module:
        """
        Load backbone model
        
        Args:
            backbone_name: Name of the backbone model
            pretrained: Whether to use pretrained weights
            
        Returns:
            Loaded backbone model
        """
        model_map = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19
        }
        
        if backbone_name not in model_map:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        return model_map[backbone_name](pretrained=pretrained)
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            return_features: Whether to return features
            
        Returns:
            Output tensor or tuple of (output, features)
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Classify
        output = self.classifier(features)
        
        if return_features:
            return output, features
        return output
    
    def get_config(self) -> Dict:
        """
        Get model configuration
        
        Returns:
            Model configuration dictionary
        """
        return {
            'backbone_name': self.backbone_name,
            'feature_dim': self.feature_dim,
            'use_attention': self.use_attention,
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
    ) -> Tuple['BaseModel', Dict]:
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