import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

class FeatureExtractor:
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        feature_dim: int = 2048
    ):
        """
        Initialize the feature extractor
        
        Args:
            model_name: Name of the CNN model to use
            pretrained: Whether to use pretrained weights
            device: Device to run the model on
            feature_dim: Dimension of extracted features
        """
        self.device = device
        self.feature_dim = feature_dim
        
        # Load pretrained model
        self.model = self._load_model(model_name, pretrained)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize feature normalizer
        self.normalizer = StandardScaler()
        
        # Initialize PCA for visualization
        self.pca = PCA(n_components=2)
        
        # Initialize feature cache
        self.feature_cache = {}
    
    def _load_model(
        self,
        model_name: str,
        pretrained: bool
    ) -> nn.Module:
        """
        Load pretrained CNN model
        
        Args:
            model_name: Name of the model to load
            pretrained: Whether to use pretrained weights
            
        Returns:
            Loaded model
        """
        model_map = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
            'densenet121': models.densenet121
        }
        
        if model_name not in model_map:
            raise ValueError(f"Model {model_name} not supported")
        
        model = model_map[model_name](pretrained=pretrained)
        
        # Remove classification layer
        if 'resnet' in model_name:
            model = nn.Sequential(*list(model.children())[:-1])
        elif 'vgg' in model_name:
            model = nn.Sequential(*list(model.features.children()))
        elif 'densenet' in model_name:
            model = nn.Sequential(*list(model.features.children()))
        
        return model
    
    def extract_features(
        self,
        images: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract features from images
        
        Args:
            images: Batch of input images
            normalize: Whether to normalize features
            
        Returns:
            Extracted features
        """
        # Move images to device
        images = images.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(images)
            features = features.view(features.size(0), -1)
        
        # Normalize features
        if normalize:
            features = self.normalize_features(features)
        
        return features
    
    def normalize_features(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize features using StandardScaler
        
        Args:
            features: Input features
            
        Returns:
            Normalized features
        """
        # Convert to numpy for sklearn
        features_np = features.cpu().numpy()
        
        # Fit normalizer if not already fitted
        if not hasattr(self.normalizer, 'mean_'):
            self.normalizer.fit(features_np)
        
        # Normalize features
        normalized = self.normalizer.transform(features_np)
        
        # Convert back to tensor
        return torch.tensor(normalized, device=self.device)
    
    def visualize_features(
        self,
        features: torch.Tensor,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize features using PCA
        
        Args:
            features: Extracted features
            labels: Optional class labels
            save_path: Optional path to save visualization
        """
        # Convert to numpy
        features_np = features.cpu().numpy()
        
        # Fit PCA if not already fitted
        if not hasattr(self.pca, 'components_'):
            self.pca.fit(features_np)
        
        # Transform features
        reduced = self.pca.transform(features_np)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # Scatter plot with labels
            sns.scatterplot(
                x=reduced[:, 0],
                y=reduced[:, 1],
                hue=labels,
                palette='viridis',
                alpha=0.7
            )
        else:
            # Simple scatter plot
            plt.scatter(
                reduced[:, 0],
                reduced[:, 1],
                alpha=0.7
            )
        
        plt.title('Feature Space Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def cache_features(
        self,
        key: str,
        features: torch.Tensor
    ):
        """
        Cache extracted features
        
        Args:
            key: Cache key
            features: Features to cache
        """
        self.feature_cache[key] = features.cpu()
    
    def get_cached_features(
        self,
        key: str
    ) -> Optional[torch.Tensor]:
        """
        Get cached features
        
        Args:
            key: Cache key
            
        Returns:
            Cached features if available
        """
        return self.feature_cache.get(key)
    
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
    
    def get_feature_importance(
        self,
        features: torch.Tensor,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Calculate feature importance
        
        Args:
            features: Extracted features
            labels: Class labels
            
        Returns:
            Feature importance scores
        """
        # Convert to numpy
        features_np = features.cpu().numpy()
        
        # Calculate feature importance using variance
        importance = np.var(features_np, axis=0)
        
        return importance
    
    def visualize_feature_importance(
        self,
        importance: np.ndarray,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Visualize feature importance
        
        Args:
            importance: Feature importance scores
            top_n: Number of top features to show
            save_path: Optional path to save visualization
        """
        # Get top features
        top_indices = np.argsort(importance)[-top_n:]
        top_importance = importance[top_indices]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.barh(range(top_n), top_importance)
        plt.yticks(range(top_n), [f'Feature {i}' for i in top_indices])
        plt.title('Top Feature Importance')
        plt.xlabel('Importance Score')
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def batch_extract_features(
        self,
        images: List[torch.Tensor],
        batch_size: int = 32,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract features from a batch of images
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            normalize: Whether to normalize features
            
        Returns:
            Extracted features
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch = torch.stack(batch)
            features = self.extract_features(batch, normalize)
            all_features.append(features)
        
        return torch.cat(all_features, dim=0) 