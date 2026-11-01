import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
import io

class ModelInterpreter:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'interpretability_results'
    ):
        """
        Initialize model interpreter
        
        Args:
            model: Model to interpret
            device: Device to run interpretation on
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set model to eval mode
        self.model.eval()
        
        # Register hooks for feature maps
        self.feature_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for feature map extraction"""
        def hook_fn(module, input, output, name):
            self.feature_maps[name] = output.detach()
        
        # Register hooks for all convolutional layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
    
    def compute_grad_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_name: str = 'last_conv'
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index
            layer_name: Name of target layer
            
        Returns:
            Grad-CAM heatmap
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        target = output[:, target_class]
        
        # Backward pass
        target.backward()
        
        # Get gradients and feature maps
        gradients = self.model.get_activations_gradient()
        feature_maps = self.feature_maps[layer_name]
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight feature maps by pooled gradients
        for i in range(feature_maps.size(1)):
            feature_maps[:, i, :, :] *= pooled_gradients[i]
        
        # Average feature maps
        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        
        # ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        
        return heatmap.cpu().numpy()
    
    def visualize_features(
        self,
        input_tensor: torch.Tensor,
        layer_name: str,
        num_features: int = 16,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize feature maps
        
        Args:
            input_tensor: Input tensor
            layer_name: Name of target layer
            num_features: Number of features to visualize
            save_path: Path to save visualization
            
        Returns:
            Feature visualization
        """
        # Forward pass
        _ = self.model(input_tensor)
        
        # Get feature maps
        feature_maps = self.feature_maps[layer_name]
        
        # Select subset of features
        feature_maps = feature_maps[0, :num_features]
        
        # Normalize feature maps
        feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())
        
        # Create grid visualization
        grid_size = int(np.ceil(np.sqrt(num_features)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        
        for i, ax in enumerate(axes.flat):
            if i < num_features:
                ax.imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save or return visualization
        if save_path:
            plt.savefig(self.output_dir / save_path)
            plt.close()
            return None
        else:
            # Convert to numpy array
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close()
            return np.array(img)
    
    def visualize_attention(
        self,
        input_tensor: torch.Tensor,
        attention_layer: str,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize attention maps
        
        Args:
            input_tensor: Input tensor
            attention_layer: Name of attention layer
            save_path: Path to save visualization
            
        Returns:
            Attention visualization
        """
        # Forward pass
        _ = self.model(input_tensor)
        
        # Get attention weights
        attention_weights = self.feature_maps[attention_layer]
        
        # Average attention weights
        attention_map = torch.mean(attention_weights, dim=1).squeeze()
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            attention_map.cpu().numpy(),
            cmap='YlOrRd',
            square=True,
            cbar=True
        )
        plt.title('Attention Map')
        
        # Save or return visualization
        if save_path:
            plt.savefig(self.output_dir / save_path)
            plt.close()
            return None
        else:
            # Convert to numpy array
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            plt.close()
            return np.array(img)
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Overlay heatmap on image
        
        Args:
            image: Original image
            heatmap: Heatmap to overlay
            alpha: Overlay transparency
            save_path: Path to save visualization
            
        Returns:
            Overlayed image
        """
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Create colormap
        cmap = plt.cm.jet
        heatmap_colored = cmap(heatmap)[..., :3]
        
        # Overlay heatmap on image
        overlayed = cv2.addWeighted(
            image.astype(np.float32),
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        # Save or return visualization
        if save_path:
            cv2.imwrite(str(self.output_dir / save_path), overlayed)
            return None
        else:
            return overlayed
    
    def save_interpretation(
        self,
        interpretation: Dict[str, Any],
        filename: str = 'interpretation_results.json'
    ):
        """
        Save interpretation results
        
        Args:
            interpretation: Interpretation results
            filename: Output filename
        """
        # Add timestamp
        interpretation['timestamp'] = datetime.now().isoformat()
        
        # Save results
        with open(self.output_dir / filename, 'w') as f:
            json.dump(interpretation, f, indent=4)
        
        self.logger.info(f'Interpretation results saved to {self.output_dir / filename}')
    
    def load_interpretation(
        self,
        filename: str = 'interpretation_results.json'
    ) -> Dict[str, Any]:
        """
        Load interpretation results
        
        Args:
            filename: Input filename
            
        Returns:
            Interpretation results
        """
        # Load results
        with open(self.output_dir / filename, 'r') as f:
            interpretation = json.load(f)
        
        return interpretation 