from typing import Dict, Any

class ModelConfig:
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        num_classes: int = 2,
        use_attention: bool = True,
        attention_reduction: int = 16,
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 32,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize model configuration
        
        Args:
            backbone_name: Name of the backbone model
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes
            use_attention: Whether to use attention mechanism
            attention_reduction: Channel reduction ratio for attention
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            weight_decay: Weight decay
            batch_size: Batch size
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            device: Device to run model on
        """
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.attention_reduction = attention_reduction
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Configuration dictionary
        """
        return {
            'backbone_name': self.backbone_name,
            'pretrained': self.pretrained,
            'num_classes': self.num_classes,
            'use_attention': self.use_attention,
            'attention_reduction': self.attention_reduction,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Model configuration
        """
        return cls(**config_dict)
    
    def save(self, path: str):
        """
        Save configuration to file
        
        Args:
            path: Path to save configuration
        """
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """
        Load configuration from file
        
        Args:
            path: Path to configuration file
            
        Returns:
            Model configuration
        """
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict) 