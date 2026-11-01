import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Union, Any, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime
import shutil

class CheckpointManager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str = 'checkpoints',
        max_checkpoints: int = 5,
        metric_name: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Initialize checkpoint manager
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            scheduler: Learning rate scheduler to checkpoint
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            metric_name: Metric to track for best model
            mode: Whether to minimize or maximize the metric
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize best metric value
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model
            additional_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if additional_info is not None:
            checkpoint['additional_info'] = additional_info
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
            # Save best metrics
            with open(self.checkpoint_dir / 'best_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if load_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def should_save_checkpoint(
        self,
        metrics: Dict[str, float]
    ) -> bool:
        """
        Check if checkpoint should be saved
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Whether to save checkpoint
        """
        if self.metric_name not in metrics:
            return False
        
        current_metric = metrics[self.metric_name]
        
        if self.mode == 'min':
            should_save = current_metric < self.best_metric
        else:
            should_save = current_metric > self.best_metric
        
        if should_save:
            self.best_metric = current_metric
        
        return should_save
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint
        
        Returns:
            Path to best checkpoint if exists
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        return best_path if best_path.exists() else None
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to latest checkpoint
        
        Returns:
            Path to latest checkpoint if exists
        """
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort checkpoints by epoch
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.max_checkpoints]:
            checkpoint.unlink()
    
    def resume_training(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint, if None use latest
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError('No checkpoints found')
        
        self.logger.info(f'Resuming training from checkpoint: {checkpoint_path}')
        
        # Load checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # Update best metric
        if self.metric_name in checkpoint['metrics']:
            self.best_metric = checkpoint['metrics'][self.metric_name]
        
        return checkpoint
    
    def save_model_config(
        self,
        config: Dict[str, Any]
    ):
        """
        Save model configuration
        
        Args:
            config: Model configuration dictionary
        """
        config_path = self.checkpoint_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def load_model_config(self) -> Dict[str, Any]:
        """
        Load model configuration
        
        Returns:
            Model configuration dictionary
        """
        config_path = self.checkpoint_dir / 'model_config.json'
        if not config_path.exists():
            raise ValueError('Model configuration not found')
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def export_model(
        self,
        export_path: Union[str, Path],
        input_shape: Tuple[int, ...]
    ):
        """
        Export model to ONNX format
        
        Args:
            export_path: Path to save exported model
            input_shape: Input tensor shape
        """
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export model
        torch.onnx.export(
            self.model,
            dummy_input,
            export_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    def backup_checkpoints(
        self,
        backup_dir: Union[str, Path]
    ):
        """
        Backup checkpoints
        
        Args:
            backup_dir: Directory to save backup
        """
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoints
        for checkpoint in self.checkpoint_dir.glob('*.pt'):
            shutil.copy2(checkpoint, backup_dir / checkpoint.name)
        
        # Copy configuration
        config_path = self.checkpoint_dir / 'model_config.json'
        if config_path.exists():
            shutil.copy2(config_path, backup_dir / 'model_config.json')
        
        # Copy best metrics
        metrics_path = self.checkpoint_dir / 'best_metrics.json'
        if metrics_path.exists():
            shutil.copy2(metrics_path, backup_dir / 'best_metrics.json') 