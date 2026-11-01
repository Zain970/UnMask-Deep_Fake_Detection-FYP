import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import copy
import torch.fx as fx
from torch.ao.quantization import (
    get_default_qconfig,
    prepare_qat,
    convert
)
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx
)

class ModelOptimizer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'optimization_results'
    ):
        """
        Initialize model optimizer
        
        Args:
            model: Model to optimize
            device: Device to run optimization on
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
    
    def prune_model(
        self,
        amount: float = 0.3,
        method: str = 'l1_unstructured',
        modules_to_prune: Optional[List[Tuple[nn.Module, str]]] = None
    ) -> nn.Module:
        """
        Prune model weights
        
        Args:
            amount: Fraction of connections to prune
            method: Pruning method ('l1_unstructured', 'random_unstructured', 'l2_structured')
            modules_to_prune: List of (module, name) tuples to prune
            
        Returns:
            Pruned model
        """
        if modules_to_prune is None:
            # Default to pruning all Conv2d and Linear layers
            modules_to_prune = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    modules_to_prune.append((module, 'weight'))
        
        # Create a copy of the model
        pruned_model = copy.deepcopy(self.model)
        
        # Apply pruning
        for module, name in modules_to_prune:
            if method == 'l1_unstructured':
                prune.l1_unstructured(module, name=name, amount=amount)
            elif method == 'random_unstructured':
                prune.random_unstructured(module, name=name, amount=amount)
            elif method == 'l2_structured':
                prune.ln_structured(module, name=name, amount=amount, n=2, dim=0)
            else:
                raise ValueError(f"Unknown pruning method: {method}")
        
        # Make pruning permanent
        for module, name in modules_to_prune:
            prune.remove(module, name)
        
        # Log pruning results
        self._log_pruning_results(pruned_model, modules_to_prune)
        
        return pruned_model
    
    def _log_pruning_results(
        self,
        pruned_model: nn.Module,
        modules_to_prune: List[Tuple[nn.Module, str]]
    ):
        """
        Log pruning results
        
        Args:
            pruned_model: Pruned model
            modules_to_prune: List of pruned modules
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'pruning_results': {}
        }
        
        for module, name in modules_to_prune:
            original_params = module.weight.numel()
            pruned_params = torch.sum(module.weight == 0).item()
            sparsity = pruned_params / original_params
            
            results['pruning_results'][f"{module.__class__.__name__}_{name}"] = {
                'original_params': original_params,
                'pruned_params': pruned_params,
                'sparsity': sparsity
            }
        
        # Save results
        with open(self.output_dir / 'pruning_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log results
        self.logger.info('Pruning Results:')
        for module_name, stats in results['pruning_results'].items():
            self.logger.info(
                f"{module_name}: "
                f"Sparsity = {stats['sparsity']:.2%}, "
                f"Pruned = {stats['pruned_params']}/{stats['original_params']}"
            )
    
    def quantize_model(
        self,
        qconfig: Optional[Any] = None,
        backend: str = 'fbgemm',
        inplace: bool = False
    ) -> nn.Module:
        """
        Quantize model
        
        Args:
            qconfig: Quantization configuration
            backend: Quantization backend
            inplace: Whether to modify model in place
            
        Returns:
            Quantized model
        """
        if qconfig is None:
            qconfig = get_default_qconfig(backend)
        
        # Create a copy of the model if not inplace
        model_to_quantize = self.model if inplace else copy.deepcopy(self.model)
        
        # Prepare model for quantization
        model_to_quantize.eval()
        model_to_quantize.qconfig = qconfig
        
        # Prepare and convert model
        prepared_model = prepare_fx(model_to_quantize, {'': qconfig})
        quantized_model = convert_fx(prepared_model)
        
        # Log quantization results
        self._log_quantization_results(quantized_model)
        
        return quantized_model
    
    def _log_quantization_results(
        self,
        quantized_model: nn.Module
    ):
        """
        Log quantization results
        
        Args:
            quantized_model: Quantized model
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'quantization_results': {}
        }
        
        # Analyze quantized model
        for name, module in quantized_model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                results['quantization_results'][name] = {
                    'weight_dtype': str(module.weight.dtype),
                    'bias_dtype': str(module.bias.dtype) if hasattr(module, 'bias') else None
                }
        
        # Save results
        with open(self.output_dir / 'quantization_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log results
        self.logger.info('Quantization Results:')
        for module_name, stats in results['quantization_results'].items():
            self.logger.info(
                f"{module_name}: "
                f"Weight dtype = {stats['weight_dtype']}, "
                f"Bias dtype = {stats['bias_dtype']}"
            )
    
    def optimize_model(
        self,
        optimizations: List[str],
        **kwargs
    ) -> nn.Module:
        """
        Apply multiple optimizations to model
        
        Args:
            optimizations: List of optimizations to apply
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized model
        """
        optimized_model = copy.deepcopy(self.model)
        
        for optimization in optimizations:
            if optimization == 'prune':
                amount = kwargs.get('prune_amount', 0.3)
                method = kwargs.get('prune_method', 'l1_unstructured')
                optimized_model = self.prune_model(
                    amount=amount,
                    method=method,
                    modules_to_prune=kwargs.get('modules_to_prune')
                )
            elif optimization == 'quantize':
                backend = kwargs.get('quantize_backend', 'fbgemm')
                optimized_model = self.quantize_model(backend=backend)
            elif optimization == 'fuse':
                optimized_model = self._fuse_modules(optimized_model)
            elif optimization == 'graph_optimize':
                optimized_model = self._optimize_graph(optimized_model)
            else:
                raise ValueError(f"Unknown optimization: {optimization}")
        
        return optimized_model
    
    def _fuse_modules(
        self,
        model: nn.Module
    ) -> nn.Module:
        """
        Fuse consecutive modules for better performance
        
        Args:
            model: Model to fuse
            
        Returns:
            Fused model
        """
        fused_model = copy.deepcopy(model)
        
        # Fuse Conv+BN+ReLU
        for name, module in fused_model.named_children():
            if isinstance(module, nn.Sequential):
                if len(module) >= 3:
                    if (isinstance(module[0], nn.Conv2d) and
                        isinstance(module[1], nn.BatchNorm2d) and
                        isinstance(module[2], nn.ReLU)):
                        torch.quantization.fuse_modules(
                            module,
                            ['0', '1', '2'],
                            inplace=True
                        )
        
        return fused_model
    
    def _optimize_graph(
        self,
        model: nn.Module
    ) -> nn.Module:
        """
        Optimize model computation graph
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        # Convert to FX graph
        graph_model = fx.symbolic_trace(model)
        
        # Apply graph optimizations
        # 1. Remove redundant operations
        # 2. Fuse operations
        # 3. Optimize memory usage
        
        return graph_model
    
    def save_optimized_model(
        self,
        optimized_model: nn.Module,
        filename: str = 'optimized_model.pt'
    ):
        """
        Save optimized model
        
        Args:
            optimized_model: Optimized model
            filename: Output filename
        """
        # Save model
        torch.save(optimized_model.state_dict(), self.output_dir / filename)
        
        # Save optimization info
        info = {
            'timestamp': datetime.now().isoformat(),
            'optimization_info': {
                'model_size': sum(p.numel() for p in optimized_model.parameters()),
                'model_size_bytes': sum(p.numel() * p.element_size() for p in optimized_model.parameters())
            }
        }
        
        with open(self.output_dir / 'optimization_info.json', 'w') as f:
            json.dump(info, f, indent=4)
        
        self.logger.info(f'Optimized model saved to {self.output_dir / filename}')
    
    def load_optimized_model(
        self,
        filename: str = 'optimized_model.pt'
    ) -> nn.Module:
        """
        Load optimized model
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded model
        """
        # Load model
        state_dict = torch.load(self.output_dir / filename)
        
        # Create model and load state
        model = copy.deepcopy(self.model)
        model.load_state_dict(state_dict)
        
        return model 