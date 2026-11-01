import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import onnx
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import shutil
from torch.onnx import export as onnx_export
from torch2trt import torch2trt

class ModelDeployer:
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'deployment_results'
    ):
        """
        Initialize model deployer
        
        Args:
            model: Model to deploy
            input_shape: Input tensor shape
            device: Device to run conversion on
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.input_shape = input_shape
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
    
    def convert_to_onnx(
        self,
        filename: str = 'model.onnx',
        opset_version: int = 12,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> str:
        """
        Convert model to ONNX format
        
        Args:
            filename: Output filename
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration
            
        Returns:
            Path to ONNX model
        """
        # Create dummy input
        dummy_input = torch.randn(self.input_shape, device=self.device)
        
        # Default dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export model
        onnx_path = self.output_dir / filename
        onnx_export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Log conversion results
        self._log_conversion_results('onnx', onnx_path)
        
        return str(onnx_path)
    
    def optimize_with_tensorrt(
        self,
        filename: str = 'model.trt',
        fp16_mode: bool = True,
        max_workspace_size: int = 1 << 30,
        max_batch_size: int = 32
    ) -> str:
        """
        Optimize model with TensorRT
        
        Args:
            filename: Output filename
            fp16_mode: Whether to use FP16 precision
            max_workspace_size: Maximum workspace size
            max_batch_size: Maximum batch size
            
        Returns:
            Path to TensorRT engine
        """
        # Create dummy input
        dummy_input = torch.randn(self.input_shape, device=self.device)
        
        # Convert to TensorRT
        trt_model = torch2trt(
            self.model,
            [dummy_input],
            fp16_mode=fp16_mode,
            max_workspace_size=max_workspace_size,
            max_batch_size=max_batch_size
        )
        
        # Save TensorRT engine
        trt_path = self.output_dir / filename
        torch.save(trt_model.state_dict(), trt_path)
        
        # Log conversion results
        self._log_conversion_results('tensorrt', trt_path)
        
        return str(trt_path)
    
    def test_onnx_model(
        self,
        onnx_path: str,
        test_input: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Test ONNX model
        
        Args:
            onnx_path: Path to ONNX model
            test_input: Test input tensor
            
        Returns:
            Tuple of (output tensor, inference time)
        """
        # Create ONNX runtime session
        session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        input_data = {input_name: test_input.cpu().numpy()}
        
        # Run inference
        import time
        start_time = time.time()
        output = session.run(None, input_data)
        inference_time = time.time() - start_time
        
        return torch.from_numpy(output[0]), inference_time
    
    def test_tensorrt_model(
        self,
        trt_path: str,
        test_input: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Test TensorRT model
        
        Args:
            trt_path: Path to TensorRT engine
            test_input: Test input tensor
            
        Returns:
            Tuple of (output tensor, inference time)
        """
        # Load TensorRT engine
        trt_model = torch2trt(self.model, [test_input])
        trt_model.load_state_dict(torch.load(trt_path))
        
        # Run inference
        import time
        start_time = time.time()
        with torch.no_grad():
            output = trt_model(test_input)
        inference_time = time.time() - start_time
        
        return output, inference_time
    
    def compare_performance(
        self,
        test_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of different model formats
        
        Args:
            test_input: Test input tensor
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary of performance metrics
        """
        # Convert models
        onnx_path = self.convert_to_onnx()
        trt_path = self.optimize_with_tensorrt()
        
        # Test original model
        original_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(test_input)
                original_times.append(time.time() - start_time)
        
        # Test ONNX model
        onnx_times = []
        for _ in range(num_runs):
            _, inference_time = self.test_onnx_model(onnx_path, test_input)
            onnx_times.append(inference_time)
        
        # Test TensorRT model
        trt_times = []
        for _ in range(num_runs):
            _, inference_time = self.test_tensorrt_model(trt_path, test_input)
            trt_times.append(inference_time)
        
        # Calculate statistics
        stats = {
            'original': {
                'mean_time': np.mean(original_times),
                'std_time': np.std(original_times),
                'min_time': np.min(original_times),
                'max_time': np.max(original_times)
            },
            'onnx': {
                'mean_time': np.mean(onnx_times),
                'std_time': np.std(onnx_times),
                'min_time': np.min(onnx_times),
                'max_time': np.max(onnx_times)
            },
            'tensorrt': {
                'mean_time': np.mean(trt_times),
                'std_time': np.std(trt_times),
                'min_time': np.min(trt_times),
                'max_time': np.max(trt_times)
            }
        }
        
        # Log results
        self._log_performance_results(stats)
        
        return stats
    
    def _log_conversion_results(
        self,
        format: str,
        model_path: Path
    ):
        """
        Log conversion results
        
        Args:
            format: Model format
            model_path: Path to converted model
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'format': format,
            'model_path': str(model_path),
            'model_size': os.path.getsize(model_path)
        }
        
        # Save results
        with open(self.output_dir / f'{format}_conversion_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log results
        self.logger.info(f'{format.upper()} Conversion Results:')
        self.logger.info(f'Model saved to: {model_path}')
        self.logger.info(f'Model size: {results["model_size"] / 1024 / 1024:.2f} MB')
    
    def _log_performance_results(
        self,
        stats: Dict[str, Dict[str, float]]
    ):
        """
        Log performance results
        
        Args:
            stats: Performance statistics
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'performance_stats': stats
        }
        
        # Save results
        with open(self.output_dir / 'performance_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log results
        self.logger.info('Performance Comparison Results:')
        for format, metrics in stats.items():
            self.logger.info(f'\n{format.upper()} Model:')
            for metric, value in metrics.items():
                self.logger.info(f'{metric}: {value:.6f} seconds') 