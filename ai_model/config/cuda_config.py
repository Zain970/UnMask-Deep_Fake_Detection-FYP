import os
import torch
import subprocess
from typing import Dict, Optional

class CUDASetup:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.cuda_version = self._get_cuda_version()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.device_info = self._get_device_info()
        
    def _get_cuda_version(self) -> str:
        """Get CUDA version from nvcc"""
        try:
            output = subprocess.check_output(['nvcc', '--version'])
            version_line = output.decode('utf-8').split('\n')[-2]
            version = version_line.split('release ')[1].split(',')[0]
            return version
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "Not available"
    
    def _get_device_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about available CUDA devices"""
        device_info = {}
        if self.cuda_available:
            for i in range(self.device_count):
                device = torch.cuda.get_device_properties(i)
                device_info[f"cuda:{i}"] = {
                    "name": device.name,
                    "total_memory": f"{device.total_memory / 1024**3:.2f} GB",
                    "compute_capability": f"{device.major}.{device.minor}",
                    "multi_processor_count": device.multi_processor_count
                }
        return device_info
    
    def setup_cuda(self) -> bool:
        """Setup CUDA environment"""
        if not self.cuda_available:
            print("CUDA is not available. Using CPU.")
            return False
            
        # Set CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU by default
        
        # Set memory allocation
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
        
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        
        # Enable deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        
        print(f"CUDA setup complete. Using device: {torch.cuda.get_device_name(0)}")
        return True
    
    def get_device(self) -> torch.device:
        """Get the appropriate device (CPU or GPU)"""
        return torch.device('cuda' if self.cuda_available else 'cpu')
    
    def print_device_info(self):
        """Print information about available devices"""
        print("\nCUDA Setup Information:")
        print(f"CUDA Available: {self.cuda_available}")
        print(f"CUDA Version: {self.cuda_version}")
        print(f"Device Count: {self.device_count}")
        
        if self.device_count > 0:
            print("\nAvailable Devices:")
            for device_id, info in self.device_info.items():
                print(f"\n{device_id}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")

# Create global instance
cuda_setup = CUDASetup() 