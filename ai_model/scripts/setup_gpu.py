#!/usr/bin/env python3
import sys
import subprocess
import platform
from pathlib import Path

def check_cuda_installation():
    """Check if CUDA is properly installed"""
    try:
        # Check nvcc version
        nvcc_version = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        print("✓ CUDA Toolkit is installed")
        print(f"  Version: {nvcc_version.split('release ')[1].split(',')[0]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ CUDA Toolkit is not installed or not in PATH")
        return False

def check_cudnn_installation():
    """Check if cuDNN is properly installed"""
    cudnn_paths = [
        '/usr/local/cuda/include/cudnn.h',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include/cudnn.h'
    ]
    
    for path in cudnn_paths:
        if Path(path).exists():
            print("✓ cuDNN is installed")
            return True
    
    print("✗ cuDNN is not installed")
    return False

def check_gpu_drivers():
    """Check if GPU drivers are properly installed"""
    system = platform.system()
    if system == 'Windows':
        try:
            subprocess.check_output(['nvidia-smi'])
            print("✓ NVIDIA drivers are installed")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ NVIDIA drivers are not installed or not in PATH")
            return False
    elif system == 'Linux':
        try:
            subprocess.check_output(['nvidia-smi'])
            print("✓ NVIDIA drivers are installed")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ NVIDIA drivers are not installed or not in PATH")
            return False
    else:
        print("✗ GPU driver check not supported on this platform")
        return False

def check_pytorch_cuda():
    """Check if PyTorch can use CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ PyTorch can use CUDA")
            print(f"  Available devices: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("✗ PyTorch cannot use CUDA")
            return False
    except ImportError:
        print("✗ PyTorch is not installed")
        return False

def setup_environment():
    """Setup the environment for GPU usage"""
    print("\nSetting up GPU environment...")
    
    # Check prerequisites
    cuda_installed = check_cuda_installation()
    cudnn_installed = check_cudnn_installation()
    drivers_installed = check_gpu_drivers()
    pytorch_ready = check_pytorch_cuda()
    
    # Print summary
    print("\nSetup Summary:")
    print(f"CUDA Toolkit: {'✓' if cuda_installed else '✗'}")
    print(f"cuDNN: {'✓' if cudnn_installed else '✗'}")
    print(f"GPU Drivers: {'✓' if drivers_installed else '✗'}")
    print(f"PyTorch CUDA: {'✓' if pytorch_ready else '✗'}")
    
    # Check if everything is ready
    if all([cuda_installed, cudnn_installed, drivers_installed, pytorch_ready]):
        print("\n✓ GPU environment is ready for use")
        return True
    else:
        print("\n✗ GPU environment is not ready. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1) 