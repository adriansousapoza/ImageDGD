"""
Device selection and setup utilities for PyTorch.
"""

import torch
import numpy as np
from typing import Optional


def setup_device(verbose: bool = True) -> torch.device:
    """
    Setup and select the best available CUDA device or fallback to CPU.
    If multiple CUDA devices are available, selects the one with most available memory.
    
    Parameters:
    ----------
    verbose: Whether to print device information
    
    Returns:
    -------
    torch.device: The selected device
    """
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        if verbose:
            print(f"Using CPU")
        return device
    
    # Clear cache before scanning
    torch.cuda.empty_cache()
    
    num_devices = torch.cuda.device_count()
    
    if verbose:
        print(f"Number of CUDA devices: {num_devices}")
    
    if num_devices > 1:
        max_memory = 0
        best_device = 0
        
        if verbose:
            print("\nScanning devices for available memory:")
        
        for i in range(num_devices):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            reserved_memory = torch.cuda.memory_reserved(i)
            available_memory = total_memory - reserved_memory
            
            if verbose:
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"   - Total Memory: {total_memory/1024**2:.2f} MB")
                print(f"   - Allocated: {allocated_memory/1024**2:.2f} MB")
                print(f"   - Reserved: {reserved_memory/1024**2:.2f} MB")
                print(f"   - Available: {available_memory/1024**2:.2f} MB")
            
            if available_memory > max_memory:
                max_memory = available_memory
                best_device = i
        
        device = torch.device(f'cuda:{best_device}')
        torch.cuda.set_device(device)
        
        if verbose:
            print(f"\nSelected device {best_device} with {max_memory/1024**2:.2f} MB available memory")
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    
    if verbose:
        props = torch.cuda.get_device_properties(device)
        print(f"Using CUDA: {torch.cuda.get_device_name(device)} ({props.total_memory / 1e9:.1f} GB total memory)")
    
    return device


def set_random_seed(seed: int = 42, device: Optional[torch.device] = None):
    """
    Set random seeds for reproducibility across numpy, torch CPU and CUDA.
    
    Parameters:
    ----------
    seed: Random seed value
    device: Optional device to check if CUDA seed setting is needed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set CUDA seeds if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_cuml_acceleration(verbose: bool = True) -> bool:
    """
    Attempt to enable cuML GPU acceleration for sklearn-compatible operations.
    
    Parameters:
    ----------
    verbose: Whether to print status messages
    
    Returns:
    -------
    bool: True if cuML acceleration was enabled, False otherwise
    """
    try:
        from cuml.accel import install
        install()
        if verbose:
            print("cuML GPU acceleration enabled")
    except ImportError:
        if verbose:
            print("cuML not installed, using CPU for sklearn operations")
