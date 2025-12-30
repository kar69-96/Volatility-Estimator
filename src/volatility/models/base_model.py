"""
GPU Utilities Module.

Provides device detection, CUDA support, and CPU fallback for PyTorch models.
Includes utilities for mixed precision training and batch processing.
"""

import os
import random
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

# Lazy import torch to avoid issues if not installed
_torch = None
_torch_available = None


def _get_torch():
    """Lazy load PyTorch."""
    global _torch, _torch_available
    if _torch_available is None:
        try:
            import torch
            _torch = torch
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch if _torch_available else None


@dataclass
class DeviceInfo:
    """Information about the compute device."""
    device_type: str  # 'cuda', 'mps', 'cpu'
    device_name: str
    device_index: Optional[int]
    cuda_available: bool
    mps_available: bool
    cuda_version: Optional[str]
    gpu_memory_total: Optional[float]  # GB
    gpu_memory_free: Optional[float]   # GB
    
    def __str__(self) -> str:
        if self.device_type == 'cuda':
            mem_str = ""
            if self.gpu_memory_total:
                mem_str = f" ({self.gpu_memory_free:.1f}/{self.gpu_memory_total:.1f} GB free)"
            return f"CUDA: {self.device_name}{mem_str}"
        elif self.device_type == 'mps':
            return f"MPS (Apple Silicon): {self.device_name}"
        else:
            return "CPU"


def get_device_info() -> DeviceInfo:
    """
    Get detailed information about available compute devices.
    
    Returns:
        DeviceInfo dataclass with device details
    """
    torch = _get_torch()
    
    if torch is None:
        return DeviceInfo(
            device_type='cpu',
            device_name='CPU (PyTorch not installed)',
            device_index=None,
            cuda_available=False,
            mps_available=False,
            cuda_version=None,
            gpu_memory_total=None,
            gpu_memory_free=None
        )
    
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if cuda_available:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        cuda_version = torch.version.cuda
        
        # Get memory info
        props = torch.cuda.get_device_properties(device_index)
        total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
        
        # Get free memory
        torch.cuda.synchronize()
        free_memory = (props.total_memory - torch.cuda.memory_allocated(device_index)) / (1024 ** 3)
        
        return DeviceInfo(
            device_type='cuda',
            device_name=device_name,
            device_index=device_index,
            cuda_available=True,
            mps_available=mps_available,
            cuda_version=cuda_version,
            gpu_memory_total=total_memory,
            gpu_memory_free=free_memory
        )
    elif mps_available:
        return DeviceInfo(
            device_type='mps',
            device_name='Apple Silicon GPU',
            device_index=None,
            cuda_available=False,
            mps_available=True,
            cuda_version=None,
            gpu_memory_total=None,
            gpu_memory_free=None
        )
    else:
        return DeviceInfo(
            device_type='cpu',
            device_name='CPU',
            device_index=None,
            cuda_available=False,
            mps_available=False,
            cuda_version=None,
            gpu_memory_total=None,
            gpu_memory_free=None
        )


def get_device(device_preference: str = 'auto') -> Any:
    """
    Get the appropriate PyTorch device based on availability.
    
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU
    
    This function prioritizes GPU acceleration when available, automatically
    falling back to CPU if no GPU is found.
    
    Args:
        device_preference: 'auto', 'cuda', 'mps', or 'cpu'
            - 'auto': Automatically select best available (GPU first, then CPU)
            - 'cuda': Force CUDA (NVIDIA GPU), fallback to CPU if unavailable
            - 'mps': Force MPS (Apple Silicon GPU), fallback to CPU if unavailable
            - 'cpu': Force CPU
            
    Returns:
        torch.device object
    """
    torch = _get_torch()
    
    if torch is None:
        raise ImportError("PyTorch is not installed. Please install it with: pip install torch")
    
    if device_preference == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device('cpu')
    
    elif device_preference == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return torch.device('cpu')
    
    elif device_preference == 'cpu':
        return torch.device('cpu')
    
    else:  # auto - prioritize GPU, fallback to CPU
        # First priority: CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            return torch.device('cuda')
        # Second priority: MPS (Apple Silicon GPU)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        # Fallback: CPU
        else:
            return torch.device('cpu')


def to_device(data: Any, device: Any = None) -> Any:
    """
    Move data to the specified device.
    
    Handles tensors, models, and nested structures (lists, tuples, dicts).
    
    Args:
        data: Data to move (tensor, model, list, tuple, dict)
        device: Target device. If None, uses auto-detected device.
        
    Returns:
        Data on the target device
    """
    torch = _get_torch()
    
    if torch is None:
        return data
    
    if device is None:
        device = get_device()
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, torch.nn.Module):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    else:
        return data


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch = _get_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def count_parameters(model: Any) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    torch = _get_torch()
    if torch is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> dict:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory statistics in GB
    """
    torch = _get_torch()
    
    if torch is None or not torch.cuda.is_available():
        return {
            'allocated': 0,
            'cached': 0,
            'total': 0,
            'free': 0
        }
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    cached = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    return {
        'allocated': round(allocated, 2),
        'cached': round(cached, 2),
        'total': round(total, 2),
        'free': round(total - allocated, 2)
    }


def clear_memory() -> None:
    """Clear GPU memory cache."""
    torch = _get_torch()
    
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class GradScaler:
    """
    Wrapper for mixed precision training gradient scaling.
    
    Automatically handles AMP (Automatic Mixed Precision) if available.
    Falls back to identity operations if not available.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize gradient scaler.
        
        Args:
            enabled: Whether to enable mixed precision
        """
        self.enabled = enabled
        torch = _get_torch()
        
        if torch is not None and enabled and torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None
    
    def scale(self, loss: Any) -> Any:
        """Scale the loss for backward pass."""
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss
    
    def step(self, optimizer: Any) -> None:
        """Perform optimizer step with unscaling."""
        if self._scaler is not None:
            self._scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update(self) -> None:
        """Update the scale factor."""
        if self._scaler is not None:
            self._scaler.update()


def get_autocast_context(device_type: str = 'cuda', enabled: bool = True):
    """
    Get autocast context manager for mixed precision.
    
    Args:
        device_type: 'cuda', 'cpu', or 'mps'
        enabled: Whether to enable autocasting
        
    Returns:
        Context manager for autocasting
    """
    torch = _get_torch()
    
    if torch is None:
        from contextlib import nullcontext
        return nullcontext()
    
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    
    if device_type == 'cuda' and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    elif device_type == 'cpu':
        return torch.cpu.amp.autocast()
    else:
        from contextlib import nullcontext
        return nullcontext()


# Convenience functions for model checkpointing
def save_checkpoint(
    model: Any,
    optimizer: Any,
    epoch: int,
    loss: float,
    path: str,
    additional_info: Optional[dict] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        additional_info: Additional metadata to save
    """
    torch = _get_torch()
    if torch is None:
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: Any, optimizer: Any = None) -> dict:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary with checkpoint metadata
    """
    torch = _get_torch()
    if torch is None:
        return {}
    
    device = get_device()
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }

