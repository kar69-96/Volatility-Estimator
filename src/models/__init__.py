"""
Deep Learning Models for Volatility Prediction.

This package contains PyTorch models for:
- Forward realized volatility prediction (Chronos)
- Fed rate event prediction (LSTM/Transformer)
"""

from src.models.base_model import (
    get_device,
    to_device,
    set_seed,
    DeviceInfo,
)

try:
    from src.models.chronos import ChronosVolatility
    _CHRONOS_AVAILABLE = True
except ImportError:
    _CHRONOS_AVAILABLE = False
    ChronosVolatility = None

__all__ = [
    'get_device',
    'to_device',
    'set_seed',
    'DeviceInfo',
    'ChronosVolatility',
]
