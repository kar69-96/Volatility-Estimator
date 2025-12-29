"""
Deep Learning Models for Volatility Prediction.

This package contains PyTorch models for:
- Forward realized volatility prediction (iTransformer)
- Fed rate event prediction (LSTM/Transformer)
- Neural GARCH conditional variance estimation
"""

from src.models.base_model import (
    get_device,
    to_device,
    set_seed,
    DeviceInfo,
)

__all__ = [
    'get_device',
    'to_device',
    'set_seed',
    'DeviceInfo',
]
