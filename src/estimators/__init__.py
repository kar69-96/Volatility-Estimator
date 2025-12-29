"""
Volatility estimator modules.

This package contains implementations of various volatility estimators:
- Close-to-Close: Standard realized volatility
- EWMA: Exponentially weighted moving average
- Parkinson: Range-based estimator using high/low
- Rogers-Satchell: Range-based estimator accounting for drift
- Yang-Zhang: Comprehensive range-based estimator
- Neural GARCH: Neural network-based conditional variance estimation
"""

from src.estimators.base import BaseEstimator as VolatilityEstimator
from src.estimators.close_to_close import CloseToCloseEstimator
from src.estimators.ewma import EWMAEstimator
from src.estimators.parkinson import ParkinsonEstimator
from src.estimators.rogers_satchell import RogersSatchellEstimator
from src.estimators.yang_zhang import YangZhangEstimator
from src.estimators.factory import (
    get_estimator,
    list_estimators,
    register_estimator,
    ESTIMATORS,
)

# Neural GARCH import (optional - requires PyTorch)
try:
    from src.estimators.neural_garch import NeuralGARCHEstimator
    _NEURAL_GARCH_AVAILABLE = True
except ImportError:
    NeuralGARCHEstimator = None
    _NEURAL_GARCH_AVAILABLE = False


def is_neural_garch_available() -> bool:
    """Check if Neural GARCH estimator is available."""
    return _NEURAL_GARCH_AVAILABLE


__all__ = [
    'VolatilityEstimator',
    'CloseToCloseEstimator',
    'EWMAEstimator',
    'ParkinsonEstimator',
    'RogersSatchellEstimator',
    'YangZhangEstimator',
    'NeuralGARCHEstimator',
    'get_estimator',
    'list_estimators',
    'register_estimator',
    'is_neural_garch_available',
    'ESTIMATORS',
]
