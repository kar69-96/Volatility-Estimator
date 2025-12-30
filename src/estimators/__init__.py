"""
Volatility estimator modules.

This package contains implementations of various volatility estimators:
- Close-to-Close: Standard realized volatility
- EWMA: Exponentially weighted moving average
- Parkinson: Range-based estimator using high/low
- Rogers-Satchell: Range-based estimator accounting for drift
- Yang-Zhang: Comprehensive range-based estimator
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

__all__ = [
    'VolatilityEstimator',
    'CloseToCloseEstimator',
    'EWMAEstimator',
    'ParkinsonEstimator',
    'RogersSatchellEstimator',
    'YangZhangEstimator',
    'get_estimator',
    'list_estimators',
    'register_estimator',
    'ESTIMATORS',
]
