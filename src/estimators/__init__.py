"""
Volatility estimator modules.

This package contains implementations of various volatility estimators:
- Close-to-Close: Standard realized volatility
- EWMA: Exponentially weighted moving average
- Parkinson: Range-based estimator using high/low
- Rogers-Satchell: Range-based estimator accounting for drift
- Yang-Zhang: Comprehensive range-based estimator
"""

from src.estimators.close_to_close import CloseToCloseEstimator
from src.estimators.ewma import EWMAEstimator
from src.estimators.parkinson import ParkinsonEstimator
from src.estimators.rogers_satchell import RogersSatchellEstimator
from src.estimators.yang_zhang import YangZhangEstimator

# Estimator registry
ESTIMATORS = {
    'close_to_close': CloseToCloseEstimator,
    'ewma': EWMAEstimator,
    'parkinson': ParkinsonEstimator,
    'rogers_satchell': RogersSatchellEstimator,
    'yang_zhang': YangZhangEstimator,
}


def get_estimator(name: str, window: int = 60, annualization_factor: int = 252, **kwargs):
    """
    Factory function to get estimator instance.

    Args:
        name: Estimator name (must be in ESTIMATORS registry)
        window: Rolling window size
        annualization_factor: Days per year
        **kwargs: Additional estimator-specific parameters (e.g., lambda for EWMA)

    Returns:
        Estimator instance

    Raises:
        ValueError: If estimator name is not found
    """
    if name not in ESTIMATORS:
        available = ', '.join(ESTIMATORS.keys())
        raise ValueError(
            f"Unknown estimator: {name}. Available estimators: {available}"
        )

    estimator_class = ESTIMATORS[name]
    return estimator_class(window=window, annualization_factor=annualization_factor, **kwargs)


def list_estimators():
    """
    List all available estimator names.

    Returns:
        List of estimator names
    """
    return list(ESTIMATORS.keys())

