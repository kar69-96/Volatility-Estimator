"""
Factory functions for volatility estimators.

Provides centralized creation and registration of estimator instances.
"""

from typing import Dict, Type

from src.estimators.base import BaseEstimator as VolatilityEstimator
from src.estimators.close_to_close import CloseToCloseEstimator
from src.estimators.ewma import EWMAEstimator
from src.estimators.parkinson import ParkinsonEstimator
from src.estimators.rogers_satchell import RogersSatchellEstimator
from src.estimators.yang_zhang import YangZhangEstimator

# Estimator registry (traditional estimators only)
ESTIMATORS: Dict[str, Type[VolatilityEstimator]] = {
    'close_to_close': CloseToCloseEstimator,
    'ewma': EWMAEstimator,
    'parkinson': ParkinsonEstimator,
    'rogers_satchell': RogersSatchellEstimator,
    'yang_zhang': YangZhangEstimator,
}


def get_estimator(name: str, **kwargs) -> VolatilityEstimator:
    """
    Factory function to get an estimator instance by name.
    
    Args:
        name: Estimator name (e.g., 'close_to_close', 'yang_zhang')
        **kwargs: Additional parameters to pass to the estimator constructor
    
    Returns:
        Configured estimator instance
    
    Raises:
        ValueError: If estimator name is not recognized
    
    Example:
        >>> estimator = get_estimator('yang_zhang', window=30)
        >>> volatility = estimator.calculate(df)
    """
    if name not in ESTIMATORS:
        available = ', '.join(ESTIMATORS.keys())
        raise ValueError(
            f"Unknown estimator '{name}'. "
            f"Available estimators: {available}"
        )
    
    estimator_class = ESTIMATORS[name]
    return estimator_class(**kwargs)


def list_estimators() -> list:
    """
    Get list of available estimator names.
    
    Returns:
        List of estimator names that can be used with get_estimator()
    """
    return list(ESTIMATORS.keys())


def register_estimator(name: str, estimator_class: Type[VolatilityEstimator]) -> None:
    """
    Register a new estimator in the factory.
    
    Args:
        name: Unique name for the estimator
        estimator_class: Estimator class (must inherit from VolatilityEstimator)
    
    Raises:
        ValueError: If name already registered or class is invalid
    """
    if name in ESTIMATORS:
        raise ValueError(f"Estimator '{name}' is already registered")
    
    if not issubclass(estimator_class, VolatilityEstimator):
        raise ValueError(
            f"Estimator class must inherit from VolatilityEstimator"
        )
    
    ESTIMATORS[name] = estimator_class

