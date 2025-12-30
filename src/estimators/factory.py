"""
Factory module for creating volatility estimator instances.

This module provides a factory pattern for creating estimator instances by name,
allowing dynamic estimator selection without hardcoding class imports.
"""

from typing import Dict, Type, Optional, List
import warnings

from src.estimators.base import BaseEstimator
from src.estimators.close_to_close import CloseToCloseEstimator
from src.estimators.ewma import EWMAEstimator
from src.estimators.parkinson import ParkinsonEstimator
from src.estimators.rogers_satchell import RogersSatchellEstimator
from src.estimators.yang_zhang import YangZhangEstimator


# Registry of available estimators
_ESTIMATORS: Dict[str, Type[BaseEstimator]] = {
    'close_to_close': CloseToCloseEstimator,
    'ewma': EWMAEstimator,
    'parkinson': ParkinsonEstimator,
    'rogers_satchell': RogersSatchellEstimator,
    'yang_zhang': YangZhangEstimator,
}

# Export the registry as ESTIMATORS for backward compatibility
ESTIMATORS = _ESTIMATORS


def get_estimator(
    name: str,
    window: int = 60,
    annualization_factor: int = 252,
    **kwargs
) -> BaseEstimator:
    """
    Create an estimator instance by name.
    
    Args:
        name: Estimator name (e.g., 'yang_zhang', 'ewma', 'close_to_close')
        window: Rolling window size (trading days)
        annualization_factor: Days per year for annualization (default: 252)
        **kwargs: Additional arguments for specific estimators
                  (e.g., lambda_param for EWMA)
    
    Returns:
        Estimator instance
        
    Raises:
        ValueError: If estimator name is not found
        TypeError: If invalid arguments are provided
    """
    name = name.lower().strip()
    
    if name not in _ESTIMATORS:
        available = ', '.join(_ESTIMATORS.keys())
        raise ValueError(
            f"Unknown estimator '{name}'. Available estimators: {available}"
        )
    
    estimator_class = _ESTIMATORS[name]
    
    # EWMA requires lambda_param
    if name == 'ewma':
        lambda_param = kwargs.get('lambda_param', 0.94)
        return estimator_class(
            window=window,
            annualization_factor=annualization_factor,
            lambda_param=lambda_param
        )
    else:
        # Other estimators only need window and annualization_factor
        # Filter out any extra kwargs that aren't needed
        return estimator_class(
            window=window,
            annualization_factor=annualization_factor
        )


def list_estimators() -> List[str]:
    """
    Get a list of available estimator names.
    
    Returns:
        List of estimator names
    """
    return list(_ESTIMATORS.keys())


def register_estimator(
    name: str,
    estimator_class: Type[BaseEstimator],
    override: bool = False
) -> None:
    """
    Register a custom estimator.
    
    Args:
        name: Estimator name (will be converted to lowercase)
        estimator_class: Estimator class (must inherit from BaseEstimator)
        override: If True, allow overriding existing estimators
        
    Raises:
        TypeError: If estimator_class is not a subclass of BaseEstimator
        ValueError: If name already exists and override=False
    """
    if not issubclass(estimator_class, BaseEstimator):
        raise TypeError(
            f"Estimator class must inherit from BaseEstimator, "
            f"got {estimator_class}"
        )
    
    name = name.lower().strip()
    
    if name in _ESTIMATORS and not override:
        raise ValueError(
            f"Estimator '{name}' already registered. "
            f"Use override=True to replace it."
        )
    
    if name in _ESTIMATORS and override:
        warnings.warn(
            f"Overriding existing estimator '{name}'",
            UserWarning
        )
    
    _ESTIMATORS[name] = estimator_class


__all__ = [
    'ESTIMATORS',
    'get_estimator',
    'list_estimators',
    'register_estimator',
]

