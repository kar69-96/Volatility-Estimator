"""
Volatility Prediction Module.

Consolidated module for volatility prediction including:
- Models (Chronos, base model utilities)
- Prediction functions
- Training infrastructure
"""

from src.volatility.models import (
    get_device,
    to_device,
    set_seed,
    DeviceInfo,
    ChronosVolatility,
)

from src.volatility.prediction import (
    is_deep_learning_available,
    predict_volatility_chronos,
    find_similar_events,
    predict_volatility_path,
    build_pattern_database,
    backtest_predictions,
)

from src.volatility.training import (
    VolatilityTrainer,
    VolatilityDataModule,
    TimeSeriesDataset,
    calculate_regression_metrics,
    calculate_classification_metrics,
)

__all__ = [
    # Models
    'get_device',
    'to_device',
    'set_seed',
    'DeviceInfo',
    'ChronosVolatility',
    # Prediction
    'is_deep_learning_available',
    'predict_volatility_chronos',
    'find_similar_events',
    'predict_volatility_path',
    'build_pattern_database',
    'backtest_predictions',
    # Training
    'VolatilityTrainer',
    'VolatilityDataModule',
    'TimeSeriesDataset',
    'calculate_regression_metrics',
    'calculate_classification_metrics',
]

