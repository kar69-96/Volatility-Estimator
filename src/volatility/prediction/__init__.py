"""
Prediction module for volatility forecasting.

Includes:
- Deep learning predictions (Chronos)
- Pattern-based predictions
- Backtesting utilities
"""

from src.volatility.prediction.predictions import (
    is_deep_learning_available,
    find_similar_events,
    predict_volatility_path,
    build_pattern_database,
    backtest_predictions,
    predict_volatility_chronos,
)

__all__ = [
    'is_deep_learning_available',
    'find_similar_events',
    'predict_volatility_path',
    'build_pattern_database',
    'backtest_predictions',
    'predict_volatility_chronos',
]
