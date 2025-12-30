"""
Prediction module for volatility forecasting.

Includes:
- Deep learning predictions (iTransformer, Neural GARCH)
- Pattern-based predictions
- Fed rate scenario analysis
- Backtesting utilities
"""

from src.prediction.predictions import (
    is_deep_learning_available,
    find_similar_events,
    predict_volatility_path,
    predict_volatility_dl,
    predict_neural_garch,
    analyze_fed_rate_scenario,
    build_pattern_database,
    backtest_predictions,
    predict_volatility_chronos,
)

__all__ = [
    'is_deep_learning_available',
    'find_similar_events',
    'predict_volatility_path',
    'predict_volatility_dl',
    'predict_neural_garch',
    'analyze_fed_rate_scenario',
    'build_pattern_database',
    'backtest_predictions',
    'predict_volatility_chronos',
]

