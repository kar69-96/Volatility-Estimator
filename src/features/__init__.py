"""
Feature engineering module for deep learning models.

Includes:
- Return-based features (log returns, squared returns, lags)
- Volatility features (realized vol, Parkinson, GK, ATR)
- Technical indicators (RSI, MACD, Bollinger, Stochastic, etc.)
- Regime indicators (trend, volatility regime, momentum)
- Preprocessing utilities (scaling, sequence creation)
"""

from src.features.feature_extractor import (
    FeatureExtractor,
    extract_all_features,
    create_sequences,
    create_multi_horizon_sequences,
)
from src.features.return_features import (
    calculate_log_returns,
    calculate_simple_returns,
    calculate_squared_returns,
    calculate_lagged_returns,
)
from src.features.volatility_features import (
    calculate_realized_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
    calculate_atr,
    calculate_atr_percent,
)
from src.features.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_bollinger_width,
    calculate_sma,
    calculate_ema,
    calculate_momentum,
    calculate_rate_of_change,
    calculate_stochastic_oscillator,
    calculate_williams_r,
    calculate_adx,
)
from src.features.regime_indicators import (
    calculate_trend_indicator,
    classify_volatility_regime,
    calculate_momentum_regime,
    calculate_volatility_percentile,
)
from src.features.preprocessors import (
    create_scaler,
    scale_features,
    handle_missing_values,
)

__all__ = [
    # Main extractor
    'FeatureExtractor',
    'extract_all_features',
    'create_sequences',
    'create_multi_horizon_sequences',
    # Return features
    'calculate_log_returns',
    'calculate_simple_returns',
    'calculate_squared_returns',
    'calculate_lagged_returns',
    # Volatility features
    'calculate_realized_volatility',
    'calculate_parkinson_volatility',
    'calculate_garman_klass_volatility',
    'calculate_atr',
    'calculate_atr_percent',
    # Technical indicators
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_bollinger_width',
    'calculate_sma',
    'calculate_ema',
    'calculate_momentum',
    'calculate_rate_of_change',
    'calculate_stochastic_oscillator',
    'calculate_williams_r',
    'calculate_adx',
    # Regime indicators
    'calculate_trend_indicator',
    'classify_volatility_regime',
    'calculate_momentum_regime',
    'calculate_volatility_percentile',
    # Preprocessors
    'create_scaler',
    'scale_features',
    'handle_missing_values',
]
