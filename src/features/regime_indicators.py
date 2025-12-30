"""
Market regime indicators for feature engineering.

Includes trend detection, volatility regime classification, and momentum indicators.
"""

import numpy as np
import pandas as pd


def calculate_trend_indicator(
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 60
) -> pd.Series:
    """
    Calculate trend indicator using moving average crossover.
    
    Args:
        prices: Price series
        short_window: Short MA window
        long_window: Long MA window
        
    Returns:
        Trend indicator: 1 (uptrend), 0 (neutral), -1 (downtrend)
    """
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    
    trend = pd.Series(0, index=prices.index)
    trend[short_ma > long_ma] = 1
    trend[short_ma < long_ma] = -1
    
    return trend


def classify_volatility_regime(
    volatility: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Classify volatility into low/medium/high regimes.
    
    Args:
        volatility: Volatility series
        window: Rolling window for percentile calculation
        
    Returns:
        Regime labels: 'low', 'medium', 'high'
    """
    rolling_33 = volatility.rolling(window=window).quantile(0.33)
    rolling_67 = volatility.rolling(window=window).quantile(0.67)
    
    regime = pd.Series('medium', index=volatility.index)
    regime[volatility <= rolling_33] = 'low'
    regime[volatility >= rolling_67] = 'high'
    
    return regime


def calculate_momentum_regime(
    prices: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate momentum regime indicator.
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        Momentum regime: 1 (strong), 0 (neutral), -1 (weak)
    """
    momentum = prices.pct_change(period)
    
    regime = pd.Series(0, index=prices.index)
    regime[momentum > 0.1] = 1  # Strong positive momentum
    regime[momentum < -0.1] = -1  # Strong negative momentum
    
    return regime


def calculate_volatility_percentile(
    volatility: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Calculate rolling percentile rank of volatility.
    
    Args:
        volatility: Volatility series
        window: Rolling window
        
    Returns:
        Percentile rank (0-100)
    """
    def percentile_rank(x):
        if len(x) < 2:
            return np.nan
        return (x.iloc[-1] <= x).sum() / len(x) * 100
    
    percentile = volatility.rolling(window=window).apply(percentile_rank, raw=False)
    return percentile

