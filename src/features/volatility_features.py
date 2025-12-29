"""
Volatility-based features for prediction models.

Includes realized volatility, Parkinson, Garman-Klass, and ATR.
"""

import numpy as np
import pandas as pd


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate realized volatility from returns.
    
    Args:
        returns: Return series (log or simple)
        window: Rolling window size
        annualize: Whether to annualize the volatility
        annualization_factor: Days per year for annualization
        
    Returns:
        Realized volatility series (optionally annualized)
    """
    variance = returns.rolling(window=window).var()
    volatility = np.sqrt(variance)
    
    if annualize:
        volatility = volatility * np.sqrt(annualization_factor)
    
    return volatility


def calculate_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    annualize: bool = True,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate Parkinson volatility estimator.
    
    Uses high-low range to estimate volatility more efficiently
    than close-to-close.
    
    Args:
        high: High prices
        low: Low prices
        window: Rolling window size
        annualize: Whether to annualize the volatility
        annualization_factor: Days per year
        
    Returns:
        Parkinson volatility series (optionally annualized)
    """
    hl_ratio = np.log(high / low)
    parkinson_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
    
    rolling_var = parkinson_var.rolling(window=window).mean()
    volatility = np.sqrt(rolling_var)
    
    if annualize:
        volatility = volatility * np.sqrt(annualization_factor)
    
    return volatility


def calculate_garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator.
    
    Uses OHLC data for more efficient volatility estimation.
    
    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window size
        annualize: Whether to annualize the volatility
        annualization_factor: Days per year
        
    Returns:
        Garman-Klass volatility series (optionally annualized)
    """
    hl_ratio = np.log(high / low)
    co_ratio = np.log(close / open_)
    
    gk_var = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)
    
    rolling_var = gk_var.rolling(window=window).mean()
    volatility = np.sqrt(rolling_var)
    
    if annualize:
        volatility = volatility * np.sqrt(annualization_factor)
    
    return volatility


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_atr_percent(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate ATR as percentage of close price.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR percentage series
    """
    atr = calculate_atr(high, low, close, period)
    atr_pct = (atr / close) * 100
    
    return atr_pct

