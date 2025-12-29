"""
Return-based features for volatility prediction.

Includes log returns, squared returns, and lagged returns.
"""

import numpy as np
import pandas as pd


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(1))


def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Simple returns series
    """
    return prices.pct_change()


def calculate_squared_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate squared log returns (proxy for realized variance).
    
    Args:
        prices: Price series
        
    Returns:
        Squared log returns series
    """
    log_returns = calculate_log_returns(prices)
    return log_returns ** 2


def calculate_lagged_returns(
    prices: pd.Series,
    lags: list = None
) -> pd.DataFrame:
    """
    Calculate lagged returns for multiple lags.
    
    Args:
        prices: Price series
        lags: List of lag periods (default: [1, 2, 3, 5, 10])
        
    Returns:
        DataFrame with lagged returns
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    
    returns = calculate_log_returns(prices)
    lagged = pd.DataFrame(index=prices.index)
    
    for lag in lags:
        lagged[f'return_lag_{lag}'] = returns.shift(lag)
    
    return lagged

