"""
Return calculation module for volatility estimation.

This module provides functions to calculate log returns and range-based
metrics needed for volatility estimators.
"""

import numpy as np
import pandas as pd


def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Series of prices (typically closing prices)
        method: Return calculation method ('log' or 'simple')

    Returns:
        Series of returns (first value will be NaN)

    Formula:
        Log returns: r_t = ln(P_t / P_{t-1})
        Simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    """
    if method == 'log':
        # Log returns: ln(P_t / P_{t-1})
        # Handle zero prices by returning NaN
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        # Simple returns: (P_t - P_{t-1}) / P_{t-1}
        returns = (prices - prices.shift(1)) / prices.shift(1)
    else:
        raise ValueError(f"Unknown return method: {method}. Use 'log' or 'simple'")

    # First row will be NaN (no previous price)
    return returns


def calculate_ranges(high: pd.Series, low: pd.Series, 
                    open: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Calculate range-based metrics for volatility estimators.

    Returns:
        DataFrame with columns:
        - high_low_ratio: ln(H/L) - used in Parkinson estimator
        - high_close_ratio: ln(H/C) - used in Rogers-Satchell
        - high_open_ratio: ln(H/O) - used in Rogers-Satchell
        - low_close_ratio: ln(L/C) - used in Rogers-Satchell
        - low_open_ratio: ln(L/O) - used in Rogers-Satchell
        - open_close_ratio: ln(O/C) - used in Yang-Zhang

    Edge cases:
        - Zero values in logs → returns NaN
        - Missing data → returns NaN
    """
    ranges = pd.DataFrame(index=high.index)

    # High/Low ratio (Parkinson)
    ranges['high_low_ratio'] = np.log(high / low)

    # High/Close ratio (Rogers-Satchell)
    ranges['high_close_ratio'] = np.log(high / close)

    # High/Open ratio (Rogers-Satchell)
    ranges['high_open_ratio'] = np.log(high / open)

    # Low/Close ratio (Rogers-Satchell)
    ranges['low_close_ratio'] = np.log(low / close)

    # Low/Open ratio (Rogers-Satchell)
    ranges['low_open_ratio'] = np.log(low / open)

    # Open/Close ratio (Yang-Zhang overnight)
    ranges['open_close_ratio'] = np.log(open / close)

    # Handle edge cases: zero or negative values result in NaN
    # This is expected behavior - estimators should handle NaN values

    return ranges


def calculate_overnight_returns(open_prices: pd.Series, 
                                close_prices: pd.Series) -> pd.Series:
    """
    Calculate overnight returns (close to open).

    Used in Yang-Zhang estimator for overnight volatility component.

    Args:
        open_prices: Series of opening prices
        close_prices: Series of closing prices (previous day)

    Returns:
        Series of overnight returns
    """
    # Overnight return = ln(O_t / C_{t-1})
    # Shift close prices forward to align with next day's open
    prev_close = close_prices.shift(1)
    overnight_returns = np.log(open_prices / prev_close)

    return overnight_returns


def calculate_open_returns(open_prices: pd.Series) -> pd.Series:
    """
    Calculate open-to-open returns.

    Used in Yang-Zhang estimator for open volatility component.

    Args:
        open_prices: Series of opening prices

    Returns:
        Series of open returns
    """
    # Open return = ln(O_t / O_{t-1})
    open_returns = np.log(open_prices / open_prices.shift(1))

    return open_returns


def forward_fill_returns(returns: pd.Series, max_gap: int = 2) -> pd.Series:
    """
    Forward fill missing returns for small gaps.

    Args:
        returns: Series of returns with potential NaN values
        max_gap: Maximum number of consecutive NaN values to fill

    Returns:
        Series with small gaps filled
    """
    filled = returns.copy()

    # Find consecutive NaN sequences
    is_na = filled.isna()
    groups = (is_na != is_na.shift()).cumsum()

    for group_id in groups.unique():
        group_mask = groups == group_id
        if is_na[group_mask].all():
            gap_size = group_mask.sum()
            if gap_size <= max_gap:
                # Forward fill using ffill() method
                filled[group_mask] = filled[group_mask].ffill()

    return filled

