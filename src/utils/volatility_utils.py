"""
Volatility utility functions for predictions and analysis.

This module provides utilities for volatility regime classification,
daily volatility calculation, and path interpolation.
"""

from typing import List
import numpy as np
import pandas as pd


def classify_volatility_regime(volatility_history: pd.Series, window: int = 60) -> pd.Series:
    """
    Classify volatility history into low/normal/high regimes.
    
    Classifies each point in the volatility history based on percentiles
    of a rolling window. Uses 33rd and 67th percentiles as thresholds.
    
    Args:
        volatility_history: Series of historical volatility values
        window: Rolling window size for percentile calculation (default: 60)
    
    Returns:
        Series of regime classifications ('Low', 'Normal', or 'High')
    """
    if len(volatility_history) == 0:
        return pd.Series(dtype=str, index=volatility_history.index)
    
    # Clean data
    clean_vol = volatility_history.dropna()
    if len(clean_vol) == 0:
        return pd.Series(dtype=str, index=volatility_history.index)
    
    # Initialize result series
    regimes = pd.Series(index=volatility_history.index, dtype=str)
    
    # Classify each point based on rolling percentiles
    for i in range(len(volatility_history)):
        if pd.isna(volatility_history.iloc[i]):
            regimes.iloc[i] = 'Normal'
            continue
        
        # Get rolling window ending at current point
        window_start = max(0, i - window + 1)
        window_data = volatility_history.iloc[window_start:i+1].dropna()
        
        if len(window_data) < 3:
            # Not enough data for classification
            regimes.iloc[i] = 'Normal'
            continue
        
        # Calculate percentiles
        p33 = window_data.quantile(0.33)
        p67 = window_data.quantile(0.67)
        
        current_vol = volatility_history.iloc[i]
        
        # Classify regime
        if current_vol <= p33:
            regimes.iloc[i] = 'Low'
        elif current_vol <= p67:
            regimes.iloc[i] = 'Normal'
        else:
            regimes.iloc[i] = 'High'
    
    return regimes


def calculate_daily_realized_volatility(
    squared_returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate daily realized volatility from squared returns.
    
    Computes rolling standard deviation of returns, then annualizes.
    This gives the realized volatility at each point in time.
    
    Args:
        squared_returns: Series of squared log returns
        window: Rolling window size (default: 20)
    
    Returns:
        Series of annualized volatility values (as percentages)
    """
    if len(squared_returns) == 0:
        return pd.Series(dtype=float)
    
    # Calculate rolling mean of squared returns (this is variance)
    # Use a rolling window to get realized variance
    rolling_variance = squared_returns.rolling(
        window=window,
        min_periods=min(5, window // 4)  # Require at least some data
    ).mean()
    
    # Convert variance to volatility (standard deviation)
    # Then annualize: multiply by sqrt(252) and convert to percentage
    volatility = np.sqrt(rolling_variance * 252) * 100
    
    return volatility


def interpolate_volatility_from_historical_patterns(
    historical_vol: pd.Series,
    start_vol: float,
    end_vol: float,
    horizon: int,
    min_patterns: int = 3
) -> np.ndarray:
    """
    Interpolate volatility path between start and end values using historical patterns.
    
    Finds historical transitions in the volatility series that are similar to the
    current transition (from start_vol to end_vol) and uses their patterns to
    create a realistic interpolation path.
    
    Args:
        historical_vol: Series of historical volatility values
        start_vol: Starting volatility value
        end_vol: Target volatility value at end of horizon
        horizon: Number of days in the path
        min_patterns: Minimum number of matching patterns required (default: 3)
    
    Returns:
        NumPy array of volatility values for the path
    """
    if horizon <= 1:
        return np.array([end_vol])
    
    # Clean historical data
    clean_historical = historical_vol.dropna()
    if len(clean_historical) < horizon + 10:
        # Not enough historical data, use simple linear interpolation
        return np.linspace(start_vol, end_vol, horizon)
    
    # Convert to numpy array for easier manipulation
    hist_array = clean_historical.values
    
    # Find historical patterns: look for transitions similar to start_vol -> end_vol
    # We'll look for segments where volatility moves from near start_vol to near end_vol
    patterns = []
    tolerance = max(abs(start_vol - end_vol) * 0.2, 2.0)  # 20% of transition size or 2% minimum
    
    for i in range(len(hist_array) - horizon):
        segment_start = hist_array[i]
        segment_end = hist_array[i + horizon - 1] if i + horizon - 1 < len(hist_array) else hist_array[-1]
        
        # Check if this segment is similar to our target transition
        start_diff = abs(segment_start - start_vol)
        end_diff = abs(segment_end - end_vol)
        
        if start_diff <= tolerance and end_diff <= tolerance:
            # Extract the pattern
            pattern = hist_array[i:i+horizon]
            if len(pattern) == horizon:
                patterns.append(pattern)
    
    # If we found enough patterns, use them
    if len(patterns) >= min_patterns:
        # Average the patterns to create a template
        patterns_array = np.array(patterns)
        template = np.mean(patterns_array, axis=0)
        
        # Normalize template to match our start and end values
        template_start = template[0]
        template_end = template[-1]
        
        if abs(template_end - template_start) > 1e-6:
            # Scale and shift to match our target start and end
            scale = (end_vol - start_vol) / (template_end - template_start)
            offset = start_vol - template_start * scale
            interpolated = template * scale + offset
        else:
            # Template is flat, use linear interpolation
            interpolated = np.linspace(start_vol, end_vol, horizon)
        
        return interpolated
    else:
        # Not enough patterns found, use linear interpolation with some smoothing
        linear_path = np.linspace(start_vol, end_vol, horizon)
        
        # Add some natural variation based on historical volatility characteristics
        if len(clean_historical) > 0:
            hist_std = clean_historical.std()
            # Add small random variation (scaled to historical variability)
            variation = np.random.normal(0, hist_std * 0.1, horizon)
            # Reduce variation as we approach the end
            decay = np.linspace(1.0, 0.3, horizon)
            variation = variation * decay
            linear_path = linear_path + variation
            # Ensure we still end at the target
            linear_path[-1] = end_vol
        
        return linear_path


__all__ = [
    'classify_volatility_regime',
    'calculate_daily_realized_volatility',
    'interpolate_volatility_from_historical_patterns',
]

