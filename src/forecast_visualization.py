"""
Forecast visualization helper functions.

This module provides utilities for calculating forecast metrics,
regime classification, percentiles, and other visualization components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def calculate_historical_volatility(
    df: pd.DataFrame,
    window: int = 20,
    method: str = 'yang_zhang'
) -> pd.Series:
    """
    Calculate historical realized volatility.
    
    Args:
        df: DataFrame with OHLC data
        window: Rolling window size
        method: Method to use ('yang_zhang', 'close_to_close', etc.)
    
    Returns:
        Series of historical volatility values
    """
    from src.returns import calculate_returns
    from src.estimators import get_estimator
    
    config = {'volatility': {'annualization_factor': 252}}
    annualization = 252
    
    if method == 'yang_zhang':
        estimator = get_estimator('yang_zhang', window, annualization)
    else:
        estimator = get_estimator('close_to_close', window, annualization)
    
    volatility = estimator.compute(df, annualize=True)
    return volatility


def classify_volatility_regime(
    volatility_value: float,
    historical_volatility: pd.Series,
    window: int = 60
) -> Tuple[str, float]:
    """
    Classify volatility into low/medium/high regime.
    
    Args:
        volatility_value: Current or forecasted volatility value
        historical_volatility: Historical volatility series
        window: Window for percentile calculation
    
    Returns:
        Tuple of (regime_label, percentile)
    """
    if len(historical_volatility) < window:
        return ("Normal", 50.0)
    
    # Use last window for percentile calculation
    recent_vol = historical_volatility.iloc[-window:].dropna()
    if len(recent_vol) == 0:
        return ("Normal", 50.0)
    
    # Calculate percentiles
    p33 = recent_vol.quantile(0.33)
    p67 = recent_vol.quantile(0.67)
    
    # Calculate percentile rank
    percentile = (recent_vol < volatility_value).sum() / len(recent_vol) * 100
    
    # Classify regime
    if volatility_value <= p33:
        regime = "Low"
    elif volatility_value <= p67:
        regime = "Normal"
    else:
        regime = "High"
    
    return (regime, percentile)


def calculate_historical_averages(
    volatility: pd.Series,
    windows: List[int] = [7, 30, 60, 90]
) -> Dict[int, float]:
    """
    Calculate historical volatility averages over different windows.
    
    Args:
        volatility: Volatility series
        windows: List of window sizes in days
    
    Returns:
        Dictionary mapping window size to average volatility
    """
    averages = {}
    for window in windows:
        if len(volatility) >= window:
            avg = volatility.iloc[-window:].mean()
            averages[window] = float(avg)
        else:
            avg = volatility.mean() if len(volatility) > 0 else 0.0
            averages[window] = float(avg)
    return averages


def calculate_percentile_ranking(
    volatility_value: float,
    historical_volatility: pd.Series,
    window: int = 252
) -> Dict[str, float]:
    """
    Calculate percentile ranking of volatility value.
    
    Args:
        volatility_value: Volatility value to rank
        historical_volatility: Historical volatility series
        window: Window size for historical context
    
    Returns:
        Dictionary with percentile and context information
    """
    if len(historical_volatility) < 10:
        return {
            'percentile': 50.0,
            'higher_than_pct': 50.0,
            'context': 'Insufficient historical data'
        }
    
    # Use last window days
    recent_vol = historical_volatility.iloc[-min(window, len(historical_volatility)):].dropna()
    if len(recent_vol) == 0:
        return {
            'percentile': 50.0,
            'higher_than_pct': 50.0,
            'context': 'No valid historical data'
        }
    
    # Calculate percentile
    percentile = (recent_vol < volatility_value).sum() / len(recent_vol) * 100
    higher_than_pct = 100 - percentile
    
    return {
        'percentile': float(percentile),
        'higher_than_pct': float(higher_than_pct),
        'context': f'Higher than {higher_than_pct:.1f}% of historical periods'
    }


def estimate_confidence_intervals(
    forecast_value: float,
    historical_volatility: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Estimate confidence intervals for forecast.
    
    Uses historical volatility distribution to estimate uncertainty.
    
    Args:
        forecast_value: Forecasted volatility value
        historical_volatility: Historical volatility series
        confidence_level: Confidence level (0.95 for 95%)
    
    Returns:
        Dictionary with lower and upper bounds
    """
    if len(historical_volatility) < 30:
        # Use simple percentage-based bounds if insufficient data
        margin = forecast_value * 0.15  # 15% margin
        return {
            'lower': float(max(0, forecast_value - margin)),
            'upper': float(forecast_value + margin)
        }
    
    # Calculate historical forecast errors (simplified)
    recent_vol = historical_volatility.iloc[-60:].dropna()
    if len(recent_vol) < 10:
        margin = forecast_value * 0.15
        return {
            'lower': float(max(0, forecast_value - margin)),
            'upper': float(forecast_value + margin)
        }
    
    # Use historical standard deviation as uncertainty proxy
    vol_std = recent_vol.std()
    z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
    
    margin = z_score * vol_std * 0.5  # Scale down for forecast uncertainty
    
    return {
        'lower': float(max(0, forecast_value - margin)),
        'upper': float(forecast_value + margin)
    }


def get_alert_level(
    forecast_value: float,
    historical_volatility: pd.Series,
    window: int = 252
) -> Tuple[str, str]:
    """
    Determine alert level based on forecast percentile.
    
    Args:
        forecast_value: Forecasted volatility value
        historical_volatility: Historical volatility series
        window: Window size for percentile calculation
    
    Returns:
        Tuple of (level, color) where level is 'normal', 'elevated', 'extreme'
    """
    if len(historical_volatility) < 10:
        return ("normal", "black")
    
    recent_vol = historical_volatility.iloc[-min(window, len(historical_volatility)):].dropna()
    if len(recent_vol) == 0:
        return ("normal", "black")
    
    percentile = (recent_vol < forecast_value).sum() / len(recent_vol) * 100
    
    if percentile >= 90:
        return ("extreme", "#000000")  # Black for extreme
    elif percentile >= 75:
        return ("elevated", "#666666")  # Gray for elevated
    else:
        return ("normal", "#000000")  # Black for normal


def prepare_forecast_dataframe(
    historical_volatility: pd.Series,
    forecast_value: float,
    forecast_horizon: int,
    historical_days: int = 90
) -> pd.DataFrame:
    """
    Prepare DataFrame with historical and forecast data for visualization.
    
    Args:
        historical_volatility: Historical volatility series
        forecast_value: Forecasted volatility value
        forecast_horizon: Number of days to forecast
        historical_days: Number of historical days to include
    
    Returns:
        DataFrame with date, volatility, and type columns
    """
    # Get last historical_days of historical data
    hist_vol = historical_volatility.iloc[-min(historical_days, len(historical_volatility)):]
    
    # Create historical DataFrame
    hist_df = pd.DataFrame({
        'date': hist_vol.index if hasattr(hist_vol.index, 'to_list') else pd.date_range(end=pd.Timestamp.today(), periods=len(hist_vol), freq='D'),
        'volatility': hist_vol.values,
        'type': 'historical'
    })
    
    # Get last date
    if hasattr(hist_vol.index, 'to_list'):
        last_date = pd.to_datetime(hist_vol.index[-1])
    else:
        last_date = pd.Timestamp.today()
    
    # Create forecast DataFrame
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'volatility': [forecast_value] * forecast_horizon,
        'type': 'forecast'
    })
    
    # Combine
    combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
    
    return combined_df


def calculate_y_axis_range(
    values: List[float],
    padding_pct: float = 0.15,
    min_range: float = 1.0,
    default_max: float = 5.0
) -> Tuple[float, float]:
    """
    Calculate a robust y-axis range for volatility charts.
    
    Args:
        values: List of all numeric values to show on chart
        padding_pct: Percentage of range to add as padding
        min_range: Minimum allowed range (max - min)
        default_max: Default max if all values are 0 or empty
        
    Returns:
        Tuple of (y_min, y_max)
    """
    # Filter out NaNs and infinities
    clean_values = [
        float(v) for v in values 
        if v is not None and not (pd.isna(v) or np.isnan(v) or np.isinf(v))
    ]
    
    if not clean_values:
        return 0.0, default_max
        
    v_min = min(clean_values)
    v_max = max(clean_values)
    
    if v_min == v_max:
        if v_min == 0:
            return 0.0, default_max
        # If all values are the same and non-zero, center around it
        return max(0, v_min - min_range/2), v_min + min_range/2
        
    v_range = v_max - v_min
    
    # Adaptive padding
    if v_range < 0.1:
        padding = max(v_max * 0.2, 0.1)
    elif v_range < 1.0:
        padding = v_range * padding_pct
    else:
        padding = v_range * 0.10  # 10% for larger ranges
        
    # Minimum padding logic
    padding = max(padding, v_max * 0.05, 0.2)
    
    y_min = max(0, v_min - padding)
    y_max = v_max + padding
    
    # Ensure minimum range visibility
    if (y_max - y_min) < min_range:
        center = (y_min + y_max) / 2
        y_min = max(0, center - min_range/2)
        y_max = center + min_range/2
        
    return float(y_min), float(y_max)


def clean_series_for_plotly(series: pd.Series) -> Tuple[List, List]:
    """
    Clean a pandas series for Plotly, removing NaNs.
    
    Returns:
        Tuple of (x_values, y_values)
    """
    clean = series.dropna()
    if len(clean) == 0:
        return [], []
    
    # Convert index to list (handles dates correctly)
    x = clean.index.tolist()
    y = clean.values.tolist()
    
    return x, y


def calculate_forecast_statistics(
    historical_volatility: pd.Series,
    forecast_value: float,
    current_volatility: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive forecast statistics.
    
    Args:
        historical_volatility: Historical volatility series
        forecast_value: Forecasted volatility value
        current_volatility: Current volatility (if None, uses last historical value)
    
    Returns:
        Dictionary with various statistics
    """
    if current_volatility is None:
        # Historical volatility is in decimal form, convert to percentage
        current_volatility = float(historical_volatility.iloc[-1]) * 100 if len(historical_volatility) > 0 else 0.0
    
    # Calculate averages (historical volatility is in decimal form, convert to percentage)
    avg_30 = (historical_volatility.iloc[-30:].mean() * 100) if len(historical_volatility) >= 30 else (historical_volatility.mean() * 100)
    avg_60 = (historical_volatility.iloc[-60:].mean() * 100) if len(historical_volatility) >= 60 else (historical_volatility.mean() * 100)
    avg_90 = (historical_volatility.iloc[-90:].mean() * 100) if len(historical_volatility) >= 90 else (historical_volatility.mean() * 100)
    
    # Calculate changes (both values should be in percentage form)
    change_abs = forecast_value - current_volatility
    change_pct = (change_abs / current_volatility * 100) if current_volatility > 0 else 0.0
    
    return {
        'current_volatility': float(current_volatility),
        'forecast_volatility': float(forecast_value),
        'change_abs': float(change_abs),
        'change_pct': float(change_pct),
        'avg_30d': float(avg_30),
        'avg_60d': float(avg_60),
        'avg_90d': float(avg_90),
    }

