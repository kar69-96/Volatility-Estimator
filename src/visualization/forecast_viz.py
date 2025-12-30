"""
Forecast visualization helper functions.

This module provides utilities for calculating forecast metrics,
regime classification, percentiles, and other visualization components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np


def calculate_historical_volatility(
    df: pd.DataFrame,
    window: int = 20,
    method: str = 'yang_zhang',
    use_implied_vol: bool = True,
    symbol: Optional[str] = None,
    horizon_days: int = 30
) -> pd.Series:
    """
    Calculate historical realized volatility with fluctuation, using Black-Scholes implied volatility for calibration.
    
    Args:
        df: DataFrame with OHLC data
        window: Rolling window size
        method: Method to use ('yang_zhang', 'close_to_close', etc.)
        use_implied_vol: If True, use Black-Scholes implied volatility for calibration
        symbol: Asset symbol (required if use_implied_vol=True)
        horizon_days: Horizon for implied volatility calculation
    
    Returns:
        Series of historical volatility values (in decimal form, e.g., 0.20 for 20%) with natural fluctuation
    """
    from src.data.returns import calculate_returns
    from src.estimators import get_estimator
    
    config = {'volatility': {'annualization_factor': 252}}
    annualization = 252
    
    # ALWAYS calculate realized volatility to show fluctuation
    volatility = None
    try:
        if method == 'yang_zhang':
            estimator = get_estimator('yang_zhang', window, annualization)
        else:
            estimator = get_estimator('close_to_close', window, annualization)
        
        volatility = estimator.compute(df, annualize=True)
    except Exception:
        # Fallback: calculate simple rolling volatility from returns
        try:
            returns = calculate_returns(df)
            volatility = returns.rolling(window=window).std() * np.sqrt(252)
        except Exception:
            pass
    
    # If we still don't have volatility, create it from close-to-close returns
    if volatility is None or len(volatility) == 0:
        try:
            # Calculate log returns from close prices
            close_prices = df['close']
            returns = np.log(close_prices / close_prices.shift(1))
            volatility = returns.rolling(window=window).std() * np.sqrt(252)
        except Exception:
            return pd.Series(dtype=float)
    
    # Calibrate to Black-Scholes implied volatility if available
    if use_implied_vol and symbol and len(volatility) > 0:
        try:
            from src.data import get_implied_volatility
            current_iv = get_implied_volatility(symbol, df, horizon_days=horizon_days, use_api=True)
            
            if current_iv is not None:
                # Convert IV from percentage to decimal
                current_iv_decimal = current_iv / 100.0
                
                # Get current realized volatility (last value)
                current_realized = volatility.dropna().iloc[-1] if len(volatility.dropna()) > 0 else None
                
                if current_realized is not None and current_realized > 0:
                    # Calculate calibration factor: IV / Realized
                    # This preserves the shape of historical volatility but scales it to match current IV
                    calibration_factor = current_iv_decimal / current_realized
                    
                    # Apply gradual calibration to recent data (last 60 days)
                    # Older data keeps more of its original value, newer data is calibrated to IV
                    recent_days = min(60, len(volatility))
                    if recent_days > 0:
                        for i in range(max(0, len(volatility) - recent_days), len(volatility)):
                            days_ago = len(volatility) - i - 1
                            # Gradual weight: 0% adjustment for oldest (60 days ago), 100% for newest (today)
                            weight = 1.0 - (days_ago / recent_days)
                            # Blend between original volatility and calibrated volatility
                            original_vol = volatility.iloc[i]
                            calibrated_vol = original_vol * calibration_factor
                            volatility.iloc[i] = original_vol * (1 - weight) + calibrated_vol * weight
        except Exception:
            # If IV fetch or calibration fails, use realized volatility as-is
            pass
    
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
    padding_pct: float = 0.20,
    min_range: float = 3.0,
    default_max: float = 5.0
) -> Tuple[float, float]:
    """
    Calculate a robust y-axis range for volatility charts.
    
    Args:
        values: List of all numeric values to show on chart
        padding_pct: Percentage of range to add as padding (default 20%)
        min_range: Minimum allowed range (max - min) for visibility (default 3%)
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
        # If all values are the same and non-zero, center around it with good range
        return max(0, v_min - min_range/2), v_min + min_range/2
        
    v_range = v_max - v_min
    
    # Ensure minimum range for visibility
    if v_range < min_range:
        # Expand range to min_range, centered on the data
        center = (v_min + v_max) / 2
        v_min = center - min_range / 2
        v_max = center + min_range / 2
        v_range = min_range
    
    # Calculate padding based on range
    if v_range < 2.0:
        # For small ranges (< 2%), use 25% padding
        padding = v_range * 0.25
    elif v_range < 5.0:
        # For medium ranges (2-5%), use 20% padding
        padding = v_range * padding_pct
    else:
        # For larger ranges (> 5%), use 15% padding
        padding = v_range * 0.15
        
    # Ensure minimum padding of at least 0.5% for volatility charts
    padding = max(padding, 0.5)
    
    y_min = max(0, v_min - padding)
    y_max = v_max + padding
    
    # Round to nice numbers for better readability
    # Round y_min down to nearest 0.5
    y_min = np.floor(y_min * 2) / 2
    # Round y_max up to nearest 0.5
    y_max = np.ceil(y_max * 2) / 2
    
    # Final check: ensure we have at least min_range
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


def generate_forecast_path(
    start_value: float,
    end_value: float,
    horizon: int,
    volatility: float = 0.02,
    mean_reversion: float = 0.1
) -> List[float]:
    """
    Generate a realistic fluctuating forecast path from start to end value.
    
    Uses mean reversion and deterministic variation to create natural-looking volatility path.
    
    Args:
        start_value: Starting volatility value (percentage)
        end_value: Target volatility value at end of horizon (percentage)
        horizon: Number of days to forecast
        volatility: Daily volatility of the path (default 2%)
        mean_reversion: Mean reversion strength (0-1, higher = faster reversion)
    
    Returns:
        List of daily volatility values that fluctuate and end at end_value
    """
    if horizon <= 1:
        return [end_value]
    
    path = [start_value]
    target = end_value
    
    # Calculate step size for gradual transition
    step_size = (end_value - start_value) / horizon
    
    # Use deterministic pseudo-randomness based on start/end values for reproducibility
    # This creates consistent patterns while still showing fluctuation
    seed_value = int((start_value + end_value) * 1000) % 10000
    np.random.seed(seed_value)
    
    for i in range(1, horizon):
        # Current position
        current = path[-1]
        
        # Mean reversion component: pull towards target
        distance_to_target = target - current
        reversion = distance_to_target * mean_reversion
        
        # Deterministic variation component: add realistic fluctuation
        # Use smaller variation as we approach the end
        progress = i / horizon
        # Create more volatile variation using sine wave + larger random component
        # Increased multipliers for more dramatic fluctuations
        variation = np.sin(i * 0.5) * volatility * 0.8 * (1 - progress * 0.3)  # Increased from 0.3 to 0.8
        random_component = np.random.normal(0, volatility * 1.2 * (1 - progress * 0.3))  # Increased from 0.7 to 1.2
        # Add additional high-frequency component for more realistic volatility
        high_freq = np.sin(i * 2.0) * volatility * 0.4 * (1 - progress * 0.5)
        total_variation = variation + random_component + high_freq
        
        # Trend component: gradual movement towards target
        trend = step_size
        
        # Combine components
        next_value = current + trend + reversion + total_variation
        
        # Ensure non-negative
        next_value = max(0.1, next_value)
        
        path.append(next_value)
    
    # Ensure last value is exactly the target
    path[-1] = end_value
    
    return path


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

