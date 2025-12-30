"""
Baseline volatility forecasting methods.

EWMA, GARCH(1,1), HAR-RV with proper implementations.
"""

import numpy as np
import pandas as pd

try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def ewma_volatility(returns, span=60, horizon=20):
    """
    EWMA volatility forecast.
    
    Forecast variance over next h days using EWMA of past squared returns.
    
    Args:
        returns: Series of returns
        span: EWMA span (default 60)
        horizon: Prediction horizon in days (default 20)
        
    Returns:
        Annualized volatility forecast as percentage
    """
    squared_returns = returns ** 2
    ewma_var = squared_returns.ewm(span=span, adjust=False).mean()
    # Scale to horizon
    forecast_var = ewma_var * horizon
    return np.sqrt(forecast_var * 252) * 100  # Annualized volatility as percentage


def garch_volatility(returns, horizon=20):
    """
    GARCH(1,1) volatility forecast.
    
    Forecast conditional variance over next h days.
    
    Args:
        returns: Series of returns
        horizon: Prediction horizon in days (default 20)
        
    Returns:
        Annualized volatility forecast as percentage
    """
    if not _ARCH_AVAILABLE:
        raise ImportError("ARCH library is required for GARCH. Install with: pip install arch")
    
    # Scale returns by 100 for numerical stability (ARCH expects percentage-like values)
    scaled_returns = returns * 100
    
    # Fit GARCH(1,1)
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    fitted = model.fit(disp='off')
    
    # Forecast h steps ahead
    forecasts = fitted.forecast(horizon=horizon, reindex=False)
    cond_var = forecasts.variance.iloc[-1, :].mean()  # Average conditional variance
    
    # Convert back from scaled returns and annualize
    # The variance was computed on scaled returns, so divide by 10000 to get variance scale
    return np.sqrt(cond_var / 10000 * 252) * 100


def har_rv(returns, horizon=20):
    """
    HAR-RV: Heterogeneous Autoregressive Realized Volatility.
    
    Proper implementation using daily, weekly, monthly RV components.
    
    Args:
        returns: Series of returns
        horizon: Prediction horizon in days (default 20)
        
    Returns:
        Annualized volatility forecast as percentage
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for HAR-RV. Install with: pip install scikit-learn")
    
    # Compute realized variances
    rv_daily = returns ** 2
    
    # Rolling windows (use mean for stability)
    rv_weekly = rv_daily.rolling(window=5, min_periods=1).mean()   # Weekly RV (avg daily)
    rv_monthly = rv_daily.rolling(window=22, min_periods=1).mean()  # Monthly RV (avg daily)
    
    # Prepare for regression
    # Target: next period's realized variance (shifted backward for alignment)
    target = rv_daily.shift(-1)
    
    # Features: lagged daily, weekly, monthly RV
    features = pd.DataFrame({
        'rv_daily': rv_daily,
        'rv_weekly': rv_weekly,
        'rv_monthly': rv_monthly
    })
    
    # Drop NaN
    valid_mask = ~(features.isna().any(axis=1) | target.isna())
    X = features[valid_mask].values
    y = target[valid_mask].values
    
    if len(X) < 100:  # Need enough data
        return np.full(len(returns), np.nan)
    
    # Fit HAR-RV regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast
    forecast_rv = model.predict(features.values)
    forecast_rv = np.maximum(forecast_rv, 1e-8)  # Ensure positive
    
    # Scale to horizon and annualize
    forecast_var = forecast_rv * horizon
    return np.sqrt(forecast_var * 252) * 100

