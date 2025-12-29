"""
Technical Indicators Module.

Implements common technical indicators used as features for volatility prediction:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- And more...
"""

from typing import Tuple

import numpy as np
import pandas as pd


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
    
    Args:
        prices: Series of closing prices
        period: Lookback period (default: 14)
        
    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    # Use exponential moving average for smoothing
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Neutral RSI for NaN values


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD shows the relationship between two EMAs of prices.
    
    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    at a specified number of standard deviations.
    
    Args:
        prices: Series of closing prices
        period: Lookback period for SMA (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        
    Returns:
        Tuple of (Upper Band, Middle Band, Lower Band)
    """
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)
    
    return upper_band, middle_band, lower_band


def calculate_bollinger_width(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.Series:
    """
    Calculate Bollinger Band Width.
    
    Measures the width of the Bollinger Bands as a percentage of the middle band.
    Useful for detecting volatility squeezes.
    
    Args:
        prices: Series of closing prices
        period: Lookback period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        
    Returns:
        Series of band width percentages
    """
    upper, middle, lower = calculate_bollinger_bands(prices, period, num_std)
    width = ((upper - lower) / middle) * 100
    return width


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR measures market volatility by decomposing the entire range
    of an asset price for that period.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: Lookback period (default: 14)
        
    Returns:
        Series of ATR values
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr


def calculate_atr_percent(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate ATR as percentage of closing price.
    
    Normalized ATR for comparing across different price levels.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: Lookback period (default: 14)
        
    Returns:
        Series of ATR percentage values
    """
    atr = calculate_atr(high, low, close, period)
    return (atr / close) * 100


def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Series of prices
        period: Lookback period
        
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Series of prices
        period: EMA span
        
    Returns:
        Series of EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Price Momentum.
    
    Momentum = Current Price - Price N periods ago
    
    Args:
        prices: Series of prices
        period: Lookback period
        
    Returns:
        Series of momentum values
    """
    return prices - prices.shift(period)


def calculate_rate_of_change(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Rate of Change (ROC).
    
    ROC = ((Current - N periods ago) / N periods ago) * 100
    
    Args:
        prices: Series of prices
        period: Lookback period
        
    Returns:
        Series of ROC percentages
    """
    shifted = prices.shift(period)
    return ((prices - shifted) / shifted) * 100


def calculate_stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Measures the location of the close relative to the high-low range
    over a specified period.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        k_period: %K lookback period (default: 14)
        d_period: %D smoothing period (default: 3)
        
    Returns:
        Tuple of (%K, %D) series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Williams %R.
    
    Similar to stochastic oscillator but measures overbought/oversold levels.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: Lookback period (default: 14)
        
    Returns:
        Series of Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    return williams_r


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures the strength of a trend regardless of its direction.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: Lookback period (default: 14)
        
    Returns:
        Series of ADX values (0-100)
    """
    # Calculate True Range
    tr = calculate_atr(high, low, close, 1) * 1  # Single period TR
    
    # Calculate directional movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smooth the indicators
    atr = calculate_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV uses volume flow to predict changes in stock price.
    
    Args:
        close: Series of closing prices
        volume: Series of trading volumes
        
    Returns:
        Series of OBV values
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    
    obv = (direction * volume).cumsum()
    
    return obv


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        volume: Series of trading volumes
        
    Returns:
        Series of VWAP values
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    return vwap


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of log returns
    """
    return np.log(prices / prices.shift(1))


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate realized volatility from returns.
    
    Args:
        returns: Series of log returns
        window: Rolling window size
        annualize: Whether to annualize (default: True)
        trading_days: Trading days per year (default: 252)
        
    Returns:
        Series of realized volatility values
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol * 100  # Convert to percentage


def calculate_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate Parkinson volatility estimator.
    
    Uses high-low range for more efficient volatility estimation.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        window: Rolling window size
        annualize: Whether to annualize
        trading_days: Trading days per year
        
    Returns:
        Series of Parkinson volatility values
    """
    log_hl = np.log(high / low)
    factor = 1 / (4 * np.log(2))
    
    vol = np.sqrt(factor * (log_hl ** 2).rolling(window=window).mean())
    
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol * 100


def calculate_garman_klass_volatility(
    open_prices: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator.
    
    Combines open, high, low, close for improved efficiency.
    
    Args:
        open_prices: Series of opening prices
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        window: Rolling window size
        annualize: Whether to annualize
        trading_days: Trading days per year
        
    Returns:
        Series of Garman-Klass volatility values
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_prices)
    
    term1 = 0.5 * (log_hl ** 2)
    term2 = (2 * np.log(2) - 1) * (log_co ** 2)
    
    vol = np.sqrt((term1 - term2).rolling(window=window).mean())
    
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol * 100







