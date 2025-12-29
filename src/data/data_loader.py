"""
Data loading and caching module for market data.

This module handles fetching OHLC data from APIs, caching to parquet files,
and data validation.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm

from src.utils import DataError, ValidationError, ensure_directory, parse_date


def fetch_data(
    symbol: str,
    start_date: str,
    end_date: str,
    retry_attempts: int = 3,
    rate_limit_delay: float = 1.0
) -> pd.DataFrame:
    """
    Fetch OHLC data from yfinance API with retry logic.

    Args:
        symbol: Asset symbol (e.g., 'SPY')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        retry_attempts: Number of retry attempts on failure
        rate_limit_delay: Delay between API requests (seconds)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume

    Raises:
        DataError: If data cannot be fetched after retries
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    # Rate limiting
    time.sleep(rate_limit_delay)

    for attempt in range(1, retry_attempts + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)

            if df.empty:
                raise DataError(f"No data returned for {symbol}")

            # Reset index to get date as column
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.date

            # Rename columns to lowercase
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Select only required columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            return df

        except Exception as e:
            if attempt < retry_attempts:
                wait_time = 2 ** (attempt - 1)  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                raise DataError(
                    f"Failed to fetch data for {symbol} after {retry_attempts} attempts: {str(e)}"
                ) from e

    raise DataError(f"Unexpected error fetching data for {symbol}")


def validate_ohlc_data(df: pd.DataFrame, raise_errors: bool = False) -> pd.DataFrame:
    """
    Validate and fix OHLC data relationships.

    Fixes:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - Drops rows with missing critical values
    - Drops rows with non-positive prices

    Args:
        df: DataFrame with OHLC columns
        raise_errors: If True, raise ValidationError instead of fixing/removing invalid data

    Returns:
        DataFrame with validated and cleaned data

    Raises:
        ValidationError: If raise_errors=True and validation fails
    """
    if df is None or df.empty:
        if raise_errors:
            raise ValidationError("Input DataFrame is empty")
        return pd.DataFrame()

    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        if raise_errors:
            raise ValidationError(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    # Make a copy to avoid modifying original
    df = df.copy()

    # Check for missing values
    missing_mask = df[required_cols].isna().any(axis=1)
    if missing_mask.sum() > 0:
        if raise_errors:
            raise ValidationError("Missing values in required OHLC columns")
        df = df[~missing_mask].copy()
    
    if df.empty:
        if raise_errors:
            raise ValidationError("No valid data after removing missing values")
        return pd.DataFrame()

    # Check for non-positive prices
    non_positive = (df[required_cols] <= 0).any(axis=1)
    if non_positive.sum() > 0:
        if raise_errors:
            raise ValidationError("non-positive prices found in OHLC data")
        df = df[~non_positive].copy()
    
    if df.empty:
        if raise_errors:
            raise ValidationError("No valid data after removing non-positive prices")
        return pd.DataFrame()

    # Check for invalid OHLC relationships
    max_oc = df[['open', 'close']].max(axis=1)
    invalid_high = df['high'] < max_oc
    invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
    invalid_hl = df['high'] < df['low']
    
    if (invalid_high.sum() > 0 or invalid_low.sum() > 0 or invalid_hl.sum() > 0):
        if raise_errors:
            raise ValidationError("invalid OHLC relationships detected")
        # Fix OHLC relationships automatically
        if invalid_high.sum() > 0:
            df.loc[invalid_high, 'high'] = max_oc[invalid_high]
        
        min_oc = df[['open', 'close']].min(axis=1)
        if invalid_low.sum() > 0:
            df.loc[invalid_low, 'low'] = min_oc[invalid_low]
        
        if invalid_hl.sum() > 0:
            df.loc[invalid_hl, 'high'] = df.loc[invalid_hl, 'low']

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    return df


def save_to_cache(df: pd.DataFrame, symbol: str, cache_dir: str, 
                 cache_format: str = 'parquet') -> Path:
    """
    Save DataFrame to cache directory.

    Args:
        df: DataFrame to cache
        symbol: Asset symbol
        cache_dir: Cache directory path
        cache_format: Format to save ('parquet' or 'csv')

    Returns:
        Path to cached file
    """
    cache_path = ensure_directory(cache_dir)
    file_path = cache_path / f"{symbol}.{cache_format}"

    if cache_format == 'parquet':
        df.to_parquet(file_path, index=False)
    elif cache_format == 'csv':
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported cache format: {cache_format}")

    return file_path


def load_from_cache(symbol: str, cache_dir: str, 
                   cache_format: str = 'parquet') -> Optional[pd.DataFrame]:
    """
    Load cached data for a symbol.

    Args:
        symbol: Asset symbol
        cache_dir: Cache directory path
        cache_format: Format to load ('parquet' or 'csv')

    Returns:
        DataFrame if cache exists, None otherwise
    """
    cache_path = Path(cache_dir)
    file_path = cache_path / f"{symbol}.{cache_format}"

    if not file_path.exists():
        return None

    try:
        if cache_format == 'parquet':
            df = pd.read_parquet(file_path)
        elif cache_format == 'csv':
            df = pd.read_csv(file_path, parse_dates=['date'])
        else:
            raise ValueError(f"Unsupported cache format: {cache_format}")

        # Ensure date column is date type
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        return df
    except Exception as e:
        # Cache file may be corrupted, return None to trigger re-fetch
        return None


def get_market_data(
    symbol: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    cache_dir: str = './data/cache',
    cache_format: str = 'parquet',
    retry_attempts: int = 3,
    rate_limit_delay: float = 1.0
) -> pd.DataFrame:
    """
    Main interface to get market data with caching.

    Flow:
    1. Check cache if use_cache=True
    2. If cache miss or invalid, fetch from API
    3. Validate data
    4. Save to cache
    5. Return data

    Args:
        symbol: Asset symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_cache: Whether to use cached data
        cache_dir: Cache directory path
        cache_format: Cache format ('parquet' or 'csv')
        retry_attempts: Number of retry attempts for API calls
        rate_limit_delay: Delay between API requests

    Returns:
        Validated DataFrame with OHLC data
    """
    # Try to load from cache first
    if use_cache:
        try:
            cached_df = load_from_cache(symbol, cache_dir, cache_format)
            if cached_df is not None and not cached_df.empty:
                # Filter by date range if needed
                cached_df['date'] = pd.to_datetime(cached_df['date'])
                start = parse_date(start_date)
                end = parse_date(end_date)

                # Check if cached data covers the requested range
                cached_start = cached_df['date'].min()
                cached_end = cached_df['date'].max()

                if cached_start <= start and cached_end >= end:
                    # Cache covers the range, filter and return
                    mask = (cached_df['date'] >= start) & (cached_df['date'] <= end)
                    filtered_df = cached_df[mask].copy()
                    filtered_df['date'] = filtered_df['date'].dt.date
                    validated_df = validate_ohlc_data(filtered_df)
                    if validated_df is not None and not validated_df.empty:
                        return validated_df
        except Exception:
            # If cache loading fails, continue to fetch from API
            pass

    # Fetch from API
    try:
        df = fetch_data(symbol, start_date, end_date, retry_attempts, rate_limit_delay)
    except Exception as e:
        raise DataError(f"Failed to fetch data for {symbol}: {str(e)}")

    if df is None or df.empty:
        raise DataError(f"No data returned for {symbol}")

    # Validate and fix data
    df = validate_ohlc_data(df)
    
    if df is None or df.empty:
        raise DataError(f"No valid data after cleaning for {symbol}")

    # Save to cache
    if use_cache:
        try:
            save_to_cache(df, symbol, cache_dir, cache_format)
        except Exception:
            # Cache save failure is non-critical, continue
            pass

    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Generate data quality report.

    Checks:
    - Missing dates (gaps > 5 days flagged)
    - Missing values percentage
    - Outlier detection (3-sigma rule)
    - Date range coverage

    Args:
        df: DataFrame with OHLC data

    Returns:
        Dictionary with quality metrics
    """
    report = {
        'total_rows': len(df),
        'date_range': (df['date'].min(), df['date'].max()) if 'date' in df.columns else None,
        'missing_values': {},
        'outliers': {},
        'date_gaps': []
    }

    # Check missing values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }

    # Check date gaps
    if 'date' in df.columns:
        df_sorted = df.sort_values('date').copy()
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        date_diffs = df_sorted['date'].diff().dt.days

        # Flag gaps > 5 days (excluding first row which will be NaN)
        large_gaps = date_diffs[date_diffs > 5]
        if len(large_gaps) > 0:
            gap_indices = large_gaps.index
            for idx in gap_indices:
                prev_date = df_sorted.loc[idx - 1, 'date']
                curr_date = df_sorted.loc[idx, 'date']
                gap_days = int(date_diffs.loc[idx])
                report['date_gaps'].append({
                    'from': prev_date.strftime('%Y-%m-%d'),
                    'to': curr_date.strftime('%Y-%m-%d'),
                    'gap_days': gap_days
                })

    # Detect outliers using 3-sigma rule
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                if std > 0:
                    outliers = values[(values < mean - 3 * std) | (values > mean + 3 * std)]
                    report['outliers'][col] = {
                        'count': len(outliers),
                        'indices': outliers.index.tolist() if len(outliers) > 0 else []
                    }

    return report


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """
    Calculate Black-Scholes option price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual, as decimal, e.g., 0.20 for 20%)
        option_type: 'call' or 'put'
    
    Returns:
        Option price
    """
    if T <= 0:
        # Option expired
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    if sigma <= 0:
        # No volatility
        if option_type == 'call':
            return max(S - K * np.exp(-r * T), 0)
        else:
            return max(K * np.exp(-r * T) - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(price, 0)  # Option price cannot be negative


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Calculate Black-Scholes vega (sensitivity to volatility).
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual, as decimal)
    
    Returns:
        Vega value
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return vega


def calculate_implied_volatility_bs(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    option_type: str = 'call',
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Optional[float]:
    """
    Calculate implied volatility using Black-Scholes inversion.
    
    Uses Newton-Raphson method to solve for sigma given market price.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annual)
        market_price: Market price of the option
        option_type: 'call' or 'put'
        max_iterations: Maximum iterations for Newton-Raphson
        tolerance: Convergence tolerance
    
    Returns:
        Implied volatility as decimal (e.g., 0.20 for 20%), or None if cannot solve
    """
    if T <= 0:
        return None
    
    if market_price <= 0:
        return None
    
    # Initial guess: use a reasonable volatility estimate
    # For most stocks, IV is typically between 10% and 100%
    sigma = 0.30  # Start with 30% as initial guess
    
    # Use Newton-Raphson method
    for i in range(max_iterations):
        # Calculate option price with current sigma
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        
        # Calculate vega (derivative with respect to volatility)
        vega = black_scholes_vega(S, K, T, r, sigma)
        
        if vega < 1e-10:  # Vega too small, cannot converge
            break
        
        # Newton-Raphson update: sigma_new = sigma_old - (price - market_price) / vega
        price_diff = price - market_price
        if abs(price_diff) < tolerance:
            # Converged
            return sigma
        
        sigma_new = sigma - price_diff / vega
        
        # Ensure sigma stays in reasonable bounds (0.1% to 500%)
        sigma_new = max(0.001, min(5.0, sigma_new))
        
        # Check for convergence
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new
        
        sigma = sigma_new
    
    # If Newton-Raphson didn't converge, try Brent's method as fallback
    try:
        def price_error(sig):
            return black_scholes_price(S, K, T, r, sig, option_type) - market_price
        
        # Brent's method finds root in interval [0.001, 5.0]
        iv = brentq(price_error, 0.001, 5.0, maxiter=100, xtol=tolerance)
        return iv
    except (ValueError, RuntimeError):
        # Brent's method failed (no root in interval or other error)
        return None


def get_risk_free_rate() -> float:
    """
    Get current risk-free rate (approximation using 10-year Treasury yield).
    
    Returns:
        Risk-free rate as decimal (e.g., 0.05 for 5%)
    """
    try:
        # Try to get 10-year Treasury rate from yfinance
        treasury = yf.Ticker("^TNX")
        hist = treasury.history(period="1d")
        if not hist.empty:
            rate = float(hist['Close'].iloc[-1]) / 100.0
            return max(0.0, min(0.10, rate))  # Clamp between 0% and 10%
    except Exception:
        pass
    
    # Default: use 4% as reasonable approximation
    return 0.04


def fetch_implied_volatility_from_options_bs(
    symbol: str,
    horizon_days: int = 30,
    retry_attempts: int = 3,
    rate_limit_delay: float = 1.0
) -> Optional[float]:
    """
    Calculate implied volatility using Black-Scholes inversion from option prices.
    
    Gets option chain data, finds ATM options, and calculates IV using
    Black-Scholes model inversion.
    
    Args:
        symbol: Asset symbol
        horizon_days: Target horizon in days
        retry_attempts: Number of retry attempts
        rate_limit_delay: Delay between requests
    
    Returns:
        Implied volatility as percentage (e.g., 15.5 for 15.5%), or None if unavailable
    """
    time.sleep(rate_limit_delay)
    
    for attempt in range(1, retry_attempts + 1):
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price is None:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    return None
            
            # Get risk-free rate
            r = get_risk_free_rate()
            
            # Get options expiration dates
            try:
                expirations = ticker.options
                if not expirations:
                    return None
            except Exception:
                return None
            
            # Find expiration closest to target horizon
            target_date = datetime.now() + timedelta(days=horizon_days)
            best_exp = None
            min_diff = float('inf')
            
            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                    days_to_exp = (exp_date - datetime.now()).days
                    if 1 <= days_to_exp <= 90:
                        diff = abs((exp_date - target_date).days)
                        if diff < min_diff:
                            min_diff = diff
                            best_exp = exp_str
                except Exception:
                    continue
            
            if best_exp is None:
                return None
            
            # Get option chain
            try:
                opt_chain = ticker.option_chain(best_exp)
                calls = opt_chain.calls
                puts = opt_chain.puts
            except Exception:
                return None
            
            # Calculate time to expiration in years
            exp_date = datetime.strptime(best_exp, '%Y-%m-%d')
            T = (exp_date - datetime.now()).days / 365.0
            
            if T <= 0:
                return None
            
            # Find ATM options (closest to current price)
            calls['strike_diff'] = abs(calls['strike'] - current_price)
            puts['strike_diff'] = abs(puts['strike'] - current_price)
            
            # Get closest calls and puts (within 5% of current price)
            atm_calls = calls[calls['strike_diff'] / current_price <= 0.05].nsmallest(10, 'strike_diff')
            atm_puts = puts[puts['strike_diff'] / current_price <= 0.05].nsmallest(10, 'strike_diff')
            
            iv_values = []
            
            # Calculate IV from call options
            for _, call in atm_calls.iterrows():
                if pd.notna(call.get('lastPrice')) and call.get('lastPrice', 0) > 0:
                    market_price = call['lastPrice']
                    K = call['strike']
                    iv = calculate_implied_volatility_bs(
                        S=current_price,
                        K=K,
                        T=T,
                        r=r,
                        market_price=market_price,
                        option_type='call'
                    )
                    if iv is not None and 0.01 <= iv <= 2.0:  # Between 1% and 200%
                        iv_values.append(iv)
            
            # Calculate IV from put options
            for _, put in atm_puts.iterrows():
                if pd.notna(put.get('lastPrice')) and put.get('lastPrice', 0) > 0:
                    market_price = put['lastPrice']
                    K = put['strike']
                    iv = calculate_implied_volatility_bs(
                        S=current_price,
                        K=K,
                        T=T,
                        r=r,
                        market_price=market_price,
                        option_type='put'
                    )
                    if iv is not None and 0.01 <= iv <= 2.0:  # Between 1% and 200%
                        iv_values.append(iv)
            
            if len(iv_values) == 0:
                return None
            
            # Return median IV (more robust than mean)
            median_iv = float(np.median(iv_values)) * 100  # Convert to percentage
            
            return median_iv
            
        except Exception as e:
            if attempt < retry_attempts:
                wait_time = 2 ** (attempt - 1)
                time.sleep(wait_time)
                continue
            return None
    
    return None


def get_implied_volatility(
    symbol: str,
    df: pd.DataFrame,
    horizon_days: int = 30,
    use_api: bool = True,
    retry_attempts: int = 3,
    rate_limit_delay: float = 1.0
) -> Optional[float]:
    """
    Get implied volatility for a symbol, matching the forecast horizon.
    
    Uses ONLY Black-Scholes inversion to calculate IV from option market prices.
    No fallback methods - returns None if options data is unavailable.
    
    Args:
        symbol: Asset symbol
        df: DataFrame with OHLC data (not used, kept for compatibility)
        horizon_days: Target horizon in days (should match forecast horizon)
        use_api: Whether to fetch from options API (default: True)
        retry_attempts: Number of retry attempts for API calls
        rate_limit_delay: Delay between API requests
    
    Returns:
        Implied volatility as a percentage, or None if unavailable
    """
    # Use ONLY Black-Scholes calculation from option prices
    iv_bs = fetch_implied_volatility_from_options_bs(
        symbol,
        horizon_days=horizon_days,
        retry_attempts=retry_attempts,
        rate_limit_delay=rate_limit_delay
    )
    
    return iv_bs

