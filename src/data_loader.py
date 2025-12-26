"""
Data loading and caching module for market data.

This module handles fetching OHLC data from APIs, caching to parquet files,
and data validation.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
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


def validate_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and fix OHLC data relationships.

    Fixes:
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - Drops rows with missing critical values
    - Drops rows with non-positive prices

    Args:
        df: DataFrame with OHLC columns

    Returns:
        DataFrame with validated and cleaned data
    """
    if df is None or df.empty:
        return pd.DataFrame()

    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return pd.DataFrame()

    # Make a copy to avoid modifying original
    df = df.copy()

    # Drop rows with missing values in critical columns instead of raising error
    missing_mask = df[required_cols].isna().any(axis=1)
    if missing_mask.sum() > 0:
        df = df[~missing_mask].copy()
    
    if df.empty:
        return pd.DataFrame()

    # Drop rows with non-positive prices instead of raising error
    non_positive = (df[required_cols] <= 0).any(axis=1)
    if non_positive.sum() > 0:
        df = df[~non_positive].copy()
    
    if df.empty:
        return pd.DataFrame()

    # Fix OHLC relationships automatically
    # High must be >= max(Open, Close)
    max_oc = df[['open', 'close']].max(axis=1)
    invalid_high = df['high'] < max_oc
    if invalid_high.sum() > 0:
        df.loc[invalid_high, 'high'] = max_oc[invalid_high]
    
    # Low must be <= min(Open, Close)
    min_oc = df[['open', 'close']].min(axis=1)
    invalid_low = df['low'] > min_oc
    if invalid_low.sum() > 0:
        df.loc[invalid_low, 'low'] = min_oc[invalid_low]
    
    # Ensure High >= Low
    invalid_hl = df['high'] < df['low']
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

