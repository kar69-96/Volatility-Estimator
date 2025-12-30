"""
Utility functions for the volatility estimator stack.

This module provides helper functions for:
- Date parsing and validation
- Logging configuration
- Configuration loading
- Caching utilities
- Data validation
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.config_loader import (
    ConfigError,
    DataError,
    ValidationError,
    load_config,
    validate_config,
)
from src.utils.logging import setup_logging
from src.utils.cache import ensure_directory, get_cache_path


def parse_date(date_str: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Parse a date string or date object into a pandas Timestamp.

    Args:
        date_str: Date as string (YYYY-MM-DD), datetime, or Timestamp

    Returns:
        pandas Timestamp object

    Raises:
        ValidationError: If date cannot be parsed
    """
    try:
        if isinstance(date_str, pd.Timestamp):
            return date_str
        if isinstance(date_str, datetime):
            return pd.Timestamp(date_str)
        if isinstance(date_str, str):
            return pd.Timestamp(date_str)
        raise ValidationError(f"Cannot parse date: {date_str}")
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid date format: {date_str}") from e


def validate_date_range(
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp]
) -> tuple:
    """
    Validate that start_date is before end_date.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Tuple of (start_date, end_date) as Timestamps

    Raises:
        ValidationError: If dates are invalid or start_date >= end_date
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start >= end:
        raise ValidationError(
            f"Start date ({start}) must be before end date ({end})"
        )

    return start, end


def validate_numeric_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value"
) -> float:
    """
    Validate that a numeric value is within a specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter for error messages

    Returns:
        The validated value

    Raises:
        ValidationError: If value is outside the range
    """
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )
    return value


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero

    Returns:
        Result of division or default
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def load_events(csv_path: str, event_type: Optional[str] = None) -> pd.DataFrame:
    """
    Load events from CSV file.

    Args:
        csv_path: Path to events CSV file
        event_type: Optional filter by event type (e.g., 'CPI', 'FOMC', 'CRISIS')

    Returns:
        DataFrame with columns: date, event_type, description, importance

    Raises:
        DataError: If file cannot be loaded or is invalid
    """
    try:
        df = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = ['date', 'event_type', 'description', 'importance']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns in events CSV: {missing_cols}")

        # Parse dates
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Filter by event type if specified
        if event_type:
            df = df[df['event_type'] == event_type].copy()

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        return df

    except FileNotFoundError:
        raise DataError(f"Events file not found: {csv_path}")
    except Exception as e:
        raise DataError(f"Failed to load events from {csv_path}: {str(e)}") from e


__all__ = [
    # Exceptions
    'ConfigError',
    'DataError',
    'ValidationError',
    # Config
    'load_config',
    'validate_config',
    # Logging
    'setup_logging',
    # Cache
    'ensure_directory',
    'get_cache_path',
    # Date utilities
    'parse_date',
    'validate_date_range',
    # Validation
    'validate_numeric_range',
    'safe_divide',
    # Events
    'load_events',
]

