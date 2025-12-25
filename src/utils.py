"""
Utility functions for the volatility estimator stack.

This module provides helper functions for date parsing, data validation,
logging setup, and custom exception classes.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd


# Custom Exception Classes
class DataError(Exception):
    """Raised when there are issues with data loading or validation."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ConfigError(Exception):
    """Raised when there are issues with configuration."""
    pass


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration with file and console handlers.

    Args:
        log_file: Path to log file. If None, only console logging.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to enable console logging

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("volatility_estimator")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


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


def validate_date_range(start_date: Union[str, pd.Timestamp], 
                       end_date: Union[str, pd.Timestamp]) -> tuple[pd.Timestamp, pd.Timestamp]:
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


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def validate_numeric_range(value: float, min_val: float, max_val: float, 
                          name: str = "value") -> float:
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

