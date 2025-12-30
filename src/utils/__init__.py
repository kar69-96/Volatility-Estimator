"""
Utility functions and classes for the volatility estimator stack.

This module provides common utilities used across the codebase:
- Custom exception classes
- Date parsing utilities
- File system utilities
- Logging setup
- Data validation functions
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Union

import pandas as pd


# Custom Exception Classes
class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class DataError(Exception):
    """Raised when data-related operations fail."""
    pass


def parse_date(date_input: Union[str, date, datetime, pd.Timestamp]) -> date:
    """
    Parse a date input into a date object.
    
    Args:
        date_input: Date as string (YYYY-MM-DD), date, datetime, or pd.Timestamp
        
    Returns:
        date object
        
    Raises:
        ValidationError: If date cannot be parsed
    """
    if isinstance(date_input, date):
        return date_input
    elif isinstance(date_input, datetime):
        return date_input.date()
    elif isinstance(date_input, pd.Timestamp):
        return date_input.date()
    elif isinstance(date_input, str):
        try:
            # Try parsing as YYYY-MM-DD
            return datetime.strptime(date_input, '%Y-%m-%d').date()
        except ValueError:
            # Try other common formats
            for fmt in ['%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(date_input, fmt).date()
                except ValueError:
                    continue
            raise ValidationError(f"Unable to parse date: {date_input}")
    else:
        raise ValidationError(f"Unsupported date type: {type(date_input)}")


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to directory (string or Path object)
        
    Returns:
        Path object for the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_events(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load events from a CSV file.
    
    Expected CSV columns:
    - date: Event date (YYYY-MM-DD)
    - event_type: Type of event
    - description: Event description
    - importance: Event importance (high, medium, low)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with events data
        
    Raises:
        DataError: If file cannot be read or required columns are missing
    """
    path = Path(csv_path)
    if not path.exists():
        raise DataError(f"Events file not found: {csv_path}")
    
    try:
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = ['date', 'event_type', 'description']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataError(f"Missing required columns in events file: {missing_cols}")
        
        # Ensure date column is string type for consistency
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
        
        return df
    except pd.errors.EmptyDataError:
        raise DataError(f"Events file is empty: {csv_path}")
    except Exception as e:
        raise DataError(f"Failed to load events file {csv_path}: {str(e)}")


def setup_logging(
    log_file: Union[str, Path, None] = None,
    log_level: str = 'INFO',
    console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to output logs to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('volatility_estimator')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        ensure_directory(log_path.parent)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_numeric_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str = "value"
) -> Union[int, float]:
    """
    Validate that a numeric value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter (for error messages)
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is outside the range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if value < min_val or value > max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )
    
    return value


__all__ = [
    'ValidationError',
    'DataError',
    'parse_date',
    'ensure_directory',
    'load_events',
    'setup_logging',
    'validate_numeric_range',
]

