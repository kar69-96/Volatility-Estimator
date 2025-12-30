"""
Data loading, caching, and preprocessing module.

This module handles:
- Market data fetching from APIs (yfinance)
- Data caching (parquet files)
- Data validation and quality checks
- Return calculations (log, simple, range-based)
- Implied volatility calculation (Black-Scholes)
"""

from src.data.data_loader import (
    get_market_data,
    check_data_quality,
    get_implied_volatility,
    get_risk_free_rate,
)
from src.data.returns import (
    calculate_returns,
    calculate_ranges,
    calculate_overnight_returns,
    calculate_open_returns,
)

__all__ = [
    'get_market_data',
    'check_data_quality',
    'get_implied_volatility',
    'get_risk_free_rate',
    'calculate_returns',
    'calculate_ranges',
    'calculate_overnight_returns',
    'calculate_open_returns',
]

