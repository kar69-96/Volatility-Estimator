"""
Unit tests for returns calculation module.
"""

import numpy as np
import pandas as pd
import pytest

from src.returns import (
    calculate_returns,
    calculate_ranges,
    calculate_overnight_returns,
    calculate_open_returns,
    forward_fill_returns
)


class TestCalculateReturns:
    """Tests for calculate_returns function."""

    def test_log_returns_basic(self):
        """Test basic log returns calculation."""
        prices = pd.Series([100, 105, 102, 108])
        returns = calculate_returns(prices, method='log')

        # First value should be NaN (no previous price)
        assert pd.isna(returns.iloc[0])

        # Manual calculation: ln(105/100) ≈ 0.04879
        expected = np.log(105 / 100)
        assert abs(returns.iloc[1] - expected) < 1e-5

    def test_simple_returns(self):
        """Test simple returns calculation."""
        prices = pd.Series([100, 105, 102, 108])
        returns = calculate_returns(prices, method='simple')

        # First value should be NaN
        assert pd.isna(returns.iloc[0])

        # Manual calculation: (105-100)/100 = 0.05
        assert abs(returns.iloc[1] - 0.05) < 1e-5

    def test_zero_price_handling(self):
        """Test handling of zero prices."""
        prices = pd.Series([100, 0, 105])
        returns = calculate_returns(prices, method='log')

        # Zero price should result in NaN
        assert pd.isna(returns.iloc[1])

    def test_missing_data(self):
        """Test handling of missing data."""
        prices = pd.Series([100, np.nan, 105, 108])
        returns = calculate_returns(prices, method='log')

        # Missing price should result in NaN
        assert pd.isna(returns.iloc[1])


class TestCalculateRanges:
    """Tests for calculate_ranges function."""

    def test_basic_ranges(self):
        """Test basic range calculation."""
        high = pd.Series([110, 115, 112])
        low = pd.Series([100, 105, 102])
        open_prices = pd.Series([105, 110, 107])
        close = pd.Series([108, 112, 109])

        ranges = calculate_ranges(high, low, open_prices, close)

        # Check high/low ratio
        expected_hl = np.log(110 / 100)
        assert abs(ranges['high_low_ratio'].iloc[0] - expected_hl) < 1e-5

        # Check that all required columns exist
        assert 'high_low_ratio' in ranges.columns
        assert 'high_close_ratio' in ranges.columns
        assert 'high_open_ratio' in ranges.columns
        assert 'low_close_ratio' in ranges.columns
        assert 'low_open_ratio' in ranges.columns
        assert 'open_close_ratio' in ranges.columns

    def test_zero_range_handling(self):
        """Test handling of zero range (H = L)."""
        high = pd.Series([100, 100])  # H = L
        low = pd.Series([100, 100])
        open_prices = pd.Series([100, 100])
        close = pd.Series([100, 100])

        ranges = calculate_ranges(high, low, open_prices, close)

        # ln(1) = 0, but should handle gracefully
        assert ranges['high_low_ratio'].iloc[0] == 0.0


class TestOvernightReturns:
    """Tests for overnight returns calculation."""

    def test_overnight_returns(self):
        """Test overnight returns calculation."""
        open_prices = pd.Series([105, 110, 108])
        close_prices = pd.Series([100, 105, 102])

        overnight = calculate_overnight_returns(open_prices, close_prices)

        # First value should be NaN (no previous close)
        assert pd.isna(overnight.iloc[0])

        # Second: ln(110/100) = ln(1.1) ≈ 0.09531
        expected = np.log(110 / 100)
        assert abs(overnight.iloc[1] - expected) < 1e-5


class TestOpenReturns:
    """Tests for open returns calculation."""

    def test_open_returns(self):
        """Test open-to-open returns."""
        open_prices = pd.Series([100, 105, 102, 108])

        open_returns = calculate_open_returns(open_prices)

        # First value should be NaN
        assert pd.isna(open_returns.iloc[0])

        # Second: ln(105/100) ≈ 0.04879
        expected = np.log(105 / 100)
        assert abs(open_returns.iloc[1] - expected) < 1e-5

