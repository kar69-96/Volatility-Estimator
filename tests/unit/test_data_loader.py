"""
Unit tests for data loader module.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.data_loader import (
    validate_ohlc_data,
    save_to_cache,
    load_from_cache,
    check_data_quality
)
from src.utils import ValidationError


class TestValidateOHLCData:
    """Tests for validate_ohlc_data function."""

    def test_valid_data(self):
        """Test validation of valid OHLC data."""
        df = pd.DataFrame({
            'open': [100, 105, 102],
            'high': [110, 115, 112],
            'low': [95, 100, 97],
            'close': [108, 112, 109]
        })

        result = validate_ohlc_data(df)
        assert len(result) == 3

    def test_invalid_high_low(self):
        """Test detection of invalid high/low relationships."""
        df = pd.DataFrame({
            'open': [100, 105],
            'high': [90, 115],  # High < Open (invalid)
            'low': [95, 100],
            'close': [108, 112]
        })

        with pytest.raises(ValidationError, match="invalid OHLC"):
            validate_ohlc_data(df)

    def test_missing_columns(self):
        """Test error handling for missing columns."""
        df = pd.DataFrame({
            'open': [100, 105],
            'high': [110, 115]
            # Missing 'low' and 'close'
        })

        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_ohlc_data(df)

    def test_empty_dataframe(self):
        """Test error handling for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValidationError, match="empty"):
            validate_ohlc_data(df)

    def test_non_positive_prices(self):
        """Test detection of non-positive prices."""
        df = pd.DataFrame({
            'open': [100, -5],  # Negative price
            'high': [110, 115],
            'low': [95, 100],
            'close': [108, 112]
        })

        with pytest.raises(ValidationError, match="non-positive"):
            validate_ohlc_data(df)


class TestCheckDataQuality:
    """Tests for check_data_quality function."""

    def test_basic_quality_check(self):
        """Test basic data quality report."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'date': dates.date,
            'open': [100 + i for i in range(10)],
            'high': [110 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [105 + i for i in range(10)],
            'volume': [1000000] * 10
        })

        report = check_data_quality(df)

        assert report['total_rows'] == 10
        assert 'missing_values' in report
        assert 'outliers' in report
        assert 'date_gaps' in report

    def test_date_gap_detection(self):
        """Test detection of date gaps."""
        # Create data with a gap
        dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-10'])  # 8-day gap
        df = pd.DataFrame({
            'date': dates.date,
            'open': [100, 105, 110],
            'high': [110, 115, 120],
            'low': [95, 100, 105],
            'close': [105, 110, 115],
            'volume': [1000000] * 3
        })

        report = check_data_quality(df)

        # Should detect gap > 5 days
        assert len(report['date_gaps']) > 0

