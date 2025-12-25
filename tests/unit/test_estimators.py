"""
Unit tests for volatility estimators.
"""

import numpy as np
import pandas as pd
import pytest

from src.estimators.close_to_close import CloseToCloseEstimator
from src.estimators.ewma import EWMAEstimator
from src.estimators.parkinson import ParkinsonEstimator
from src.estimators.rogers_satchell import RogersSatchellEstimator
from src.estimators.yang_zhang import YangZhangEstimator
from src.utils import ValidationError


class TestCloseToCloseEstimator:
    """Tests for CloseToCloseEstimator."""

    def test_basic_calculation(self):
        """Test basic volatility calculation."""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        df = pd.DataFrame({
            'date': dates.date,
            'close': prices
        })

        estimator = CloseToCloseEstimator(window=20)
        volatility = estimator.compute(df, annualize=False)

        # Should have NaN for first 19 rows (insufficient data)
        assert pd.isna(volatility.iloc[19])
        # Should have valid values after window
        assert not pd.isna(volatility.iloc[20])

    def test_annualization(self):
        """Test annualization."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        df = pd.DataFrame({
            'date': dates.date,
            'close': prices
        })

        estimator = CloseToCloseEstimator(window=20, annualization_factor=252)
        daily_vol = estimator.compute(df, annualize=False)
        annual_vol = estimator.compute(df, annualize=True)

        # Annualized should be larger
        valid_idx = ~pd.isna(annual_vol)
        assert (annual_vol[valid_idx] > daily_vol[valid_idx]).all()

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices = np.random.randn(10) * 100
        df = pd.DataFrame({
            'date': dates.date,
            'close': prices
        })

        estimator = CloseToCloseEstimator(window=20)
        with pytest.raises(ValidationError):
            estimator.compute(df)

    def test_missing_close_column(self):
        """Test error handling for missing close column."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D').date,
            'open': np.random.randn(100) * 100
        })

        estimator = CloseToCloseEstimator(window=20)
        with pytest.raises(ValueError, match="close"):
            estimator.calculate(df)


class TestEWMAEstimator:
    """Tests for EWMAEstimator."""

    def test_basic_calculation(self):
        """Test basic EWMA calculation."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
        df = pd.DataFrame({
            'date': dates.date,
            'close': prices
        })

        estimator = EWMAEstimator(window=60, lambda_param=0.94)
        volatility = estimator.compute(df, annualize=False)

        # Should have valid values after initialization period
        assert not pd.isna(volatility.iloc[30])

    def test_lambda_validation(self):
        """Test lambda parameter validation."""
        with pytest.raises(ValidationError):
            EWMAEstimator(lambda_param=0.5)  # Too low

        with pytest.raises(ValidationError):
            EWMAEstimator(lambda_param=1.5)  # Too high

        # Valid lambda should work
        estimator = EWMAEstimator(lambda_param=0.94)
        assert estimator.lambda_param == 0.94


class TestParkinsonEstimator:
    """Tests for ParkinsonEstimator."""

    def test_basic_calculation(self):
        """Test basic Parkinson calculation."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        base_prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        df = pd.DataFrame({
            'date': dates.date,
            'open': base_prices * 0.99,
            'high': base_prices * 1.02,
            'low': base_prices * 0.98,
            'close': base_prices
        })

        estimator = ParkinsonEstimator(window=20)
        volatility = estimator.compute(df, annualize=False)

        # Should have valid values after window
        assert not pd.isna(volatility.iloc[20])

    def test_missing_columns(self):
        """Test error handling for missing columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D').date,
            'close': np.random.randn(100) * 100
        })

        estimator = ParkinsonEstimator(window=20)
        with pytest.raises(ValueError, match="high"):
            estimator.calculate(df)


class TestRogersSatchellEstimator:
    """Tests for RogersSatchellEstimator."""

    def test_basic_calculation(self):
        """Test basic Rogers-Satchell calculation."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        base_prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        df = pd.DataFrame({
            'date': dates.date,
            'open': base_prices * 0.99,
            'high': base_prices * 1.02,
            'low': base_prices * 0.98,
            'close': base_prices
        })

        estimator = RogersSatchellEstimator(window=20)
        volatility = estimator.compute(df, annualize=False)

        # Should have valid values after window
        assert not pd.isna(volatility.iloc[20])


class TestYangZhangEstimator:
    """Tests for YangZhangEstimator."""

    def test_basic_calculation(self):
        """Test basic Yang-Zhang calculation."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        base_prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        df = pd.DataFrame({
            'date': dates.date,
            'open': base_prices * 0.99,
            'high': base_prices * 1.02,
            'low': base_prices * 0.98,
            'close': base_prices
        })

        estimator = YangZhangEstimator(window=20)
        volatility = estimator.compute(df, annualize=False)

        # Should have valid values after window
        assert not pd.isna(volatility.iloc[20])

    def test_k_parameter(self):
        """Test k parameter calculation."""
        estimator = YangZhangEstimator(window=60)
        # k should be between 0 and 1
        assert 0 < estimator.k < 1


class TestEstimatorRegistry:
    """Tests for estimator registry."""

    def test_get_estimator(self):
        """Test estimator factory function."""
        from src.estimators import get_estimator

        # Test each estimator
        estimators = ['close_to_close', 'ewma', 'parkinson', 'rogers_satchell', 'yang_zhang']
        for name in estimators:
            if name == 'ewma':
                est = get_estimator(name, window=60, lambda_param=0.94)
            else:
                est = get_estimator(name, window=60)
            assert est is not None

    def test_invalid_estimator(self):
        """Test error handling for invalid estimator name."""
        from src.estimators import get_estimator

        with pytest.raises(ValueError):
            get_estimator('invalid_estimator', window=60)

    def test_list_estimators(self):
        """Test listing available estimators."""
        from src.estimators import list_estimators

        estimators = list_estimators()
        assert len(estimators) == 5
        assert 'close_to_close' in estimators
        assert 'yang_zhang' in estimators

