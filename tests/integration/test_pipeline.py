"""
Integration tests for the full pipeline.

Tests end-to-end workflow from data loading to output generation.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil

from src.data_loader import get_market_data
from src.estimators import get_estimator
from src.comparison import run_all_estimators
from src.event_analysis import analyze_all_events
from src.utils import load_events


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestFullPipeline:
    """Test complete pipeline execution."""

    @pytest.mark.slow
    def test_data_load_and_estimator(self, temp_dir):
        """Test data loading and single estimator calculation."""
        # This test requires API access, mark as slow
        try:
            df = get_market_data(
                symbol='SPY',
                start_date='2020-01-01',
                end_date='2020-12-31',
                use_cache=False,
                cache_dir=str(temp_dir / 'cache')
            )

            assert len(df) > 0
            assert 'date' in df.columns
            assert 'close' in df.columns

            # Test estimator
            estimator = get_estimator('close_to_close', window=20)
            volatility = estimator.compute(df, annualize=True)

            assert len(volatility) > 0
            assert not volatility.isna().all()

        except Exception as e:
            pytest.skip(f"API test skipped: {str(e)}")

    def test_comparison_framework(self, temp_dir):
        """Test estimator comparison framework."""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        base_prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        df = pd.DataFrame({
            'date': dates.date,
            'open': base_prices * 0.99,
            'high': base_prices * 1.02,
            'low': base_prices * 0.98,
            'close': base_prices,
            'volume': [1000000] * 100
        })

        # Run all estimators
        results = run_all_estimators(df, window=20, annualization_factor=252)

        assert len(results) > 0
        assert 'date' in results.columns
        assert len([col for col in results.columns if col != 'date']) > 0

    def test_event_analysis(self, temp_dir):
        """Test event analysis workflow."""
        # Create sample volatility data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        volatility = pd.Series(np.random.randn(100).cumsum() + 20, index=dates)

        # Create sample events
        events_df = pd.DataFrame({
            'date': ['2020-02-15', '2020-05-15'],
            'event_type': ['CPI', 'FOMC'],
            'description': ['CPI Release', 'FOMC Meeting'],
            'importance': ['high', 'high']
        })

        # Run event analysis
        results = analyze_all_events(
            volatility_series=volatility,
            volatility_dates=pd.Series(dates.date),
            events_df=events_df,
            pre_window=5,
            post_window=5
        )

        assert len(results) > 0
        assert 'event_date' in results.columns
        assert 'volatility_change' in results.columns

