"""
Close-to-Close Volatility Estimator.

This is the standard realized volatility estimator using closing prices.

Formula:
    σ² = (1/(n-1)) * Σ(rᵢ - r̄)²

Where:
    rᵢ = ln(P_t / P_{t-1}) - log returns
    r̄ = mean return
    n = window size

Annualized: σ_annual = σ_daily * √252

Reference:
    Standard realized volatility measure, widely used in finance.
"""

import numpy as np
import pandas as pd

from src.estimators.base import BaseEstimator
from src.data.returns import calculate_returns


class CloseToCloseEstimator(BaseEstimator):
    """
    Close-to-Close volatility estimator.

    Uses rolling window of log returns to compute realized volatility.
    This is the baseline volatility measure.
    """

    def __init__(self, window: int = 60, annualization_factor: int = 252):
        """
        Initialize close-to-close estimator.

        Args:
            window: Rolling window size (default: 60 trading days)
            annualization_factor: Days per year (default: 252)
        """
        super().__init__(window, annualization_factor)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate close-to-close volatility.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Series of daily volatility estimates
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Calculate log returns
        returns = calculate_returns(data['close'], method='log')

        # Calculate rolling variance
        # Use ddof=1 for sample standard deviation (n-1 denominator)
        rolling_var = returns.rolling(window=self.window, min_periods=self.window).var(ddof=1)

        # Convert variance to standard deviation (volatility)
        # Handle NaN values (insufficient data periods)
        volatility = np.sqrt(rolling_var)

        return volatility


