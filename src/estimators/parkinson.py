"""
Parkinson Volatility Estimator.

Uses intraday high/low range instead of just closing prices.
More efficient than close-to-close (uses 2 data points vs 1).

Formula:
    σ² = (1/(4ln2)) * (1/n) * Σ(ln(Hᵢ/Lᵢ))²

Where:
    Hᵢ = High price on day i
    Lᵢ = Low price on day i
    n = window size

Annualized: σ_annual = σ_daily * √252

Reference:
    Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance
    of the Rate of Return." Journal of Business, 53(1), 61-65.
"""

import numpy as np
import pandas as pd

from src.estimators.base import BaseEstimator
from src.estimators.close_to_close import CloseToCloseEstimator
from src.data.returns import calculate_ranges


class ParkinsonEstimator(BaseEstimator):
    """
    Parkinson volatility estimator using high/low range.

    More efficient than close-to-close as it uses intraday information.
    """

    def __init__(self, window: int = 60, annualization_factor: int = 252):
        """
        Initialize Parkinson estimator.

        Args:
            window: Rolling window size (default: 60 trading days)
            annualization_factor: Days per year (default: 252)
        """
        super().__init__(window, annualization_factor)
        self.constant = 1 / (4 * np.log(2))  # 1/(4ln2)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Parkinson volatility.

        Args:
            data: DataFrame with 'high' and 'low' columns

        Returns:
            Series of daily volatility estimates
        """
        required_cols = ['high', 'low']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data must contain columns: {missing_cols}")

        # Calculate high/low ratios
        ranges = calculate_ranges(
            high=data['high'],
            low=data['low'],
            open=data.get('open', pd.Series()),
            close=data.get('close', pd.Series())
        )

        # Calculate ln(H/L) squared
        high_low_ratio = ranges['high_low_ratio']
        squared_ratio = high_low_ratio ** 2

        # Handle zero range days (H = L) - use close-to-close as fallback
        zero_range_mask = (data['high'] == data['low']) | high_low_ratio.isna()

        if zero_range_mask.any() and 'close' in data.columns:
            # Fallback to close-to-close for zero range days
            fallback_estimator = CloseToCloseEstimator(
                window=self.window,
                annualization_factor=self.annualization_factor
            )
            fallback_vol = fallback_estimator.calculate(data)
            fallback_var = fallback_vol ** 2  # Convert to variance

            # Use fallback for zero range days
            squared_ratio[zero_range_mask] = fallback_var[zero_range_mask]

        # Calculate rolling mean of squared ratios
        rolling_mean = squared_ratio.rolling(
            window=self.window,
            min_periods=self.window
        ).mean()

        # Apply constant: σ² = (1/(4ln2)) * mean(ln(H/L)²)
        variance = self.constant * rolling_mean

        # Convert variance to volatility
        volatility = np.sqrt(variance)

        return volatility


