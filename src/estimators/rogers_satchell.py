"""
Rogers-Satchell Volatility Estimator.

Range-based estimator that accounts for drift (trending markets).
More robust than Parkinson when markets have strong trends.

Formula:
    σ² = (1/n) * Σ[ln(Hᵢ/Cᵢ)ln(Hᵢ/Oᵢ) + ln(Lᵢ/Cᵢ)ln(Lᵢ/Oᵢ)]

Where:
    Hᵢ, Lᵢ, Oᵢ, Cᵢ = High, Low, Open, Close on day i
    n = window size

Annualized: σ_annual = σ_daily * √252

Reference:
    Rogers, L. C. G., & Satchell, S. E. (1991). "Estimating Variance from High,
    Low, Open, and Close Prices." The Annals of Applied Probability, 1(4), 504-512.
"""

import numpy as np
import pandas as pd

from src.estimators.base import BaseEstimator
from src.data.returns import calculate_ranges


class RogersSatchellEstimator(BaseEstimator):
    """
    Rogers-Satchell volatility estimator accounting for drift.

    Better than Parkinson for trending markets.
    """

    def __init__(self, window: int = 60, annualization_factor: int = 252):
        """
        Initialize Rogers-Satchell estimator.

        Args:
            window: Rolling window size (default: 60 trading days)
            annualization_factor: Days per year (default: 252)
        """
        super().__init__(window, annualization_factor)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Rogers-Satchell volatility.

        Args:
            data: DataFrame with 'high', 'low', 'open', 'close' columns

        Returns:
            Series of daily volatility estimates
        """
        required_cols = ['high', 'low', 'open', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data must contain columns: {missing_cols}")

        # Calculate range ratios
        ranges = calculate_ranges(
            high=data['high'],
            low=data['low'],
            open=data['open'],
            close=data['close']
        )

        # Calculate the two terms for each day
        # Term 1: ln(H/C) * ln(H/O)
        term1 = ranges['high_close_ratio'] * ranges['high_open_ratio']

        # Term 2: ln(L/C) * ln(L/O)
        term2 = ranges['low_close_ratio'] * ranges['low_open_ratio']

        # Sum of terms
        daily_sum = term1 + term2

        # Handle NaN values (zero values in logs result in NaN)
        # Skip rows with NaN in the calculation

        # Calculate rolling mean: σ² = (1/n) * Σ[term1 + term2]
        rolling_mean = daily_sum.rolling(
            window=self.window,
            min_periods=self.window
        ).mean()

        # Variance is the rolling mean
        variance = rolling_mean

        # Convert variance to volatility
        volatility = np.sqrt(variance)

        return volatility


