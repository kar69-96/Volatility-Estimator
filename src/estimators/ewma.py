"""
Exponentially Weighted Moving Average (EWMA) Volatility Estimator.

This estimator gives more weight to recent observations, making it more
responsive to market shocks and volatility clustering.

Formula:
    σ²ₜ = λσ²ₜ₋₁ + (1-λ)r²ₜ

Where:
    λ (lambda): Decay factor (typically 0.90, 0.94, or 0.97)
    σ²ₜ₋₁: Previous period's variance
    r²ₜ: Current period's squared return

Initialization:
    Use first 30 days to compute initial variance (close-to-close method)

Annualized: σ_annual = σ_daily * √252

Reference:
    RiskMetrics Technical Document (J.P. Morgan, 1996)
    Standard λ = 0.94 for daily data
"""

import numpy as np
import pandas as pd

from src.estimators.base import BaseEstimator
from src.estimators.close_to_close import CloseToCloseEstimator
from src.returns import calculate_returns
from src.utils import validate_numeric_range


class EWMAEstimator(BaseEstimator):
    """
    Exponentially Weighted Moving Average volatility estimator.

    More responsive to recent shocks than close-to-close estimator.
    Captures volatility clustering phenomenon.
    """

    def __init__(
        self,
        window: int = 60,
        annualization_factor: int = 252,
        lambda_param: float = 0.94,
        initialization_window: int = 30
    ):
        """
        Initialize EWMA estimator.

        Args:
            window: Rolling window size (not used in EWMA, kept for compatibility)
            annualization_factor: Days per year (default: 252)
            lambda_param: Decay factor (default: 0.94, RiskMetrics standard)
            initialization_window: Days to use for initial variance (default: 30)
        """
        super().__init__(window, annualization_factor)

        # Validate lambda
        self.lambda_param = validate_numeric_range(
            lambda_param, 0.8, 0.99, name="lambda"
        )
        self.initialization_window = initialization_window

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate EWMA volatility.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Series of daily volatility estimates
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Calculate log returns
        returns = calculate_returns(data['close'], method='log')

        # Initialize variance using first N days (close-to-close method)
        if len(returns) < self.initialization_window:
            # Not enough data for initialization
            variance = pd.Series(index=returns.index, dtype=float)
            variance[:] = np.nan
            return np.sqrt(variance)

        # Calculate initial variance (sample variance of first N days)
        initial_returns = returns.iloc[:self.initialization_window].dropna()
        if len(initial_returns) < 2:
            variance = pd.Series(index=returns.index, dtype=float)
            variance[:] = np.nan
            return np.sqrt(variance)

        initial_variance = initial_returns.var(ddof=1)

        # Initialize variance series
        variance = pd.Series(index=returns.index, dtype=float)
        variance.iloc[0] = initial_variance

        # Calculate squared returns
        squared_returns = returns ** 2

        # EWMA recursion: σ²ₜ = λσ²ₜ₋₁ + (1-λ)r²ₜ
        for i in range(1, len(variance)):
            prev_variance = variance.iloc[i - 1]

            # Handle NaN values
            if pd.isna(prev_variance):
                variance.iloc[i] = np.nan
                continue

            current_squared_return = squared_returns.iloc[i]

            if pd.isna(current_squared_return):
                # If return is NaN, carry forward previous variance
                variance.iloc[i] = prev_variance
            else:
                # EWMA update
                variance.iloc[i] = (
                    self.lambda_param * prev_variance +
                    (1 - self.lambda_param) * current_squared_return
                )

        # Convert variance to volatility (standard deviation)
        volatility = np.sqrt(variance)

        return volatility

