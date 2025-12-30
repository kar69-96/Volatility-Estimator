"""
Yang-Zhang Volatility Estimator.

Most comprehensive range-based estimator. Accounts for:
- Overnight volatility (close to open gaps)
- Open-to-close volatility
- Intraday range (via Rogers-Satchell)

Formula:
    σ² = σ²_overnight + kσ²_open + (1-k)σ²_RS

Where:
    σ²_overnight: Variance of overnight returns (close to open)
    σ²_open: Variance of open-to-open returns
    σ²_RS: Rogers-Satchell estimator
    k = 0.34/(1.34 + (n+1)/(n-1))

Annualized: σ_annual = σ_daily * √252

Reference:
    Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation
    Based on High, Low, Open, and Close Prices." Journal of Business,
    73(3), 477-491.
"""

import numpy as np
import pandas as pd

from src.estimators.base import BaseEstimator
from src.estimators.rogers_satchell import RogersSatchellEstimator
from src.data.returns import calculate_overnight_returns, calculate_open_returns


class YangZhangEstimator(BaseEstimator):
    """
    Yang-Zhang volatility estimator.

    Most comprehensive estimator, accounts for overnight gaps and drift.
    Considered the most accurate range-based estimator.
    """

    def __init__(self, window: int = 60, annualization_factor: int = 252):
        """
        Initialize Yang-Zhang estimator.

        Args:
            window: Rolling window size (default: 60 trading days)
            annualization_factor: Days per year (default: 252)
        """
        super().__init__(window, annualization_factor)

        # Calculate k parameter
        # k = 0.34/(1.34 + (n+1)/(n-1))
        n = window
        if n > 1:
            self.k = 0.34 / (1.34 + (n + 1) / (n - 1))
        else:
            self.k = 0.34 / 1.34  # Fallback

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Yang-Zhang volatility.

        Args:
            data: DataFrame with 'high', 'low', 'open', 'close' columns

        Returns:
            Series of daily volatility estimates
        """
        required_cols = ['high', 'low', 'open', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data must contain columns: {missing_cols}")

        # Component 1: Overnight volatility (close to open)
        overnight_returns = calculate_overnight_returns(
            open_prices=data['open'],
            close_prices=data['close']
        )

        # Rolling variance of overnight returns
        overnight_var = overnight_returns.rolling(
            window=self.window,
            min_periods=self.window
        ).var(ddof=1)

        # Component 2: Open-to-open volatility
        open_returns = calculate_open_returns(data['open'])

        # Rolling variance of open returns
        open_var = open_returns.rolling(
            window=self.window,
            min_periods=self.window
        ).var(ddof=1)

        # Component 3: Rogers-Satchell estimator
        rs_estimator = RogersSatchellEstimator(
            window=self.window,
            annualization_factor=self.annualization_factor
        )
        rs_vol = rs_estimator.calculate(data)
        rs_var = rs_vol ** 2  # Convert to variance

        # Combine components: σ² = σ²_overnight + kσ²_open + (1-k)σ²_RS
        variance = overnight_var + self.k * open_var + (1 - self.k) * rs_var

        # Convert variance to volatility
        volatility = np.sqrt(variance)

        return volatility


