"""
Base estimator class for volatility estimators.

All volatility estimators inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import ValidationError


class BaseEstimator(ABC):
    """
    Abstract base class for volatility estimators.

    All estimators must implement:
    - calculate(): Compute volatility estimates
    - validate_inputs(): Validate input data
    - annualize(): Convert daily volatility to annualized
    """

    def __init__(self, window: int = 60, annualization_factor: int = 252):
        """
        Initialize estimator.

        Args:
            window: Rolling window size (trading days)
            annualization_factor: Days per year for annualization (default: 252)
        """
        self.window = window
        self.annualization_factor = annualization_factor

        # Validate window size
        if window < 10:
            raise ValidationError(f"Window size must be at least 10, got {window}")
        if window > 500:
            raise ValidationError(f"Window size must be at most 500, got {window}")

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility estimates.

        Args:
            data: DataFrame with required columns (varies by estimator)

        Returns:
            Series of volatility estimates (daily, not annualized)
        """
        pass

    def validate_inputs(self, data: pd.DataFrame) -> None:
        """
        Validate input data.

        Args:
            data: DataFrame to validate

        Raises:
            ValidationError: If data is invalid
        """
        if data.empty:
            raise ValidationError("Input data is empty")

        if len(data) < self.window:
            raise ValidationError(
                f"Insufficient data: need at least {self.window} rows, got {len(data)}"
            )

    def annualize(self, daily_volatility: pd.Series) -> pd.Series:
        """
        Convert daily volatility to annualized volatility.

        Formula: σ_annual = σ_daily * √(annualization_factor)

        Args:
            daily_volatility: Series of daily volatility estimates

        Returns:
            Series of annualized volatility estimates
        """
        return daily_volatility * np.sqrt(self.annualization_factor)

    def compute(self, data: pd.DataFrame, annualize: bool = True) -> pd.Series:
        """
        Main interface: validate, calculate, and optionally annualize.

        Args:
            data: Input data DataFrame
            annualize: Whether to annualize the results

        Returns:
            Series of volatility estimates
        """
        self.validate_inputs(data)
        daily_vol = self.calculate(data)

        if annualize:
            return self.annualize(daily_vol)
        else:
            return daily_vol


