"""
Neural GARCH Estimator.

Extends BaseEstimator to provide neural network-based GARCH volatility estimation
that integrates with the existing volatility estimator framework.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.estimators.base import BaseEstimator
from src.utils import ValidationError


class NeuralGARCHEstimator(BaseEstimator):
    """
    Neural GARCH volatility estimator.
    
    Uses a neural network to model conditional variance dynamics,
    allowing for more flexible nonlinear relationships than traditional GARCH.
    
    The model learns:
        σ²_t = f(ε²_{t-1}, ..., ε²_{t-p}, σ²_{t-1}, ..., σ²_{t-q})
    where f is a neural network.
    """
    
    def __init__(
        self,
        window: int = 60,
        annualization_factor: int = 252,
        p: int = 1,
        q: int = 1,
        hidden_layers: List[int] = None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        device: str = 'auto',
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize Neural GARCH estimator.
        
        Args:
            window: Rolling window size (for initial variance estimation)
            annualization_factor: Days per year for annualization
            p: Number of lagged squared return terms (ARCH order)
            q: Number of lagged variance terms (GARCH order)
            hidden_layers: List of hidden layer sizes (default: [32, 16])
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            device: 'auto', 'cuda', 'mps', or 'cpu'
            pretrained_path: Path to pretrained model checkpoint
        """
        super().__init__(window, annualization_factor)
        
        self.p = p
        self.q = q
        self.hidden_layers = hidden_layers or [32, 16]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.pretrained_path = pretrained_path
        
        self._model = None
        self._fitted = False
    
    def _ensure_model(self):
        """Ensure model is initialized."""
        if self._model is None:
            try:
                from src.models.neural_garch import NeuralGARCHWrapper
                
                self._model = NeuralGARCHWrapper(
                    p=self.p,
                    q=self.q,
                    hidden_layers=self.hidden_layers,
                    device=self.device,
                )
                
                # Load pretrained weights if available
                if self.pretrained_path and Path(self.pretrained_path).exists():
                    self._model.load(self.pretrained_path)
                    self._fitted = True
                    
            except ImportError as e:
                raise ValidationError(
                    f"PyTorch is required for Neural GARCH. Install with: pip install torch. Error: {e}"
                )
    
    def validate_inputs(self, data: pd.DataFrame) -> None:
        """
        Validate input data for Neural GARCH.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        super().validate_inputs(data)
        
        # Need close prices for returns
        if 'close' not in data.columns:
            raise ValidationError("Data must contain 'close' column")
        
        # Need enough data for GARCH lags
        min_required = max(self.p, self.q) + self.window
        if len(data) < min_required:
            raise ValidationError(
                f"Insufficient data: need at least {min_required} rows, got {len(data)}"
            )
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility using Neural GARCH.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Series of volatility estimates (daily, not annualized)
        """
        self._ensure_model()
        
        # Calculate log returns
        close = data['close'].values
        returns = np.log(close[1:] / close[:-1])
        
        # Fit model if not already fitted
        if not self._fitted:
            self._model.fit(
                returns=returns,
                epochs=self.epochs,
                lr=self.learning_rate,
                verbose=False,
            )
            self._fitted = True
        
        # Predict conditional volatility
        volatility = self._model.predict_volatility(returns, annualize=False)
        
        # Pad to match original data length
        # Neural GARCH needs max(p, q) initial observations
        n_pad = len(data) - len(volatility)
        
        if n_pad > 0:
            # Use rolling std for initial values
            initial_vol = pd.Series(returns[:n_pad]).rolling(
                window=min(self.window, n_pad), min_periods=1
            ).std().values * 100  # Convert to percentage
            
            volatility = np.concatenate([initial_vol, volatility])
        
        # Ensure same length as input
        if len(volatility) < len(data):
            diff = len(data) - len(volatility)
            volatility = np.concatenate([[np.nan] * diff, volatility])
        elif len(volatility) > len(data):
            volatility = volatility[-len(data):]
        
        return pd.Series(volatility, index=data.index)
    
    def fit(self, data: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Fit the Neural GARCH model.
        
        Args:
            data: DataFrame with 'close' column
            verbose: Print training progress
            
        Returns:
            Training history
        """
        self.validate_inputs(data)
        self._ensure_model()
        
        # Calculate returns
        close = data['close'].values
        returns = np.log(close[1:] / close[:-1])
        
        # Fit model
        history = self._model.fit(
            returns=returns,
            epochs=self.epochs,
            lr=self.learning_rate,
            verbose=verbose,
        )
        
        self._fitted = True
        return history
    
    def save_model(self, path: str) -> None:
        """
        Save trained model to file.
        
        Args:
            path: Path to save model
        """
        if self._model is None:
            raise ValidationError("Model not initialized")
        
        self._model.save(path)
    
    def load_model(self, path: str) -> None:
        """
        Load trained model from file.
        
        Args:
            path: Path to model file
        """
        self._ensure_model()
        self._model.load(path)
        self._fitted = True
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._fitted
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model details
        """
        self._ensure_model()
        
        return {
            'estimator': 'Neural GARCH',
            'p': self.p,
            'q': self.q,
            'hidden_layers': self.hidden_layers,
            'fitted': self._fitted,
            'device': str(self._model.device) if self._model else 'N/A',
        }


# Convenience function for creating estimator
def get_neural_garch_estimator(
    p: int = 1,
    q: int = 1,
    window: int = 60,
    **kwargs
) -> NeuralGARCHEstimator:
    """
    Create a Neural GARCH estimator with default parameters.
    
    Args:
        p: ARCH order
        q: GARCH order
        window: Rolling window size
        **kwargs: Additional arguments for NeuralGARCHEstimator
        
    Returns:
        NeuralGARCHEstimator instance
    """
    return NeuralGARCHEstimator(
        window=window,
        p=p,
        q=q,
        **kwargs
    )

