"""
Preprocessing utilities for feature engineering.

Includes scaling, sequence creation, and data normalization.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_scaler(scaler_type: str = 'standard'):
    """
    Create a scaler instance.
    
    Args:
        scaler_type: 'standard' (z-score) or 'minmax' (0-1)
        
    Returns:
        Scaler instance
    """
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")


def scale_features(
    features: pd.DataFrame,
    scaler=None,
    fit: bool = False
) -> Tuple[pd.DataFrame, object]:
    """
    Scale features using specified scaler.
    
    Args:
        features: Feature DataFrame
        scaler: Scaler instance (if None, creates StandardScaler)
        fit: Whether to fit the scaler
        
    Returns:
        Tuple of (scaled features, fitted scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        scaled = scaler.fit_transform(features)
    else:
        scaled = scaler.transform(features)
    
    scaled_df = pd.DataFrame(
        scaled,
        index=features.index,
        columns=features.columns
    )
    
    return scaled_df, scaler


def create_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int = 60,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        features: Feature DataFrame
        target: Target series
        sequence_length: Length of input sequences
        forecast_horizon: Steps ahead to predict
        
    Returns:
        Tuple of (X sequences, y targets)
    """
    X, y = [], []
    
    for i in range(len(features) - sequence_length - forecast_horizon + 1):
        X.append(features.iloc[i:i + sequence_length].values)
        y.append(target.iloc[i + sequence_length + forecast_horizon - 1])
    
    return np.array(X), np.array(y)


def create_multi_horizon_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int = 60,
    forecast_horizons: list = None
) -> Tuple[np.ndarray, dict]:
    """
    Create sequences for multiple forecast horizons.
    
    Args:
        features: Feature DataFrame
        target: Target series
        sequence_length: Length of input sequences
        forecast_horizons: List of horizons to predict (default: [1, 5, 10, 20])
        
    Returns:
        Tuple of (X sequences, dict of y targets by horizon)
    """
    if forecast_horizons is None:
        forecast_horizons = [1, 5, 10, 20]
    
    max_horizon = max(forecast_horizons)
    X = []
    y_dict = {h: [] for h in forecast_horizons}
    
    for i in range(len(features) - sequence_length - max_horizon + 1):
        X.append(features.iloc[i:i + sequence_length].values)
        
        for horizon in forecast_horizons:
            y_dict[horizon].append(
                target.iloc[i + sequence_length + horizon - 1]
            )
    
    X = np.array(X)
    y_dict = {h: np.array(y_dict[h]) for h in forecast_horizons}
    
    return X, y_dict


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'forward_fill'
) -> pd.DataFrame:
    """
    Handle missing values in feature DataFrame.
    
    Args:
        df: Feature DataFrame
        method: 'forward_fill', 'backward_fill', 'drop', or 'interpolate'
        
    Returns:
        DataFrame with missing values handled
    """
    if method == 'forward_fill':
        return df.fillna(method='ffill')
    elif method == 'backward_fill':
        return df.fillna(method='bfill')
    elif method == 'drop':
        return df.dropna()
    elif method == 'interpolate':
        return df.interpolate(method='linear')
    else:
        raise ValueError(f"Unknown method: {method}")

