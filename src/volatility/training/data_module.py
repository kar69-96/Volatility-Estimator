"""
Data Module for Deep Learning Training.

Provides PyTorch DataLoader wrappers and data preprocessing
for time series volatility prediction.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Lazy imports
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class TimeSeriesDataset:
    """
    PyTorch Dataset for time series data.
    
    Creates sequences of features and corresponding targets.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int = 252,
        prediction_horizon: int = 1,
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,) or (n_samples, n_horizons)
            seq_length: Length of input sequences
            prediction_horizon: How far ahead to predict
        """
        torch = _get_torch()
        
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        
        # Calculate valid indices
        self.n_samples = len(features) - seq_length - prediction_horizon + 1
        
    def __len__(self) -> int:
        return max(0, self.n_samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        torch = _get_torch()
        
        # Get sequence
        seq_start = idx
        seq_end = idx + self.seq_length
        
        X = self.features[seq_start:seq_end]
        
        # Get target(s)
        target_idx = seq_end + self.prediction_horizon - 1
        
        if self.targets.ndim == 1:
            y = self.targets[target_idx]
        else:
            y = self.targets[target_idx]
        
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class MultiHorizonDataset:
    """
    Dataset for multi-horizon prediction.
    
    Returns targets at multiple prediction horizons.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int = 252,
        horizons: List[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
            seq_length: Length of input sequences
            horizons: List of prediction horizons
        """
        torch = _get_torch()
        
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.horizons = horizons or [1, 5, 10, 20]
        
        self.max_horizon = max(self.horizons)
        self.n_samples = len(features) - seq_length - self.max_horizon + 1
        
    def __len__(self) -> int:
        return max(0, self.n_samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        torch = _get_torch()
        
        seq_start = idx
        seq_end = idx + self.seq_length
        
        X = self.features[seq_start:seq_end]
        
        # Get targets at each horizon
        y = []
        for h in self.horizons:
            target_idx = seq_end + h - 1
            y.append(self.targets[target_idx])
        
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class VolatilityDataModule:
    """
    Data module for volatility prediction.
    
    Handles data loading, preprocessing, and train/val/test splitting.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = 'realized_vol_20d',
        seq_length: int = 252,
        prediction_horizons: List[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """
        Initialize data module.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            seq_length: Input sequence length
            prediction_horizons: List of prediction horizons
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
        """
        self.df = df
        self.target_column = target_column
        self.seq_length = seq_length
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 20]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for training."""
        # Get feature columns (exclude date and target)
        feature_cols = [
            c for c in self.df.columns 
            if c not in ['date', self.target_column]
        ]
        
        # Convert to numpy arrays
        self.features = self.df[feature_cols].values
        self.targets = self.df[self.target_column].values
        
        # Handle NaN values
        nan_mask = np.isnan(self.features).any(axis=1) | np.isnan(self.targets)
        if nan_mask.any():
            # Forward fill NaN values
            df_temp = pd.DataFrame(self.features)
            df_temp = df_temp.ffill().bfill()
            self.features = df_temp.values
            
            targets_temp = pd.Series(self.targets)
            targets_temp = targets_temp.ffill().bfill()
            self.targets = targets_temp.values
        
        # Split data (time series aware - no shuffling)
        n = len(self.features)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        self.train_features = self.features[:train_end]
        self.train_targets = self.targets[:train_end]
        
        self.val_features = self.features[train_end:val_end]
        self.val_targets = self.targets[train_end:val_end]
        
        self.test_features = self.features[val_end:]
        self.test_targets = self.targets[val_end:]
        
        self.n_features = self.features.shape[1]
    
    def get_train_dataloader(self):
        """Get training DataLoader."""
        torch = _get_torch()
        
        dataset = MultiHorizonDataset(
            self.train_features,
            self.train_targets,
            self.seq_length,
            self.prediction_horizons,
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def get_val_dataloader(self):
        """Get validation DataLoader."""
        torch = _get_torch()
        
        dataset = MultiHorizonDataset(
            self.val_features,
            self.val_targets,
            self.seq_length,
            self.prediction_horizons,
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def get_test_dataloader(self):
        """Get test DataLoader."""
        torch = _get_torch()
        
        dataset = MultiHorizonDataset(
            self.test_features,
            self.test_targets,
            self.seq_length,
            self.prediction_horizons,
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def create_train_val_test_split(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into train/val/test sets.
    
    Maintains temporal order (no shuffling).
    
    Args:
        data: Array to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        
    Returns:
        Tuple of (train, val, test) arrays
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return (
        data[:train_end],
        data[train_end:val_end],
        data[val_end:],
    )


def normalize_features(
    train: np.ndarray,
    val: np.ndarray = None,
    test: np.ndarray = None,
    method: str = 'standard',
) -> Tuple[np.ndarray, ...]:
    """
    Normalize features using training set statistics.
    
    Args:
        train: Training data
        val: Validation data (optional)
        test: Test data (optional)
        method: 'standard' (z-score) or 'minmax' (0-1)
        
    Returns:
        Normalized arrays
    """
    if method == 'standard':
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        train_norm = (train - mean) / std
        
        results = [train_norm]
        
        if val is not None:
            results.append((val - mean) / std)
        if test is not None:
            results.append((test - mean) / std)
            
    else:  # minmax
        min_val = train.min(axis=0)
        max_val = train.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        
        train_norm = (train - min_val) / range_val
        
        results = [train_norm]
        
        if val is not None:
            results.append((val - min_val) / range_val)
        if test is not None:
            results.append((test - min_val) / range_val)
    
    return tuple(results)







