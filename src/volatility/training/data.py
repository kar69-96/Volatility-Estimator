"""
Data preparation for Chronos volatility prediction.

CRITICAL: Use raw squared returns (single channel) as input to Chronos.
No feature engineering.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, ConcatDataset



def prepare_raw_signal(df):
    """
    Prepare raw signal: squared returns (single channel).
    
    Chronos was pretrained on raw time series, not engineered features.
    Using squared returns preserves volatility structure while keeping it raw.
    
    Args:
        df: DataFrame with OHLC data (must have 'close' column)
        
    Returns:
        Series of squared returns with same index as input (after dropping first NaN)
    """
    returns = np.log(df['close'] / df['close'].shift(1))
    squared_returns = returns ** 2
    return squared_returns.dropna()


def compute_target(returns, horizon=20):
    """
    Compute log-realized variance target: log(Σr²).
    
    CRITICAL TIMING: Predict variance over [t+1, t+h] using data up to time t.
    Target at index t corresponds to realized variance over [t+1, t+h].
    
    Args:
        returns: Series of returns
        horizon: Prediction horizon in days
        
    Returns:
        Series of log-realized variance targets
    """
    squared_returns = returns ** 2
    # Realized variance: sum of squared returns over next h days
    realized_variance = squared_returns.rolling(window=horizon).sum()
    # Shift backward by h: at time t, we predict variance for [t+1, t+h]
    # So target[t] = log(Σr² from t+1 to t+h)
    # Use larger epsilon and clip to prevent log(0) and extreme values
    target = np.log(np.clip(realized_variance.shift(-horizon) + 1e-6, a_min=1e-6, a_max=1e6))
    # Replace any remaining NaN/inf with a reasonable default (log of small variance)
    target = target.replace([np.inf, -np.inf], np.log(1e-6))
    target = target.fillna(np.log(1e-6))
    return target


class VolatilityDataset(Dataset):
    """
    Dataset for volatility prediction.
    
    CRITICAL: Ensures correct timing alignment.
    - Input at index idx: data from [idx, idx+seq_length-1]
    - Target at index idx: realized variance over [idx+seq_length, idx+seq_length+horizon-1]
    """
    def __init__(self, raw_signal, target, seq_length=252, horizon=20, use_recent_weighting=True):
        """
        Args:
            raw_signal: Squared returns (single-channel time series)
            target: Log-realized variance targets
            seq_length: Input sequence length (default: 252 trading days = 1 year)
            horizon: Prediction horizon (must match target computation)
            use_recent_weighting: If True, compute sample weights favoring recent data
        """
        self.raw_signal = raw_signal.values if isinstance(raw_signal, pd.Series) else raw_signal
        self.target = target.values if isinstance(target, pd.Series) else target
        self.seq_length = seq_length
        self.horizon = horizon
        self.use_recent_weighting = use_recent_weighting
        
        # Valid indices: need seq_length input + horizon for target
        self.valid_len = len(self.target) - seq_length - horizon + 1
        
        # Compute sample weights if requested (favor more recent samples)
        if use_recent_weighting and self.valid_len > 0:
            self.sample_weights = self._compute_sample_weights()
        else:
            self.sample_weights = None
    
    def _compute_sample_weights(self):
        """
        Compute sample weights that favor more recent data.
        
        Uses exponential weighting: w[i] = exp(alpha * (i - max_idx) / max_idx)
        where alpha controls the strength of recent bias (higher = more bias toward recent).
        
        Returns:
            Array of weights normalized to sum to len(weights)
        """
        if self.valid_len <= 0:
            return None
        
        # Use exponential weighting: more recent samples get higher weights
        # alpha controls the strength: 2.0 means recent samples get ~7x more weight than oldest
        alpha = 2.0
        indices = np.arange(self.valid_len)
        
        # Normalize indices to [0, 1] where 0 is oldest, 1 is newest
        normalized = indices / max(self.valid_len - 1, 1)
        
        # Exponential weighting: exp(alpha * normalized) gives more weight to recent
        weights = np.exp(alpha * normalized)
        
        # Normalize so sum equals length (for WeightedRandomSampler compatibility)
        weights = weights / weights.sum() * len(weights)
        
        return weights.astype(np.float32)
        
    def get_sample_weights(self):
        """
        Get sample weights for WeightedRandomSampler.
        
        Returns:
            Array of weights, or None if not using weighting
        """
        return self.sample_weights
        
    def __len__(self):
        return max(0, self.valid_len)
        
    def __getitem__(self, idx):
        """
        Get sample at index idx.
        
        Input: raw signal from [idx, idx+seq_length-1]
        Target: log-realized variance for [idx+seq_length, idx+seq_length+horizon-1]
        """
        if idx >= self.valid_len:
            raise IndexError(f"Index {idx} out of range")
        
        # Input sequence (raw squared returns)
        seq_end = idx + self.seq_length
        raw_seq = self.raw_signal[idx:seq_end].astype(np.float32)
        
        # Replace NaN/inf with small positive value
        raw_seq = np.nan_to_num(raw_seq, nan=1e-8, posinf=1.0, neginf=1e-8)
        raw_seq = np.clip(raw_seq, a_min=0.0, a_max=1e6)  # Clip extreme values
        
        # Convert to tensor - shape (seq_length,)
        input_tensor = torch.FloatTensor(raw_seq)
        
        # Target: log-realized variance for period starting at seq_end
        target_idx = seq_end  # Target is aligned with end of input sequence
        y = self.target[target_idx]
        
        # Replace NaN/inf in target
        if np.isnan(y) or np.isinf(y):
            y = np.log(1e-6)  # Default to log of small variance
        
        return input_tensor, torch.FloatTensor([y])


def create_weighted_sampler_for_concat_dataset(concat_dataset, batch_size):
    """
    Create a WeightedRandomSampler for a ConcatDataset of VolatilityDataset objects.
    
    This combines weights from all individual datasets, giving more weight to recent
    samples across all datasets.
    
    Args:
        concat_dataset: ConcatDataset containing VolatilityDataset objects
        batch_size: Batch size for the sampler
        
    Returns:
        WeightedRandomSampler if weights are available, None otherwise
    """
    if not isinstance(concat_dataset, ConcatDataset):
        # Single dataset - use its weights directly
        if hasattr(concat_dataset, 'get_sample_weights'):
            weights = concat_dataset.get_sample_weights()
            if weights is not None:
                return WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(concat_dataset),
                    replacement=True
                )
        return None
    
    # For ConcatDataset, we need to combine weights from all datasets
    all_weights = []
    dataset_lengths = []
    
    for dataset in concat_dataset.datasets:
        if hasattr(dataset, 'get_sample_weights'):
            weights = dataset.get_sample_weights()
            if weights is not None:
                all_weights.append(weights)
                dataset_lengths.append(len(dataset))
            else:
                # If a dataset doesn't have weights, use uniform weights
                uniform_weights = np.ones(len(dataset), dtype=np.float32)
                all_weights.append(uniform_weights)
                dataset_lengths.append(len(dataset))
        else:
            # Dataset doesn't support weighting, use uniform
            uniform_weights = np.ones(len(dataset), dtype=np.float32)
            all_weights.append(uniform_weights)
            dataset_lengths.append(len(dataset))
    
    if not all_weights:
        return None
    
    # Concatenate all weights
    combined_weights = np.concatenate(all_weights)
    
    # Create sampler
    return WeightedRandomSampler(
        weights=torch.FloatTensor(combined_weights),
        num_samples=len(concat_dataset),
        replacement=True
    )

