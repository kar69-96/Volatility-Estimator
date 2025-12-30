"""
Simple inference for Chronos volatility prediction.

Predicts volatility with uncertainty intervals using raw squared returns.
"""

import numpy as np
import torch


def predict(model, raw_signal, device='cuda'):
    """
    Predict volatility with uncertainty intervals.
    
    Args:
        model: Trained ChronosVolatility model
        raw_signal: Squared returns (single-channel time series, last 60 days)
                   Can be numpy array, list, or torch tensor
        device: Device for inference
    
    Returns:
        Dictionary with volatility prediction and intervals
    """
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(raw_signal, (list, np.ndarray)):
        raw_signal = torch.FloatTensor(raw_signal)
    
    # Ensure correct shape: (1, seq_len) for batch inference
    if raw_signal.dim() == 1:
        raw_signal = raw_signal.unsqueeze(0)
    
    # Move to device
    raw_signal = raw_signal.to(device)
    
    with torch.no_grad():
        # Forward pass
        quantiles = model(raw_signal)  # (1, 3): [q10, q50, q90]
        q10, q50, q90 = quantiles[0].cpu().numpy()
    
    # Convert log-variance to variance
    var_q10 = np.exp(q10)
    var_q50 = np.exp(q50)
    var_q90 = np.exp(q90)
    
    # Convert to volatility (annualized)
    # Assuming daily variance, multiply by 252 trading days
    vol_q10 = np.sqrt(var_q10 * 252) * 100  # Lower bound (q10 variance) as percentage
    vol_q50 = np.sqrt(var_q50 * 252) * 100  # Point estimate (q50 variance) as percentage
    vol_q90 = np.sqrt(var_q90 * 252) * 100  # Upper bound (q90 variance) as percentage
    
    return {
        'volatility': vol_q50,
        'lower': vol_q10,
        'upper': vol_q90,
        'log_variance_q50': q50,
        'log_variance_q10': q10,
        'log_variance_q90': q90
    }

