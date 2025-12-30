"""
Evaluation metrics for volatility prediction.

QLIKE and basic metrics.
"""

import numpy as np


def qlike_metric(pred_log_var, true_log_var):
    """
    QLIKE metric: log(σ²_pred/σ²_true) + σ²_true/σ²_pred - 1.
    
    Args:
        pred_log_var: Predicted log-variance
        true_log_var: True log-realized variance
        
    Returns:
        QLIKE metric value
    """
    pred_var = np.exp(pred_log_var)
    true_var = np.exp(true_log_var)
    
    # Avoid division by zero
    eps = 1e-8
    term1 = np.log(pred_var / (true_var + eps) + eps)
    term2 = true_var / (pred_var + eps)
    
    return np.mean(term1 + term2 - 1.0)


def mse_log_variance(pred_log_var, true_log_var):
    """
    MSE in log-variance space.
    
    Args:
        pred_log_var: Predicted log-variance
        true_log_var: True log-realized variance
        
    Returns:
        Mean squared error
    """
    return np.mean((pred_log_var - true_log_var) ** 2)

