"""
Evaluation Metrics for Deep Learning Models.

Provides metrics for both regression (volatility prediction)
and classification (Fed rate prediction) tasks.
"""

from typing import Dict, List, Optional

import numpy as np


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = '',
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    - R²: Coefficient of determination
    - Correlation: Pearson correlation coefficient
    
    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of metric values
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {f'{prefix}mae': np.nan, f'{prefix}rmse': np.nan}
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAPE (avoid division by zero)
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Correlation
    if len(y_true) > 1:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = np.nan
    
    return {
        f'{prefix}mae': float(mae),
        f'{prefix}mse': float(mse),
        f'{prefix}rmse': float(rmse),
        f'{prefix}mape': float(mape) if not np.isnan(mape) else None,
        f'{prefix}r2': float(r2) if not np.isnan(r2) else None,
        f'{prefix}correlation': float(corr) if not np.isnan(corr) else None,
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[str] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Metrics:
    - Accuracy
    - Precision (per class and macro)
    - Recall (per class and macro)
    - F1 Score (per class and macro)
    - Confusion matrix
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_labels: Names of classes
        
    Returns:
        Dictionary of metric values
    """
    from collections import Counter
    
    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)
    
    if len(y_true) == 0:
        return {'accuracy': np.nan}
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for c in classes:
        true_positives = np.sum((y_true == c) & (y_pred == c))
        false_positives = np.sum((y_true != c) & (y_pred == c))
        false_negatives = np.sum((y_true == c) & (y_pred != c))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    # Macro averages
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    
    result = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'confusion_matrix': confusion.tolist(),
    }
    
    # Add per-class metrics
    if class_labels is None:
        class_labels = [f'class_{c}' for c in classes]
    
    for i, label in enumerate(class_labels[:n_classes]):
        result[f'precision_{label}'] = float(precision_per_class[i])
        result[f'recall_{label}'] = float(recall_per_class[i])
        result[f'f1_{label}'] = float(f1_per_class[i])
    
    return result


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate directional accuracy for volatility changes.
    
    Measures how often the model correctly predicts whether
    volatility will increase or decrease.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy (0-1)
    """
    # Calculate changes
    true_changes = np.diff(y_true)
    pred_changes = np.diff(y_pred)
    
    # Compare directions
    true_direction = np.sign(true_changes)
    pred_direction = np.sign(pred_changes)
    
    return float(np.mean(true_direction == pred_direction))


def calculate_hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate hit rate for volatility predictions.
    
    A 'hit' is when the prediction error is within the threshold
    percentage of the true value.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Acceptable percentage error (e.g., 0.1 for 10%)
        
    Returns:
        Dictionary with hit rate metrics
    """
    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'hit_rate': np.nan}
    
    # Calculate percentage errors
    pct_errors = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
    
    # Calculate hit rates at different thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    result = {}
    for t in thresholds:
        hit_rate = np.mean(pct_errors <= t)
        result[f'hit_rate_{int(t*100)}pct'] = float(hit_rate)
    
    return result


def calculate_volatility_forecast_metrics(
    true_volatility: np.ndarray,
    predicted_volatility: np.ndarray,
    realized_returns: np.ndarray = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive volatility forecast metrics.
    
    Args:
        true_volatility: True realized volatility
        predicted_volatility: Predicted volatility
        realized_returns: Actual returns (optional, for additional metrics)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Standard regression metrics
    regression = calculate_regression_metrics(true_volatility, predicted_volatility)
    metrics.update(regression)
    
    # Hit rate metrics
    hit_rates = calculate_hit_rate(true_volatility, predicted_volatility)
    metrics.update(hit_rates)
    
    # Directional accuracy
    metrics['directional_accuracy'] = calculate_directional_accuracy(
        true_volatility, predicted_volatility
    )
    
    # Mincer-Zarnowitz regression (if realized returns available)
    if realized_returns is not None:
        # Calculate realized squared returns
        realized_squared = realized_returns ** 2
        
        # Simple MZ test: regress realized on predicted
        # Perfect forecast: intercept=0, slope=1
        if len(predicted_volatility) == len(realized_squared):
            try:
                # Use simple OLS
                X = np.column_stack([np.ones_like(predicted_volatility), predicted_volatility])
                y = realized_squared
                
                # OLS: beta = (X'X)^(-1) X'y
                XtX = X.T @ X
                Xty = X.T @ y
                
                beta = np.linalg.solve(XtX, Xty)
                
                metrics['mz_intercept'] = float(beta[0])
                metrics['mz_slope'] = float(beta[1])
            except:
                pass
    
    return metrics


def print_metrics_report(metrics: Dict[str, float], title: str = 'Evaluation Metrics'):
    """
    Print a formatted metrics report.
    
    Args:
        metrics: Dictionary of metrics
        title: Report title
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    for key, value in metrics.items():
        if value is None:
            print(f"  {key}: N/A")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            print(f"  {key}:")
            for row in value:
                print(f"    {row}")
        else:
            print(f"  {key}: {value}")
    
    print(f"{'='*50}\n")

