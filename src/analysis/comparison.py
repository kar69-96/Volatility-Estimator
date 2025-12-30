"""
Estimator comparison module.

This module provides functionality to compare multiple volatility estimators
side-by-side and generate comparison statistics.
"""

import numpy as np
import pandas as pd

from src.estimators import ESTIMATORS, get_estimator


def run_all_estimators(
    data: pd.DataFrame,
    window: int = 60,
    annualization_factor: int = 252,
    lambda_param: float = 0.94
) -> pd.DataFrame:
    """
    Run all estimators on the same data and return results side-by-side.

    Args:
        data: DataFrame with OHLC data
        window: Rolling window size
        annualization_factor: Days per year
        lambda_param: Lambda parameter for EWMA

    Returns:
        DataFrame with columns: date, close_to_close, ewma, parkinson,
        rogers_satchell, yang_zhang
    """
    results = pd.DataFrame()
    results['date'] = data['date']

    # Run each estimator
    for name in ESTIMATORS.keys():
        try:
            if name == 'ewma':
                estimator = get_estimator(
                    name, window, annualization_factor, lambda_param=lambda_param
                )
            else:
                estimator = get_estimator(name, window, annualization_factor)

            volatility = estimator.compute(data, annualize=True)
            results[name] = volatility

        except Exception as e:
            # If estimator fails, fill with NaN
            results[name] = np.nan
            print(f"Warning: {name} estimator failed: {str(e)}")

    return results


def calculate_correlation_matrix(volatility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix between estimators.

    Args:
        volatility_df: DataFrame with volatility estimates (columns are estimators)

    Returns:
        Correlation matrix DataFrame
    """
    # Remove date column if present
    if 'date' in volatility_df.columns:
        vol_data = volatility_df.drop(columns=['date'])
    else:
        vol_data = volatility_df

    # Calculate correlation
    correlation = vol_data.corr()

    return correlation


def calculate_mse_matrix(volatility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean squared error matrix between estimators.

    Args:
        volatility_df: DataFrame with volatility estimates (columns are estimators)

    Returns:
        MSE matrix DataFrame
    """
    # Remove date column if present
    if 'date' in volatility_df.columns:
        vol_data = volatility_df.drop(columns=['date'])
    else:
        vol_data = volatility_df

    # Calculate MSE for each pair
    estimators = vol_data.columns
    mse_matrix = pd.DataFrame(index=estimators, columns=estimators)

    for est1 in estimators:
        for est2 in estimators:
            if est1 == est2:
                mse_matrix.loc[est1, est2] = 0.0
            else:
                # Calculate MSE
                diff = vol_data[est1] - vol_data[est2]
                mse = (diff ** 2).mean()
                mse_matrix.loc[est1, est2] = mse

    return mse_matrix.astype(float)


def generate_comparison_statistics(volatility_df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for each estimator.

    Args:
        volatility_df: DataFrame with volatility estimates

    Returns:
        Dictionary with statistics for each estimator
    """
    # Remove date column if present
    if 'date' in volatility_df.columns:
        vol_data = volatility_df.drop(columns=['date'])
    else:
        vol_data = volatility_df

    stats = {}
    for estimator in vol_data.columns:
        vol_series = vol_data[estimator].dropna()
        if len(vol_series) > 0:
            stats[estimator] = {
                'mean': float(vol_series.mean()),
                'std': float(vol_series.std()),
                'min': float(vol_series.min()),
                'max': float(vol_series.max()),
                'count': int(len(vol_series))
            }
        else:
            stats[estimator] = {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }

    return stats


