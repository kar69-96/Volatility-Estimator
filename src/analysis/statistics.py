"""
Statistical analysis utilities for volatility comparison.

Includes correlation matrices, MSE calculations, and summary statistics.
"""

from typing import Dict

import numpy as np
import pandas as pd


def calculate_correlation_matrix(volatilities: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Calculate correlation matrix between different volatility estimates.
    
    Args:
        volatilities: Dictionary of {method_name: volatility_series}
        
    Returns:
        Correlation matrix DataFrame
    """
    df = pd.DataFrame(volatilities)
    return df.corr()


def calculate_mse_matrix(
    volatilities: Dict[str, pd.Series],
    reference: str = None
) -> pd.DataFrame:
    """
    Calculate Mean Squared Error matrix between volatility estimates.
    
    Args:
        volatilities: Dictionary of {method_name: volatility_series}
        reference: Reference method name (if None, compare all pairs)
        
    Returns:
        MSE matrix DataFrame
    """
    methods = list(volatilities.keys())
    n = len(methods)
    
    if reference:
        # MSE against reference
        mse_dict = {}
        ref_series = volatilities[reference]
        
        for method in methods:
            if method != reference:
                series = volatilities[method]
                aligned_ref, aligned_series = ref_series.align(series, join='inner')
                mse = ((aligned_ref - aligned_series) ** 2).mean()
                mse_dict[method] = mse
        
        return pd.Series(mse_dict).to_frame('MSE')
    else:
        # Pairwise MSE
        mse_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    mse_matrix.loc[method1, method2] = 0.0
                else:
                    series1 = volatilities[method1]
                    series2 = volatilities[method2]
                    aligned1, aligned2 = series1.align(series2, join='inner')
                    mse = ((aligned1 - aligned2) ** 2).mean()
                    mse_matrix.loc[method1, method2] = mse
        
        return mse_matrix


def calculate_summary_statistics(volatility: pd.Series) -> Dict:
    """
    Calculate comprehensive summary statistics for a volatility series.
    
    Args:
        volatility: Volatility series
        
    Returns:
        Dictionary of statistics
    """
    clean = volatility.dropna()
    
    if len(clean) == 0:
        return {}
    
    return {
        'mean': clean.mean(),
        'median': clean.median(),
        'std': clean.std(),
        'min': clean.min(),
        'max': clean.max(),
        'q25': clean.quantile(0.25),
        'q75': clean.quantile(0.75),
        'skewness': clean.skew() if len(clean) > 2 else np.nan,
        'kurtosis': clean.kurtosis() if len(clean) > 2 else np.nan,
    }


def generate_comparison_statistics(
    volatilities: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Generate summary statistics for multiple volatility estimates.
    
    Args:
        volatilities: Dictionary of {method_name: volatility_series}
        
    Returns:
        DataFrame with statistics for each method
    """
    stats_dict = {}
    
    for method, series in volatilities.items():
        stats_dict[method] = calculate_summary_statistics(series)
    
    return pd.DataFrame(stats_dict).T

