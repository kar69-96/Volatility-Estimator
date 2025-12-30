"""
Risk and performance metrics for portfolio analysis.

Includes Sharpe ratio, Treynor ratio, Beta, Maximum Drawdown, and VaR.
"""

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate (decimal)
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if excess_returns.std() == 0:
        return np.nan
    
    sharpe = excess_returns.mean() / excess_returns.std()
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)
    
    return sharpe_annualized


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate (decimal)
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.nan
    
    sortino = excess_returns.mean() / downside_returns.std()
    sortino_annualized = sortino * np.sqrt(periods_per_year)
    
    return sortino_annualized


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """
    Calculate beta (systematic risk).
    
    Args:
        asset_returns: Asset return series
        market_returns: Market return series
        
    Returns:
        Beta coefficient
    """
    # Align series
    aligned_asset, aligned_market = asset_returns.align(market_returns, join='inner')
    
    if len(aligned_asset) < 2:
        return np.nan
    
    covariance = aligned_asset.cov(aligned_market)
    market_variance = aligned_market.var()
    
    if market_variance == 0:
        return np.nan
    
    return covariance / market_variance


def calculate_treynor_ratio(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Treynor ratio (return per unit of systematic risk).
    
    Args:
        returns: Asset return series
        market_returns: Market return series
        risk_free_rate: Annual risk-free rate (decimal)
        periods_per_year: Number of periods per year
        
    Returns:
        Treynor ratio
    """
    beta = calculate_beta(returns, market_returns)
    
    if np.isnan(beta) or beta == 0:
        return np.nan
    
    excess_return = returns.mean() - (risk_free_rate / periods_per_year)
    treynor = (excess_return * periods_per_year) / beta
    
    return treynor


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Maximum drawdown (as negative percentage)
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.
    
    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR (as negative value)
    """
    return returns.quantile(1 - confidence_level)


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
    
    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaR (as negative value)
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_all_risk_metrics(
    returns: pd.Series,
    market_returns: pd.Series = None,
    prices: pd.Series = None,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate all risk metrics for an asset.
    
    Args:
        returns: Asset return series
        market_returns: Market return series (optional, for beta/Treynor)
        prices: Price series (optional, for max drawdown)
        risk_free_rate: Annual risk-free rate (decimal)
        
    Returns:
        Dictionary of risk metrics
    """
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'var_95': calculate_var(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
    }
    
    if market_returns is not None:
        metrics['beta'] = calculate_beta(returns, market_returns)
        metrics['treynor_ratio'] = calculate_treynor_ratio(
            returns, market_returns, risk_free_rate
        )
    
    if prices is not None:
        metrics['max_drawdown'] = calculate_max_drawdown(prices)
    
    return metrics

