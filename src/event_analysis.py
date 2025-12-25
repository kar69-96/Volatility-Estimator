"""
Event-driven volatility analysis module.

This module analyzes how economic events impact volatility, measuring
pre-event vs post-event volatility changes and persistence.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import DataError, parse_date


def analyze_event_impact(
    volatility_series: pd.Series,
    volatility_dates: pd.Series,
    event_date: pd.Timestamp,
    pre_window: int = 5,
    post_window: int = 5
) -> dict:
    """
    Analyze the impact of a single event on volatility.

    Calculates:
    - Pre-event average volatility (mean of pre_window days)
    - Post-event average volatility (mean of post_window days)
    - Volatility change: Δσ = (σ_post - σ_pre) / σ_pre
    - Persistence: days until volatility returns to pre-event level
    - Statistical significance: t-test for mean difference (p-value)

    Args:
        volatility_series: Series of volatility values
        volatility_dates: Series of dates corresponding to volatility values
        event_date: Date of the event (pd.Timestamp or date)
        pre_window: Number of days before event to analyze
        post_window: Number of days after event to analyze

    Returns:
        Dictionary with analysis metrics:
        - pre_event_volatility: Average volatility before event
        - post_event_volatility: Average volatility after event
        - volatility_change: Absolute change
        - volatility_change_pct: Percentage change
        - persistence_days: Days until return to baseline
        - statistical_significance: Boolean (p < 0.05)
        - p_value: p-value from t-test
        - t_statistic: t-statistic from t-test
    """
    event_date = parse_date(event_date)

    # Convert dates to datetime for comparison
    if isinstance(volatility_dates.iloc[0], str):
        volatility_dates = pd.to_datetime(volatility_dates)
    elif hasattr(volatility_dates.iloc[0], 'date'):
        volatility_dates = pd.to_datetime(volatility_dates)

    # Find event date index
    event_idx = None
    for i, date in enumerate(volatility_dates):
        if pd.to_datetime(date).date() == event_date.date():
            event_idx = i
            break

    if event_idx is None:
        # Event date not in data, return NaN metrics
        return {
            'pre_event_volatility': np.nan,
            'post_event_volatility': np.nan,
            'volatility_change': np.nan,
            'volatility_change_pct': np.nan,
            'persistence_days': np.nan,
            'statistical_significance': False,
            'p_value': np.nan,
            't_statistic': np.nan
        }

    # Extract pre-event and post-event periods
    pre_start = max(0, event_idx - pre_window)
    pre_end = event_idx
    post_start = event_idx + 1
    post_end = min(len(volatility_series), event_idx + 1 + post_window)

    pre_vol = volatility_series.iloc[pre_start:pre_end].dropna()
    post_vol = volatility_series.iloc[post_start:post_end].dropna()

    if len(pre_vol) == 0 or len(post_vol) == 0:
        # Insufficient data
        return {
            'pre_event_volatility': np.nan,
            'post_event_volatility': np.nan,
            'volatility_change': np.nan,
            'volatility_change_pct': np.nan,
            'persistence_days': np.nan,
            'statistical_significance': False,
            'p_value': np.nan,
            't_statistic': np.nan
        }

    # Calculate averages
    pre_avg = pre_vol.mean()
    post_avg = post_vol.mean()

    # Calculate volatility change
    vol_change = post_avg - pre_avg
    vol_change_pct = (vol_change / pre_avg) * 100 if pre_avg > 0 else np.nan

    # Statistical significance test (t-test)
    if len(pre_vol) >= 2 and len(post_vol) >= 2:
        t_stat, p_value = stats.ttest_ind(post_vol, pre_vol)
        significant = p_value < 0.05
    else:
        t_stat = np.nan
        p_value = np.nan
        significant = False

    # Calculate persistence: days until volatility returns to pre-event level
    persistence_days = calculate_persistence(
        volatility_series.iloc[post_start:],
        pre_avg,
        max_days=30  # Don't look more than 30 days ahead
    )

    return {
        'pre_event_volatility': float(pre_avg),
        'post_event_volatility': float(post_avg),
        'volatility_change': float(vol_change),
        'volatility_change_pct': float(vol_change_pct),
        'persistence_days': int(persistence_days) if not pd.isna(persistence_days) else np.nan,
        'statistical_significance': significant,
        'p_value': float(p_value) if not pd.isna(p_value) else np.nan,
        't_statistic': float(t_stat) if not pd.isna(t_stat) else np.nan
    }


def calculate_persistence(
    volatility_series: pd.Series,
    baseline_volatility: float,
    max_days: int = 30
) -> int:
    """
    Calculate days until volatility returns to baseline level.

    Args:
        volatility_series: Volatility series starting from event date
        baseline_volatility: Pre-event baseline volatility
        max_days: Maximum days to look ahead

    Returns:
        Number of days until return to baseline, or max_days if not reached
    """
    if pd.isna(baseline_volatility) or len(volatility_series) == 0:
        return max_days

    # Look for first day where volatility is within 5% of baseline
    threshold = baseline_volatility * 0.05  # 5% tolerance

    for i in range(min(len(volatility_series), max_days)):
        vol = volatility_series.iloc[i]
        if not pd.isna(vol):
            if abs(vol - baseline_volatility) <= threshold:
                return i + 1

    return max_days


def analyze_all_events(
    volatility_series: pd.Series,
    volatility_dates: pd.Series,
    events_df: pd.DataFrame,
    pre_window: int = 5,
    post_window: int = 5
) -> pd.DataFrame:
    """
    Analyze all events in the events DataFrame.

    Args:
        volatility_series: Series of volatility values
        volatility_dates: Series of dates corresponding to volatility values
        events_df: DataFrame with event information (columns: date, event_type, description, importance)
        pre_window: Number of days before event to analyze
        post_window: Number of days after event to analyze

    Returns:
        DataFrame with analysis results:
        - event_date: Event date
        - event_type: Type of event
        - description: Event description
        - pre_vol: Pre-event volatility
        - post_vol: Post-event volatility
        - volatility_change: Absolute change
        - volatility_change_pct: Percentage change
        - persistence_days: Days until return to baseline
        - significant: Statistical significance flag
    """
    results = []

    for _, event_row in events_df.iterrows():
        event_date = parse_date(event_row['date'])

        analysis = analyze_event_impact(
            volatility_series=volatility_series,
            volatility_dates=volatility_dates,
            event_date=event_date,
            pre_window=pre_window,
            post_window=post_window
        )

        results.append({
            'event_date': event_date,
            'event_type': event_row['event_type'],
            'description': event_row['description'],
            'importance': event_row.get('importance', 'medium'),
            'pre_vol': analysis['pre_event_volatility'],
            'post_vol': analysis['post_event_volatility'],
            'volatility_change': analysis['volatility_change'],
            'volatility_change_pct': analysis['volatility_change_pct'],
            'persistence_days': analysis['persistence_days'],
            'significant': analysis['statistical_significance'],
            'p_value': analysis['p_value'],
            't_statistic': analysis['t_statistic']
        })

    return pd.DataFrame(results)

