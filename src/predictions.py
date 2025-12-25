"""
Volatility prediction module using historical pattern learning.

This module predicts future volatility based on historical patterns
around similar events.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.event_analysis import analyze_event_impact
from src.utils import parse_date


def classify_volatility_regime(
    volatility_series: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Classify volatility into low/medium/high regimes using rolling percentiles.

    Classification:
    - Low: bottom tertile (33rd percentile)
    - Medium: middle tertile
    - High: top tertile (67th percentile)

    Args:
        volatility_series: Series of volatility values
        window: Rolling window size for percentile calculation

    Returns:
        Series with regime labels: 'low', 'medium', 'high'
    """
    # Calculate rolling percentiles
    rolling_33 = volatility_series.rolling(window=window, min_periods=window).quantile(0.33)
    rolling_67 = volatility_series.rolling(window=window, min_periods=window).quantile(0.67)

    # Classify
    regime = pd.Series(index=volatility_series.index, dtype=str)

    for i in range(len(volatility_series)):
        if pd.isna(rolling_33.iloc[i]) or pd.isna(rolling_67.iloc[i]):
            regime.iloc[i] = np.nan
        else:
            vol = volatility_series.iloc[i]
            if pd.isna(vol):
                regime.iloc[i] = np.nan
            elif vol <= rolling_33.iloc[i]:
                regime.iloc[i] = 'low'
            elif vol <= rolling_67.iloc[i]:
                regime.iloc[i] = 'medium'
            else:
                regime.iloc[i] = 'high'

    return regime


def find_similar_events(
    current_event: pd.Series,
    historical_events: pd.DataFrame,
    volatility_history: pd.Series,
    volatility_dates: pd.Series,
    weights: Optional[dict] = None,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Find similar historical events based on matching criteria.

    Matching criteria:
    - Event type (exact match)
    - Pre-event volatility regime (same regime)
    - Market condition (optional)

    Args:
        current_event: Current event (Series with event_type, date, etc.)
        historical_events: DataFrame of historical events
        volatility_history: Historical volatility series
        volatility_dates: Dates corresponding to volatility series
        weights: Dictionary with weights for matching criteria
                 Default: event_type=0.4, volatility_regime=0.3, market_condition=0.3
        top_n: Number of top similar events to return

    Returns:
        DataFrame of similar events sorted by similarity score
    """
    if weights is None:
        weights = {
            'event_type': 0.4,
            'volatility_regime': 0.3,
            'market_condition': 0.3
        }

    # Classify volatility regimes
    regimes = classify_volatility_regime(volatility_history)

    # Get current event's pre-event regime
    current_date = parse_date(current_event['date'])
    current_idx = None
    for i, date in enumerate(volatility_dates):
        if pd.to_datetime(date).date() == current_date.date():
            current_idx = i
            break

    if current_idx is None:
        return pd.DataFrame()  # Current event not in data

    current_regime = regimes.iloc[current_idx] if current_idx < len(regimes) else np.nan

    # Calculate similarity scores
    similarities = []

    for _, hist_event in historical_events.iterrows():
        hist_date = parse_date(hist_event['date'])

        # Skip if event is after current event (can't use future data)
        if hist_date >= current_date:
            continue

        # Find historical event index
        hist_idx = None
        for i, date in enumerate(volatility_dates):
            if pd.to_datetime(date).date() == hist_date.date():
                hist_idx = i
                break

        if hist_idx is None:
            continue

        hist_regime = regimes.iloc[hist_idx] if hist_idx < len(regimes) else np.nan

        # Calculate similarity score
        score = 0.0

        # Event type match
        if hist_event['event_type'] == current_event['event_type']:
            score += weights['event_type']

        # Volatility regime match
        if not pd.isna(current_regime) and not pd.isna(hist_regime):
            if hist_regime == current_regime:
                score += weights['volatility_regime']

        # Market condition (simplified: use volatility level)
        # This is a placeholder - could be enhanced with trend detection
        if not pd.isna(volatility_history.iloc[hist_idx]) and not pd.isna(volatility_history.iloc[current_idx]):
            vol_diff = abs(volatility_history.iloc[hist_idx] - volatility_history.iloc[current_idx])
            vol_range = volatility_history.max() - volatility_history.min()
            if vol_range > 0:
                similarity = 1.0 - min(vol_diff / vol_range, 1.0)
                score += weights.get('market_condition', 0.3) * similarity

        similarities.append({
            'event_date': hist_date,
            'event_type': hist_event['event_type'],
            'description': hist_event.get('description', ''),
            'similarity_score': score,
            'volatility_regime': hist_regime
        })

    # Sort by similarity and return top N
    if not similarities:
        return pd.DataFrame()

    similar_df = pd.DataFrame(similarities)
    similar_df = similar_df.sort_values('similarity_score', ascending=False).head(top_n)

    return similar_df


def build_pattern_database(
    events_df: pd.DataFrame,
    volatility_series: pd.Series,
    volatility_dates: pd.Series,
    pre_window: int = 5,
    post_window: int = 5
) -> dict:
    """
    Build database of historical event-volatility patterns.

    For each historical event:
    - Store: event_type, pre_vol_regime, volatility_path (array)

    Args:
        events_df: DataFrame of historical events
        volatility_series: Historical volatility series
        volatility_dates: Dates corresponding to volatility series
        pre_window: Days before event
        post_window: Days after event

    Returns:
        Dictionary mapping event indices to pattern data
    """
    patterns = {}

    # Classify regimes
    regimes = classify_volatility_regime(volatility_series)

    for idx, event_row in events_df.iterrows():
        event_date = parse_date(event_row['date'])

        # Find event index in volatility data
        event_idx = None
        for i, date in enumerate(volatility_dates):
            if pd.to_datetime(date).date() == event_date.date():
                event_idx = i
                break

        if event_idx is None:
            continue

        # Extract volatility path around event
        path_start = max(0, event_idx - pre_window)
        path_end = min(len(volatility_series), event_idx + 1 + post_window)

        volatility_path = volatility_series.iloc[path_start:path_end].values
        pre_vol_regime = regimes.iloc[event_idx] if event_idx < len(regimes) else np.nan

        patterns[idx] = {
            'event_type': event_row['event_type'],
            'event_date': event_date,
            'pre_vol_regime': pre_vol_regime,
            'volatility_path': volatility_path,
            'event_idx': event_idx,
            'pre_window': pre_window,
            'post_window': post_window
        }

    return patterns


def predict_volatility_path(
    event_date: pd.Timestamp,
    pattern_database: dict,
    volatility_series: pd.Series,
    volatility_dates: pd.Series,
    similar_events: pd.DataFrame,
    lookback_window: int = 10,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Predict volatility path for an upcoming event.

    Process:
    1. Find similar historical events
    2. Extract volatility trajectories (pre-event → post-event) for each
    3. Compute average path: σ_predicted(t) = mean(σ_historical_similar(t))
    4. Calculate confidence intervals: ±1.96 * std (95% confidence)
    5. Weight recent events more heavily (exponential decay)

    Args:
        event_date: Date of the event to predict
        pattern_database: Dictionary of historical patterns
        volatility_series: Historical volatility series
        volatility_dates: Dates corresponding to volatility series
        similar_events: DataFrame of similar historical events
        lookback_window: Number of days to predict
        confidence_level: Confidence level for bands (default: 0.95)

    Returns:
        DataFrame with columns: date, predicted, lower_bound, upper_bound
    """
    if len(similar_events) == 0:
        # No similar events found
        return pd.DataFrame(columns=['date', 'predicted', 'lower_bound', 'upper_bound'])

    # Find event index
    event_idx = None
    for i, date in enumerate(volatility_dates):
        if pd.to_datetime(date).date() == event_date.date():
            event_idx = i
            break

    if event_idx is None:
        return pd.DataFrame(columns=['date', 'predicted', 'lower_bound', 'upper_bound'])

    # Extract volatility paths from similar events
    paths = []
    weights = []

    for _, similar_event in similar_events.iterrows():
        hist_date = similar_event['event_date']

        # Find pattern in database
        pattern = None
        for pat_idx, pat_data in pattern_database.items():
            if pat_data['event_date'].date() == hist_date.date():
                pattern = pat_data
                break

        if pattern is None:
            continue

        # Extract post-event path (skip pre-event part)
        vol_path = pattern['volatility_path']
        post_window = pattern['post_window']
        post_path = vol_path[-post_window:] if len(vol_path) > post_window else vol_path

        if len(post_path) > 0:
            paths.append(post_path)
            # Weight by similarity score and recency
            similarity = similar_event['similarity_score']
            days_ago = (event_date.date() - hist_date.date()).days
            recency_weight = np.exp(-days_ago / 365.0)  # Exponential decay
            weights.append(similarity * recency_weight)

    if len(paths) == 0:
        return pd.DataFrame(columns=['date', 'predicted', 'lower_bound', 'upper_bound'])

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)

    # Find maximum path length
    max_len = min(max(len(p) for p in paths), lookback_window)

    # Calculate weighted average and confidence intervals
    predicted = []
    lower_bounds = []
    upper_bounds = []

    # Z-score for confidence level
    z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645

    for t in range(max_len):
        # Collect values at time t from all paths
        values_at_t = []
        weights_at_t = []

        for i, path in enumerate(paths):
            if t < len(path):
                if not pd.isna(path[t]):
                    values_at_t.append(path[t])
                    weights_at_t.append(weights[i])

        if len(values_at_t) == 0:
            predicted.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
        else:
            values_at_t = np.array(values_at_t)
            weights_at_t = np.array(weights_at_t)
            weights_at_t = weights_at_t / weights_at_t.sum()

            # Weighted mean
            mean_val = np.average(values_at_t, weights=weights_at_t)
            predicted.append(mean_val)

            # Weighted standard deviation
            variance = np.average((values_at_t - mean_val) ** 2, weights=weights_at_t)
            std_val = np.sqrt(variance)

            # Confidence bands
            lower_bounds.append(mean_val - z_score * std_val)
            upper_bounds.append(mean_val + z_score * std_val)

    # Generate dates
    event_date_dt = pd.to_datetime(event_date)
    dates = pd.date_range(start=event_date_dt, periods=max_len, freq='D')

    return pd.DataFrame({
        'date': dates.date,
        'predicted': predicted,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    })


def backtest_predictions(
    events_df: pd.DataFrame,
    volatility_series: pd.Series,
    volatility_dates: pd.Series,
    pattern_database: dict,
    pre_window: int = 5,
    post_window: int = 5,
    lookback_window: int = 10
) -> dict:
    """
    Backtest prediction accuracy on historical events.

    For each historical event:
    1. Use only data BEFORE event date
    2. Generate prediction
    3. Compare predicted vs actual volatility

    Args:
        events_df: DataFrame of historical events
        volatility_series: Full historical volatility series
        volatility_dates: Dates corresponding to volatility series
        pattern_database: Dictionary of historical patterns
        pre_window: Days before event
        post_window: Days after event
        lookback_window: Prediction horizon

    Returns:
        Dictionary with accuracy metrics:
        - prediction_mae: Mean Absolute Error
        - prediction_rmse: Root Mean Squared Error
        - prediction_correlation: Correlation coefficient
        - total_predictions: Number of events backtested
        - backtest_results: DataFrame with detailed results
    """
    backtest_results = []

    for idx, event_row in events_df.iterrows():
        event_date = parse_date(event_row['date'])

        # Find event index
        event_idx = None
        for i, date in enumerate(volatility_dates):
            if pd.to_datetime(date).date() == event_date.date():
                event_idx = i
                break

        if event_idx is None:
            continue

        # Use only data before event
        historical_vol = volatility_series.iloc[:event_idx]
        historical_dates = volatility_dates.iloc[:event_idx]

        if len(historical_vol) < 60:  # Need sufficient history
            continue

        # Build pattern database from historical data only
        hist_events = events_df[events_df['date'] < event_date].copy()
        if len(hist_events) == 0:
            continue

        hist_pattern_db = build_pattern_database(
            hist_events, historical_vol, historical_dates, pre_window, post_window
        )

        if len(hist_pattern_db) == 0:
            continue

        # Find similar events
        similar = find_similar_events(
            event_row, hist_events, historical_vol, historical_dates
        )

        if len(similar) == 0:
            continue

        # Generate prediction
        prediction = predict_volatility_path(
            event_date, hist_pattern_db, historical_vol, historical_dates,
            similar, lookback_window
        )

        if len(prediction) == 0:
            continue

        # Get actual volatility
        actual_start = event_idx + 1
        actual_end = min(len(volatility_series), event_idx + 1 + post_window)
        actual_vol = volatility_series.iloc[actual_start:actual_end].values

        # Align prediction and actual
        min_len = min(len(prediction), len(actual_vol))
        if min_len == 0:
            continue

        pred_values = prediction['predicted'].iloc[:min_len].values
        actual_values = actual_vol[:min_len]

        # Remove NaN values
        valid_mask = ~(pd.isna(pred_values) | pd.isna(actual_values))
        if valid_mask.sum() == 0:
            continue

        pred_valid = pred_values[valid_mask]
        actual_valid = actual_values[valid_mask]

        # Calculate metrics
        errors = pred_valid - actual_valid
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        correlation = np.corrcoef(pred_valid, actual_valid)[0, 1] if len(pred_valid) > 1 else np.nan

        backtest_results.append({
            'event_date': event_date,
            'event_type': event_row['event_type'],
            'predicted_mean': float(np.mean(pred_valid)) if len(pred_valid) > 0 else np.nan,
            'actual_mean': float(np.mean(actual_valid)) if len(actual_valid) > 0 else np.nan,
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation) if not pd.isna(correlation) else np.nan,
            'n_points': int(len(pred_valid))
        })

    if len(backtest_results) == 0:
        return {
            'prediction_mae': np.nan,
            'prediction_rmse': np.nan,
            'prediction_correlation': np.nan,
            'total_predictions': 0,
            'backtest_results': pd.DataFrame()
        }

    results_df = pd.DataFrame(backtest_results)

    return {
        'prediction_mae': float(results_df['mae'].mean()),
        'prediction_rmse': float(results_df['rmse'].mean()),
        'prediction_correlation': float(results_df['correlation'].mean()),
        'total_predictions': len(results_df),
        'backtest_results': results_df
    }

