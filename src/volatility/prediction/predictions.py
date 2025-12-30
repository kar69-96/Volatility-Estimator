"""
Volatility prediction module using historical pattern learning and deep learning.

This module predicts future volatility based on:
1. Historical patterns around similar events
2. Deep learning models (Chronos)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import parse_date
from src.utils.volatility_utils import classify_volatility_regime

# Deep learning imports (optional)
_DL_AVAILABLE = False
try:
    from src.volatility.models.chronos import ChronosVolatility
    from src.volatility.models.base_model import get_device
    from src.data.returns import calculate_returns
    import torch
    _DL_AVAILABLE = True
except ImportError:
    pass


def is_deep_learning_available() -> bool:
    """Check if deep learning modules are available."""
    return _DL_AVAILABLE




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


# ============================================================================
# Deep Learning Prediction Functions
# ============================================================================

def train_dl_models(
    df: pd.DataFrame,
    model_type: str = 'volatility',
    save_dir: str = './models/checkpoints',
    epochs: int = 100,
    device: str = 'auto',
    verbose: bool = True,
) -> Dict:
    """
    Train deep learning models.
    
    Args:
        df: DataFrame with OHLC data
        model_type: Model type (not currently used)
        save_dir: Directory to save trained models
        epochs: Number of training epochs
        device: Computing device
        verbose: Print training progress
        
    Returns:
        Dictionary with training results
    """
    if not _DL_AVAILABLE:
        return {
            'error': 'Deep learning not available. Install PyTorch: pip install torch',
            'success': False
        }
    
    return {
        'error': 'Training not yet implemented. Use Chronos model for predictions.',
        'success': False
    }


def load_chronos_from_hf(
    model_id: str = 'karkar69/chronos-volatility',
    device: str = 'auto',
) -> Optional[Any]:
    """
    Load Chronos volatility model from Hugging Face.
    
    Args:
        model_id: Hugging Face model ID (default: 'karkar69/chronos-volatility')
        device: Device for inference ('auto', 'cuda', 'cpu')
    
    Returns:
        Loaded ChronosVolatility model or None if unavailable
    """
    # Try to import required modules for Chronos
    try:
        from src.volatility.models.chronos import ChronosVolatility
        from src.volatility.models.base_model import get_device
        import torch
    except ImportError:
        return None
    
    try:
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM
        from src.volatility.models.chronos import ChronosVolatility
        from src.volatility.models.base_model import get_device
        import torch
        
        # Get device
        torch_device = get_device(device)
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained('amazon/chronos-t5-mini')
        
        # Load PEFT adapter from adapter/ subfolder
        adapter_model = PeftModel.from_pretrained(base_model, model_id, subfolder="adapter")
        
        # Merge adapter weights into base model for inference
        merged_model = adapter_model.merge_and_unload()
        
        # Create ChronosVolatility wrapper - we need to create it without loading base
        # Get hidden dimension from merged model config
        hidden_dim = getattr(merged_model.config, 'd_model', None)
        if hidden_dim is None:
            hidden_dim = getattr(merged_model.config, 'hidden_size', None)
        if hidden_dim is None:
            hidden_dim = 64  # Default fallback
        
        # Create model wrapper manually to avoid reloading base model
        chronos_model = ChronosVolatility.__new__(ChronosVolatility)
        torch.nn.Module.__init__(chronos_model)
        chronos_model.base = merged_model
        chronos_model.model_id = 'amazon/chronos-t5-mini'
        chronos_model.hidden_dim = hidden_dim
        chronos_model.quantile_head = torch.nn.Linear(hidden_dim, 3)
        chronos_model.value_embedding = torch.nn.Linear(1, hidden_dim)
        
        # Load custom heads from heads.pt file
        try:
            from huggingface_hub import hf_hub_download
            import os
            try:
                heads_path = hf_hub_download(
                    repo_id=model_id,
                    filename="heads.pt",
                    cache_dir=None
                )
                if os.path.exists(heads_path):
                    checkpoint = torch.load(heads_path, map_location=torch_device)
                    # The checkpoint uses keys like 'quantile_head.state_dict' and 'value_embedding.state_dict'
                    if 'quantile_head.state_dict' in checkpoint:
                        chronos_model.quantile_head.load_state_dict(checkpoint['quantile_head.state_dict'])
                    elif 'quantile_head' in checkpoint:
                        chronos_model.quantile_head.load_state_dict(checkpoint['quantile_head'])
                    
                    if 'value_embedding.state_dict' in checkpoint:
                        chronos_model.value_embedding.load_state_dict(checkpoint['value_embedding.state_dict'])
                    elif 'value_embedding' in checkpoint:
                        chronos_model.value_embedding.load_state_dict(checkpoint['value_embedding'])
            except Exception as e:
                # Custom heads might not be available, will use default heads
                print(f"Warning: Could not load custom heads from heads.pt: {e}")
        except Exception as e:
            print(f"Warning: Could not download heads.pt: {e}")
        
        # Move to device
        chronos_model = chronos_model.to(torch_device)
        chronos_model.eval()
        
        return chronos_model
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        # Check if it's a 404 error
        if "404" in error_msg or "Entry Not Found" in error_msg:
            print(f"Warning: Model '{model_id}' not found on Hugging Face Hub.")
            print("Please check that the model exists at: https://huggingface.co/" + model_id)
        else:
            # Print the actual error for debugging
            print(f"Error loading Chronos model from Hugging Face: {error_msg}")
            import traceback
            traceback.print_exc()
        return None


def predict_volatility_chronos(
    df: pd.DataFrame,
    prediction_window: int = 20,
    model_id: str = 'karkar69/chronos-volatility',
    device: str = 'auto',
    symbol: Optional[str] = None,
) -> Dict:
    """
    Predict volatility using Chronos model from Hugging Face for rolling window.
    
    The model predicts quantiles (q10, q50, q90) for log-variance for a 20-day forward horizon.
    This function performs rolling predictions for every day in the prediction window.
    
    Uses Black-Scholes implied volatility from option prices as the baseline/current volatility
    when symbol is provided. Falls back to realized volatility if IV is unavailable.
    
    Args:
        df: DataFrame with OHLC data (must have 'close' and 'date' columns)
        prediction_window: Number of days to predict ahead (default: 20)
        model_id: Hugging Face model ID (default: 'karkar69/chronos-volatility')
        device: Device for inference ('auto', 'cuda', 'cpu')
        symbol: Asset symbol (e.g., 'AAPL', 'META'). If provided, uses Black-Scholes
                implied volatility from option prices as baseline. Optional.
    
    Returns:
        Dictionary with:
        - 'volatility': List of volatility predictions (q50) for each day in prediction window
        - 'lower': List of lower bounds (q10) for each day
        - 'upper': List of upper bounds (q90) for each day
        - 'dates': List of dates for predictions
        - 'model_info': Dictionary with model metadata
    """
    # Try to import required modules
    try:
        from src.data.returns import calculate_returns
        from src.volatility.models.base_model import get_device
        import torch
        import numpy as np
    except ImportError as e:
        return {
            'error': f'Required modules not available: {str(e)}',
            'volatility': None
        }
    
    try:
        # Get device
        torch_device = get_device(device)
        
        # Calculate squared returns (import already done above)
        returns = calculate_returns(df['close'], method='log')
        squared_returns = returns ** 2
        
        # Need at least 252 days of data (1 year of trading days)
        seq_length = 252  # Match training sequence length
        if len(squared_returns.dropna()) < seq_length:
            return {
                'error': f'Insufficient data: need at least {seq_length} days, got {len(squared_returns.dropna())}',
                'volatility': None
            }
        
        # Load model (cached per session)
        model = load_chronos_from_hf(model_id=model_id, device=device)
        if model is None:
            return {
                'error': 'Failed to load Chronos model from Hugging Face',
                'volatility': None
            }
        
        # Get the last 252 days of squared returns for input (1 year)
        squared_returns_clean = squared_returns.dropna()
        input_seq = squared_returns_clean.iloc[-seq_length:].values if len(squared_returns_clean) >= seq_length else squared_returns_clean.values
        
        # Initialize lists for predictions (will be populated in the prediction loop)
        volatility_predictions = []
        lower_bounds = []
        upper_bounds = []
        
        # The model predicts 20-day forward horizon quantiles
        # For each day in prediction window, we'll use the model's prediction
        # Since the model is trained to predict 20-day forward, we'll use q50 for the target horizon
        
        model.eval()
        with torch.no_grad():
            # Prepare input
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(torch_device)
            
            # Get prediction (q10, q50, q90 for 20-day forward horizon)
            quantiles = model(input_tensor)  # (1, 3)
            q10, q50, q90 = quantiles[0].cpu().numpy()
            
            # The model predicts log-realized variance for a 20-day forward horizon
            # Realized variance = sum of squared returns over 20 days (not average)
            # Target format from training: log(sum(r²) over next 20 days)
            
            # Convert log-realized variance to realized variance
            # This is the cumulative sum of squared returns over 20 days
            realized_variance_20day_q10 = np.exp(q10)
            realized_variance_20day_q50 = np.exp(q50)
            realized_variance_20day_q90 = np.exp(q90)
            
            # Convert 20-day realized variance to daily variance estimate
            # Realized variance is sum, so divide by 20 to get average daily variance
            # daily_variance = (sum of r² over 20 days) / 20
            daily_variance_q10 = realized_variance_20day_q10 / 20.0
            daily_variance_q50 = realized_variance_20day_q50 / 20.0
            daily_variance_q90 = realized_variance_20day_q90 / 20.0
            
            # Convert daily variance to annualized volatility percentage
            # Annualized variance = daily_variance * 252
            # Annualized volatility = sqrt(annualized_variance) * 100
            vol_q10 = np.sqrt(daily_variance_q10 * 252) * 100
            vol_q50 = np.sqrt(daily_variance_q50 * 252) * 100
            vol_q90 = np.sqrt(daily_variance_q90 * 252) * 100
            
            # Calculate current volatility to use as starting point
            # Prioritize Black-Scholes implied volatility from option prices if symbol is provided
            current_vol = None
            if symbol:
                try:
                    from src.data import get_implied_volatility
                    # Get Black-Scholes implied volatility from option prices
                    iv_bs = get_implied_volatility(
                        symbol=symbol,
                        df=df,
                        horizon_days=prediction_window,
                        use_api=True
                    )
                    if iv_bs is not None:
                        # IV is returned as percentage, use it directly
                        current_vol = iv_bs
                except Exception:
                    # If IV fetch fails, fall back to realized volatility below
                    pass
            
            # Fall back to realized volatility if IV not available
            if current_vol is None:
                # Use recent squared returns to estimate current realized volatility
                recent_squared_returns = squared_returns.dropna().iloc[-20:] if len(squared_returns.dropna()) >= 20 else squared_returns.dropna()
                current_rv_20day = recent_squared_returns.sum() if len(recent_squared_returns) > 0 else 0.0
                current_daily_var = current_rv_20day / len(recent_squared_returns) if len(recent_squared_returns) > 0 else daily_variance_q50
                current_vol = np.sqrt(current_daily_var * 252) * 100
            
            # Calculate historical daily volatility from squared returns
            # This gives us the actual volatility path the stock has taken historically
            from src.utils.volatility_utils import (
                calculate_daily_realized_volatility,
                interpolate_volatility_from_historical_patterns
            )
            
            # Calculate full historical volatility series for pattern matching
            historical_vol = calculate_daily_realized_volatility(
                squared_returns.dropna(),
                window=20
            )
            
            # Generate prediction dates first to know how many trading days we actually need
            last_date = pd.to_datetime(df['date'].iloc[-1])
            prediction_dates_calendar = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=prediction_window * 2,  # Generate enough to account for weekends
                freq='D'
            ).tolist()
            
            # Filter out weekends (keep only trading days)
            prediction_dates = [d for d in prediction_dates_calendar if d.weekday() < 5][:prediction_window]
            num_trading_days = len(prediction_dates)
            
            # Use historical patterns to interpolate between current and predicted volatility
            # This is based on actual historical transitions, not synthetic patterns
            vol_path_q50 = interpolate_volatility_from_historical_patterns(
                historical_vol=historical_vol,
                start_vol=current_vol,
                end_vol=vol_q50,
                horizon=num_trading_days,
                min_patterns=3
            )
            
            # Generate paths for bounds (q10 and q90) using same historical pattern approach
            vol_path_q10 = interpolate_volatility_from_historical_patterns(
                historical_vol=historical_vol,
                start_vol=current_vol * 0.85,  # Approximate current lower bound
                end_vol=vol_q10,
                horizon=num_trading_days,
                min_patterns=2  # Lower threshold for bounds
            )
            
            vol_path_q90 = interpolate_volatility_from_historical_patterns(
                historical_vol=historical_vol,
                start_vol=current_vol * 1.15,  # Approximate current upper bound
                end_vol=vol_q90,
                horizon=num_trading_days,
                min_patterns=2
            )
            
            # Store the generated paths (already trimmed to trading days)
            # Convert numpy arrays to lists of floats
            volatility_predictions = [float(v) for v in vol_path_q50]
            lower_bounds = [float(v) for v in vol_path_q10]
            upper_bounds = [float(v) for v in vol_path_q90]
            
            # Ensure all paths have the same length (should match num_trading_days)
            if len(volatility_predictions) != num_trading_days:
                # If lengths don't match, trim or pad to match
                if len(volatility_predictions) > num_trading_days:
                    volatility_predictions = volatility_predictions[:num_trading_days]
                    lower_bounds = lower_bounds[:num_trading_days]
                    upper_bounds = upper_bounds[:num_trading_days]
                else:
                    # Pad with last value if needed (shouldn't happen, but safety check)
                    last_vol = volatility_predictions[-1] if volatility_predictions else vol_q50
                    last_lower = lower_bounds[-1] if lower_bounds else vol_q10
                    last_upper = upper_bounds[-1] if upper_bounds else vol_q90
                    while len(volatility_predictions) < num_trading_days:
                        volatility_predictions.append(last_vol)
                        lower_bounds.append(last_lower)
                        upper_bounds.append(last_upper)
        
        # Generate prediction dates (if not already generated above)
        if 'prediction_dates' not in locals():
            last_date = pd.to_datetime(df['date'].iloc[-1])
            prediction_dates_calendar = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=prediction_window * 2,
                freq='D'
            ).tolist()
            prediction_dates = [d for d in prediction_dates_calendar if d.weekday() < 5][:prediction_window]
        
        return {
            'volatility': volatility_predictions,
            'lower': lower_bounds,
            'upper': upper_bounds,
            'dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
            'model_info': {
                'model_id': model_id,
                'device': str(torch_device),
                'prediction_window': prediction_window,
                'input_sequence_length': 252,
            }
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {
            'error': f"{str(e)}\n\nFull traceback:\n{error_trace}",
            'volatility': None
        }


def get_dl_model_summary() -> Dict:
    """
    Get summary of available deep learning models.
    
    Returns:
        Dictionary with model information
    """
    summary = {
        'deep_learning_available': _DL_AVAILABLE,
        'models': {}
    }
    
    if _DL_AVAILABLE:
        summary['models'] = {
            'chronos_volatility': {
                'description': 'Chronos-based forward volatility prediction',
                'input': 'Historical squared returns (252 days = 1 year)',
                'output': 'Volatility predictions at 20-day horizon (q10, q50, q90)',
                'architecture': 'Chronos T5 Mini with LoRA',
            },
        }
        
        # Check for GPU
        try:
            from src.volatility.models.base_model import get_device_info
            device_info = get_device_info()
            summary['device'] = str(device_info)
        except:
            summary['device'] = 'Unknown'
    
    return summary
