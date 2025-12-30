#!/usr/bin/env python3
"""Test trained Chronos model on NVDA volatility predictions."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from src.models.chronos import ChronosVolatility
from src.training.data import prepare_raw_signal, compute_target
from src.models.base_model import get_device
from src.data.data_loader import get_market_data


def load_nvda_data(data_dir, start_date, end_date):
    """Load or download NVDA data."""
    data_path = data_dir / 'NVDA.parquet'
    
    if not data_path.exists():
        print(f"Downloading NVDA data...", end=' ')
        try:
            df = get_market_data(
                symbol='NVDA',
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                cache_dir=str(data_dir),
                cache_format='parquet'
            )
            print(f"✓ {len(df)} rows")
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    else:
        df = pd.read_parquet(data_path)
        print(f"Loaded NVDA data from cache: {len(df)} rows")
    
    # Ensure date column exists and set as index
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
        df = df.set_index('date').sort_index()
    elif df.index.name == 'date' or (hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index)):
        df = df.sort_index()
    
    return df


def prepare_prediction_data(df, seq_length=60):
    """Prepare data for prediction (same as training)."""
    # Compute returns
    returns = np.log(df['close'] / df['close'].shift(1))
    returns = returns.dropna()
    
    # Prepare raw signal (squared returns)
    raw_signal = returns ** 2
    
    # Use last seq_length values for prediction
    if len(raw_signal) < seq_length:
        raise ValueError(f"Not enough data: need at least {seq_length} days, got {len(raw_signal)}")
    
    # Get the last seq_length values
    input_seq = raw_signal.iloc[-seq_length:].values.astype(np.float32)
    
    # Clean NaN/inf
    input_seq = np.nan_to_num(input_seq, nan=1e-8, posinf=1.0, neginf=1e-8)
    input_seq = np.clip(input_seq, a_min=0.0, a_max=1e6)
    
    return input_seq, returns, df.index[-seq_length:]


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"\nLoading model from {checkpoint_path}...")
    
    # Initialize model (must match training configuration)
    model = ChronosVolatility(use_lora=True).to(device)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model


def make_predictions(model, input_seq, device, num_samples=100):
    """
    Make volatility predictions using the model.
    
    Args:
        model: Trained ChronosVolatility model
        input_seq: Input sequence of squared returns (seq_length,)
        device: Computing device
        num_samples: Number of prediction samples to generate
        
    Returns:
        Dictionary with predictions (quantiles in log-variance space and volatility)
    """
    # Prepare input tensor: (1, seq_length)
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get quantile predictions (q10, q50, q90) in log-variance space
        quantiles_log_var = model(input_tensor)  # (1, 3)
        quantiles_log_var = quantiles_log_var.cpu().numpy().flatten()  # (3,)
    
    # Convert log-variance to volatility (annualized)
    # log_variance -> variance -> volatility (std) -> annualized
    # variance = exp(log_variance)
    # volatility = sqrt(variance) * sqrt(252) for annualization
    quantiles_var = np.exp(np.clip(quantiles_log_var, a_min=-20, a_max=20))
    quantiles_vol = np.sqrt(quantiles_var) * np.sqrt(252)  # Annualized volatility
    
    # Also compute expected volatility from q50
    expected_vol = quantiles_vol[1]  # q50 is the median
    
    return {
        'quantiles_log_var': quantiles_log_var,  # [q10, q50, q90] in log-variance space
        'quantiles_vol': quantiles_vol,  # [q10, q50, q90] annualized volatility (%)
        'expected_vol': expected_vol,  # Expected (median) annualized volatility
        'q10_vol': quantiles_vol[0],
        'q50_vol': quantiles_vol[1],
        'q90_vol': quantiles_vol[2],
    }


def visualize_predictions(df, returns, predictions, output_dir):
    """Create visualization of predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Get dates for the last 200 days (for context)
    plot_days = min(200, len(df))
    plot_dates = df.index[-plot_days:]
    
    # Plot 1: Price and recent volatility
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    prices = df['close'].iloc[-plot_days:]
    recent_returns = returns.iloc[-plot_days:]
    recent_vol = recent_returns.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized %
    
    ax1.plot(plot_dates, prices, 'b-', label='NVDA Close Price', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax1_twin.plot(plot_dates[-len(recent_vol):], recent_vol, 'r--', label='20-day Realized Vol', alpha=0.7)
    ax1_twin.set_ylabel('Realized Volatility (%)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('NVDA Price and Recent Volatility')
    
    # Plot 2: Returns
    ax2 = axes[1]
    ax2.plot(plot_dates[-len(recent_returns):], recent_returns * 100, 'g-', alpha=0.6, linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Returns (%)')
    ax2.set_title('Recent Daily Returns')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction with confidence intervals
    ax3 = axes[2]
    
    # Add prediction point at the end
    pred_date = plot_dates[-1] + timedelta(days=1)
    
    # Plot prediction
    q10 = predictions['q10_vol']
    q50 = predictions['q50_vol']
    q90 = predictions['q90_vol']
    
    ax3.barh(['Predicted Vol'], [q50], color='blue', alpha=0.6, label='Median (q50)')
    ax3.errorbar([q50], [0], xerr=[[q50 - q10], [q90 - q50]], 
                 fmt='o', color='red', markersize=10, capsize=10, capthick=2,
                 label=f'90% CI: [{q10:.1f}%, {q90:.1f}%]')
    
    # Add text annotations
    ax3.text(q50, 0, f'  {q50:.1f}%', va='center', fontsize=12, fontweight='bold')
    ax3.text(q10, -0.15, f'q10: {q10:.1f}%', ha='center', fontsize=10, alpha=0.7)
    ax3.text(q90, -0.15, f'q90: {q90:.1f}%', ha='center', fontsize=10, alpha=0.7)
    
    ax3.set_xlabel('Predicted Annualized Volatility (%)')
    ax3.set_title(f'NVDA 20-Day Forward Volatility Prediction\n(Prediction date: {plot_dates[-1].strftime("%Y-%m-%d")})')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = output_dir / 'nvda_volatility_prediction.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    plt.close()


def main():
    """Main testing function."""
    print("=" * 60)
    print("NVDA Volatility Prediction Test")
    print("=" * 60)
    
    # Configuration (must match training)
    seq_length = 60
    horizon = 20
    checkpoint_path = Path('models/checkpoints/chronos_5ticker.pt')
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"\n✗ Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python3 scripts/train.py")
        return
    
    # Data directory
    data_dir = Path('data/cache')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Date range (use last 10 years for context, but prediction uses most recent data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    
    # Load NVDA data
    print(f"\nLoading NVDA data...")
    df = load_nvda_data(data_dir, start_date, end_date)
    if df is None:
        print("✗ Failed to load NVDA data")
        return
    
    # Prepare prediction data
    print(f"\nPreparing prediction data...")
    try:
        input_seq, returns, input_dates = prepare_prediction_data(df, seq_length)
        print(f"✓ Using data from {input_dates[0]} to {input_dates[-1]}")
        print(f"✓ Input sequence length: {len(input_seq)}")
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        return
    
    # Load model
    device = get_device('auto')
    print(f"\nUsing device: {device}")
    
    try:
        model = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Make predictions
    print(f"\nMaking predictions...")
    predictions = make_predictions(model, input_seq, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nPrediction Date: {input_dates[-1]}")
    print(f"Forecast Horizon: {horizon} trading days (~{horizon/5:.1f} weeks)")
    print(f"\nPredicted Annualized Volatility:")
    print(f"  Median (q50):  {predictions['q50_vol']:.2f}%")
    print(f"  Lower (q10):   {predictions['q10_vol']:.2f}%")
    print(f"  Upper (q90):   {predictions['q90_vol']:.2f}%")
    print(f"  90% Confidence Interval: [{predictions['q10_vol']:.2f}%, {predictions['q90_vol']:.2f}%]")
    
    # Context: recent realized volatility
    recent_realized = returns.iloc[-20:].std() * np.sqrt(252) * 100  # Annualized %
    print(f"\nRecent 20-day Realized Volatility: {recent_realized:.2f}%")
    
    if predictions['q50_vol'] > recent_realized:
        print(f"→ Prediction suggests volatility will INCREASE")
    else:
        print(f"→ Prediction suggests volatility will DECREASE")
    
    # Create visualization
    print(f"\nCreating visualization...")
    visualize_predictions(df, returns, predictions, 'results')
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

