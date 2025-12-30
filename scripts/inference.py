#!/usr/bin/env python3
"""Simple inference script."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd

from src.models.chronos import ChronosVolatility
from src.prediction.inference import predict
from src.training.data import prepare_raw_signal
from src.models.base_model import get_device


def main(ticker='AAPL'):
    """Main inference function."""
    # Load data
    data_path = Path('data/cache') / f'{ticker}.parquet'
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return
    
    df = pd.read_parquet(data_path)
    
    # Prepare raw signal (last 60 days of squared returns)
    raw_signal = prepare_raw_signal(df)
    if len(raw_signal) < 60:
        print(f"Error: Not enough data. Need at least 60 days, got {len(raw_signal)}")
        return
    
    context = raw_signal.iloc[-60:].values  # Last 60 days
    
    # Load model
    device = get_device('auto')
    checkpoint_path = Path('models/checkpoints/chronos.pt')
    
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using scripts/train.py")
        return
    
    model = ChronosVolatility(use_lora=True).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Predict
    result = predict(model, context, device=device)
    
    print(f"\nPredicted volatility for {ticker}:")
    print(f"  Point estimate (q50): {result['volatility']:.2f}%")
    print(f"  90% interval: [{result['lower']:.2f}%, {result['upper']:.2f}%]")
    print(f"  Log-variance (q50): {result['log_variance_q50']:.4f}")


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    main(ticker)

