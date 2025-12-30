#!/usr/bin/env python3
"""Simple training script with 5-ticker test dataset for quick validation."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Subset

from src.models.chronos import ChronosVolatility
from src.training.data import prepare_raw_signal, compute_target, VolatilityDataset
from src.training.finetune import train
from src.models.base_model import get_device
from src.data.data_loader import get_market_data
from datetime import datetime, timedelta


def load_and_prepare_ticker_data(ticker, data_dir, start_date, end_date):
    """Load or download data for a single ticker and prepare it for training."""
    data_path = data_dir / f'{ticker}.parquet'
    
    # If cache file doesn't exist, download data
    if not data_path.exists():
        print(f"Downloading {ticker}...", end=' ')
        try:
            df = get_market_data(
                symbol=ticker,
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
        # Load from cache
        df = pd.read_parquet(data_path)
    
    # Ensure date column exists and set as index for alignment
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
        df = df.set_index('date').sort_index()
    elif df.index.name == 'date' or (hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index)):
        df = df.sort_index()
    
    # Compute returns
    returns = np.log(df['close'] / df['close'].shift(1))
    returns = returns.dropna()
    
    # Prepare raw signal (squared returns)
    raw_signal = returns ** 2
    
    # Compute target (log-realized variance)
    target = compute_target(returns, horizon=20)
    
    # Align raw_signal and target
    common_idx = raw_signal.index.intersection(target.index)
    raw_signal = raw_signal.loc[common_idx]
    target = target.loc[common_idx]
    
    # Create dataset
    dataset = VolatilityDataset(raw_signal, target, seq_length=60, horizon=20)
    return dataset


def main():
    """Main training function with 5-ticker test dataset."""
    # Simple 5-ticker test dataset
    train_tickers = ['AAPL', 'MSFT', 'GOOG', 'SPY', 'TSLA']
    test_ticker = 'NVDA'  # Held-out test ticker
    
    print(f"Training on {len(train_tickers)} tickers: {train_tickers}")
    print(f"Test ticker: {test_ticker}")
    
    # Data directory
    data_dir = Path('data/cache')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Date range for downloading data (last 10 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    
    # Load and prepare training data
    train_datasets = []
    print(f"\nLoading training data...")
    for ticker in train_tickers:
        dataset = load_and_prepare_ticker_data(ticker, data_dir, start_date, end_date)
        if dataset is not None and len(dataset) > 0:
            train_datasets.append(dataset)
            print(f"  ✓ {ticker}: {len(dataset)} samples")
    
    if not train_datasets:
        print("Error: No training datasets loaded!")
        return
    
    # Combine all training tickers
    combined_dataset = ConcatDataset(train_datasets)
    print(f"\n✓ Total training samples: {len(combined_dataset):,}")
    
    # Split into train/validation sets (80/20)
    train_size = int(0.8 * len(combined_dataset))
    indices = torch.randperm(len(combined_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_data = Subset(combined_dataset, train_indices.tolist())
    val_data = Subset(combined_dataset, val_indices.tolist())
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Model
    device = get_device('auto')
    print(f"\nUsing device: {device}")
    
    model = ChronosVolatility(use_lora=True).to(device)
    
    # Train
    print("\nStarting training...")
    # Use lower learning rate to prevent gradient explosion and NaN losses
    model = train(model, train_loader, val_loader, epochs=50, lr=1e-5, device=device)
    
    # Save
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'chronos_5ticker.pt'
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining complete. Model saved to {checkpoint_path}")
    
    # Test on held-out ticker
    print(f"\nPreparing test ticker: {test_ticker}")
    test_dataset = load_and_prepare_ticker_data(test_ticker, data_dir, start_date, end_date)
    if test_dataset is not None:
        print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
        # TODO: Run evaluation on test dataset
    else:
        print(f"✗ Failed to load test ticker {test_ticker}")


if __name__ == '__main__':
    main()
