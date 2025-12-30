#!/usr/bin/env python3
"""Simple training script with cross-ticker training."""

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


def main():
    """Main training function."""
    # CRITICAL: Reserve one ticker for held-out validation
    train_tickers = ['AAPL', 'MSFT', 'GOOG', 'SPY', 'TSLA']  # Train on these
    held_out_ticker = 'NVDA'  # Test on this (unseen during training)
    
    # Data directory
    data_dir = Path('data/cache')
    
    # Load and prepare training data (cross-ticker)
    train_datasets = []
    
    # Date range for downloading data (last 10 years should be sufficient)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    
    for ticker in train_tickers:
        data_path = data_dir / f'{ticker}.parquet'
        
        # If cache file doesn't exist, download data
        if not data_path.exists():
            print(f"Cache file not found for {ticker}, downloading data...")
            try:
                df = get_market_data(
                    symbol=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True,
                    cache_dir=str(data_dir),
                    cache_format='parquet'
                )
                print(f"Downloaded {ticker}: {len(df)} rows")
            except Exception as e:
                print(f"Error downloading {ticker}: {e}, skipping...")
                continue
        else:
            # Load from cache
            df = pd.read_parquet(data_path)
        
        # Ensure date column exists and set as index for alignment
        if 'date' in df.columns:
            # Convert date to datetime if needed, then set as index
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
            df = df.set_index('date').sort_index()
        elif df.index.name == 'date' or (hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index)):
            # Already has date index
            df = df.sort_index()
        
        # Compute returns
        returns = np.log(df['close'] / df['close'].shift(1))
        returns = returns.dropna()
        
        # Prepare raw signal (squared returns) - use the same index as returns
        raw_signal = returns ** 2
        
        # Compute target (log-realized variance)
        target = compute_target(returns, horizon=20)
        
        # Align raw_signal and target (both should have same index now)
        common_idx = raw_signal.index.intersection(target.index)
        raw_signal = raw_signal.loc[common_idx]
        target = target.loc[common_idx]
        
        # Create dataset
        dataset = VolatilityDataset(raw_signal, target, seq_length=60, horizon=20)
        if len(dataset) > 0:
            train_datasets.append(dataset)
            print(f"Loaded {ticker}: {len(dataset)} samples")
    
    if not train_datasets:
        print("Error: No training datasets loaded!")
        return
    
    # Combine all training tickers
    combined_dataset = ConcatDataset(train_datasets)
    print(f"Total training samples: {len(combined_dataset)}")
    
    # Temporal split (time series aware)
    # For time series, we should split by time, not randomly
    # But for simplicity, we'll do a random split here
    # In production, use time-based split
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
    print(f"Using device: {device}")
    
    model = ChronosVolatility(use_lora=True).to(device)
    
    # Train
    print("\nStarting training...")
    model = train(model, train_loader, val_loader, epochs=50, lr=1e-4, device=device)
    
    # Save
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'chronos.pt'
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining complete. Model saved to {checkpoint_path}")
    
    # Test on held-out ticker
    print(f"\nTesting on held-out ticker: {held_out_ticker}")
    test_path = data_dir / f'{held_out_ticker}.parquet'
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        # TODO: Run evaluation on held-out ticker
        print(f"Test data loaded for {held_out_ticker}")
    else:
        print(f"Cache file not found for {held_out_ticker}, downloading data...")
        try:
            test_df = get_market_data(
                symbol=held_out_ticker,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                cache_dir=str(data_dir),
                cache_format='parquet'
            )
            print(f"Downloaded {held_out_ticker}: {len(test_df)} rows")
            # TODO: Run evaluation on held-out ticker
            print(f"Test data loaded for {held_out_ticker}")
        except Exception as e:
            print(f"Warning: Failed to download {held_out_ticker}: {e}, skipping held-out test")


if __name__ == '__main__':
    main()

