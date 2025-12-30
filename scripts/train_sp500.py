#!/usr/bin/env python3
"""Full S&P 500 training script - fetches S&P 500 list and trains on all companies."""

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


def get_sp500_tickers():
    """
    Fetch list of S&P 500 ticker symbols from Wikipedia.
    
    Returns:
        List of ticker symbols (strings)
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        print(f"Fetching S&P 500 ticker list from Wikipedia...")
        
        # Try with different backends
        try:
            tables = pd.read_html(url, flavor='lxml')
        except:
            try:
                tables = pd.read_html(url, flavor='html5lib')
            except:
                tables = pd.read_html(url)
        
        sp500_table = tables[0]
        
        # The Symbol column might be named differently, try common names
        if 'Symbol' in sp500_table.columns:
            tickers = sp500_table['Symbol'].tolist()
        elif 'Ticker symbol' in sp500_table.columns:
            tickers = sp500_table['Ticker symbol'].tolist()
        else:
            # Try first column
            tickers = sp500_table.iloc[:, 0].tolist()
        
        # Clean tickers - replace dots with hyphens for yfinance compatibility
        tickers = [str(t).replace('.', '-') for t in tickers if pd.notna(t)]
        tickers = [t.strip() for t in tickers if t]
        
        print(f"✓ Loaded {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"✗ Error fetching S&P 500 list from Wikipedia: {e}")
        print("  Note: You may need to install html5lib or lxml: pip install html5lib lxml")
        print("Falling back to hardcoded list...")
        # Fallback: return a subset if web scraping fails
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'JNJ']


def load_and_prepare_ticker_data(ticker, data_dir, start_date, end_date):
    """Load or download data for a single ticker and prepare it for training."""
    data_path = data_dir / f'{ticker}.parquet'
    
    # If cache file doesn't exist, download data
    if not data_path.exists():
        try:
            df = get_market_data(
                symbol=ticker,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                cache_dir=str(data_dir),
                cache_format='parquet'
            )
            return df
        except Exception as e:
            return None
    else:
        # Load from cache
        try:
            df = pd.read_parquet(data_path)
            return df
        except Exception:
            return None


def process_ticker_to_dataset(df, ticker):
    """Process DataFrame into VolatilityDataset."""
    if df is None or df.empty:
        return None
    
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
    
    # Check if 'close' column exists
    if 'close' not in df.columns:
        return None
    
    # Compute returns
    returns = np.log(df['close'] / df['close'].shift(1))
    returns = returns.dropna()
    
    if len(returns) < 100:  # Need sufficient data
        return None
    
    # Prepare raw signal (squared returns)
    raw_signal = returns ** 2
    
    # Compute target (log-realized variance)
    target = compute_target(returns, horizon=20)
    
    # Align raw_signal and target
    common_idx = raw_signal.index.intersection(target.index)
    if len(common_idx) < 100:  # Need sufficient data
        return None
    
    raw_signal = raw_signal.loc[common_idx]
    target = target.loc[common_idx]
    
    # Create dataset
    dataset = VolatilityDataset(raw_signal, target, seq_length=60, horizon=20)
    return dataset


def main():
    """Main training function with full S&P 500 dataset."""
    # Fetch S&P 500 ticker list
    all_tickers = get_sp500_tickers()
    
    if not all_tickers:
        print("Error: Could not fetch ticker list!")
        return
    
    # Split into training and test sets (80/20 split by tickers)
    np.random.seed(42)
    np.random.shuffle(all_tickers)
    
    split_idx = int(0.8 * len(all_tickers))
    train_tickers = all_tickers[:split_idx]
    test_tickers = all_tickers[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Training tickers: {len(train_tickers)}")
    print(f"  Test tickers: {len(test_tickers)}")
    
    # Data directory
    data_dir = Path('data/cache')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Date range for downloading data (last 10 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    
    # Load and prepare training data
    train_datasets = []
    print(f"\nLoading/preparing training data for {len(train_tickers)} tickers...")
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(train_tickers, 1):
        # Download or load from cache
        df = load_and_prepare_ticker_data(ticker, data_dir, start_date, end_date)
        
        if df is None:
            failed += 1
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(train_tickers)} processed, {successful} successful, {failed} failed")
            continue
        
        # Process to dataset
        if i % 10 == 0:
            print(f"  [{i}/{len(train_tickers)}] Processing {ticker}...", end=' ')
        
        dataset = process_ticker_to_dataset(df, ticker)
        
        if dataset is not None and len(dataset) > 0:
            train_datasets.append(dataset)
            successful += 1
            if i % 10 == 0 or i == len(train_tickers):
                total_samples = sum(len(d) for d in train_datasets)
                print(f"✓ {successful} successful, {total_samples:,} total samples")
        else:
            failed += 1
            if i % 10 == 0:
                print(f"✗ insufficient data")
    
    if not train_datasets:
        print("Error: No training datasets loaded!")
        return
    
    # Combine all training tickers
    combined_dataset = ConcatDataset(train_datasets)
    print(f"\n✓ Successfully loaded {len(train_datasets)}/{len(train_tickers)} training tickers")
    print(f"✓ Total training samples: {len(combined_dataset):,}")
    
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
    model = train(model, train_loader, val_loader, epochs=50, lr=1e-4, device=device)
    
    # Save
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'chronos_sp500.pt'
    
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining complete. Model saved to {checkpoint_path}")
    
    # Test on held-out tickers
    print(f"\nPreparing {len(test_tickers)} test tickers for evaluation...")
    test_datasets = []
    for i, ticker in enumerate(test_tickers, 1):
        df = load_and_prepare_ticker_data(ticker, data_dir, start_date, end_date)
        if df is not None:
            dataset = process_ticker_to_dataset(df, ticker)
            if dataset is not None:
                test_datasets.append((ticker, dataset))
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(test_tickers)} test tickers, {len(test_datasets)} loaded")
    
    print(f"\n✓ Loaded {len(test_datasets)} test tickers for evaluation")
    # TODO: Run evaluation on test datasets


if __name__ == '__main__':
    main()

