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
import subprocess
import socket

from src.volatility.models.chronos import ChronosVolatility
from src.volatility.training.data import prepare_raw_signal, compute_target, VolatilityDataset, create_weighted_sampler_for_concat_dataset
from src.volatility.training.finetune import train
from src.volatility.models.base_model import get_device
from src.data.data_loader import get_market_data
from datetime import datetime, timedelta


def is_lambda_instance():
    """Check if running on a Lambda Labs instance."""
    # Check hostname for Lambda patterns
    try:
        hostname = socket.gethostname()
        if 'lambda' in hostname.lower() or 'lambdalabs' in hostname.lower():
            return True
    except:
        pass
    
    # Check if lambdacloud CLI is available (official Lambda Labs CLI)
    try:
        result = subprocess.run(['which', 'lambdacloud'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
    except:
        pass
    
    # Check if lambda CLI is available (third-party lambda-cli)
    try:
        result = subprocess.run(['which', 'lambda'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
    except:
        pass
    
    # Check environment variable (can be set manually)
    if os.environ.get('LAMBDA_INSTANCE_ID'):
        return True
    
    return False


def get_lambda_instance_id():
    """Get Lambda Labs instance ID from metadata or environment."""
    # Try environment variable first (most reliable)
    instance_id = os.environ.get('LAMBDA_INSTANCE_ID')
    if instance_id:
        return instance_id
    
    # Try to get from lambdacloud CLI (official)
    try:
        result = subprocess.run(['lambdacloud', 'instance', 'list', '--format', 'json'],
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            instances = json.loads(result.stdout)
            # Get the first running instance
            for instance in instances:
                if instance.get('status') == 'running':
                    return instance.get('id')
    except:
        pass
    
    # Try to get from lambda CLI (third-party lambda-cli)
    try:
        result = subprocess.run(['lambda', 'ls', '--json'],
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            instances = json.loads(result.stdout)
            # Get the first running instance
            if isinstance(instances, list) and len(instances) > 0:
                return instances[0].get('id') or instances[0].get('instance_id')
    except:
        pass
    
    # Try to get from instance metadata (if available)
    try:
        import urllib.request
        metadata_url = 'http://169.254.169.254/latest/meta-data/instance-id'
        with urllib.request.urlopen(metadata_url, timeout=2) as response:
            return response.read().decode('utf-8')
    except:
        pass
    
    return None


def terminate_lambda_instance(instance_id=None):
    """Terminate Lambda Labs instance after training completes."""
    # Check if auto-termination is disabled
    if os.environ.get('LAMBDA_NO_AUTO_TERMINATE', '').lower() in ('1', 'true', 'yes'):
        print("\n⚠ Auto-termination disabled (LAMBDA_NO_AUTO_TERMINATE is set)")
        print("  Instance will remain running. Terminate manually when done.")
        return False
    
    if not is_lambda_instance():
        return False
    
    if instance_id is None:
        instance_id = get_lambda_instance_id()
    
    if not instance_id:
        print("\n⚠ Warning: Could not determine Lambda instance ID. Please terminate manually.")
        print("  To enable auto-termination, set LAMBDA_INSTANCE_ID environment variable:")
        print("  export LAMBDA_INSTANCE_ID='your-instance-id'")
        print("  Or run: lambdacloud instance terminate <instance-id>")
        print("         OR: lambda rm <instance-id>")
        return False
    
    print(f"\n{'='*60}")
    print("Training complete. Terminating Lambda Labs instance...")
    print(f"{'='*60}")
    
    # Try lambdacloud CLI first (official Lambda Labs CLI)
    try:
        print(f"Attempting to terminate instance {instance_id} via lambdacloud...")
        result = subprocess.run(['lambdacloud', 'instance', 'terminate', instance_id],
                               capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✓ Successfully terminated instance: {instance_id}")
            print("  Instance will shut down shortly. You may be disconnected.")
            return True
    except FileNotFoundError:
        # Try lambda CLI (third-party lambda-cli) as fallback
        try:
            print(f"lambdacloud not found, trying lambda CLI...")
            result = subprocess.run(['lambda', 'rm', instance_id],
                                   capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✓ Successfully terminated instance: {instance_id}")
                print("  Instance will shut down shortly. You may be disconnected.")
                return True
            else:
                print(f"✗ Failed to terminate with lambda CLI: {result.stderr}")
        except FileNotFoundError:
            print("⚠ Neither lambdacloud nor lambda CLI found.")
            print("  Install one of:")
            print("    pip install lambdacloud  # Official Lambda Labs CLI")
            print("    pip install lambda-cli  # Third-party CLI")
        except Exception as e:
            print(f"✗ Error with lambda CLI: {e}")
    except subprocess.TimeoutExpired:
        print("⚠ Timeout while terminating instance. Please terminate manually.")
    except Exception as e:
        print(f"✗ Error terminating instance: {e}")
    
    # If both CLIs failed, provide manual instructions
    print(f"\n⚠ Could not automatically terminate instance.")
    print(f"  Instance ID: {instance_id}")
    print(f"  Please terminate manually:")
    print(f"    lambdacloud instance terminate {instance_id}")
    print(f"    OR: lambda rm {instance_id}")
    print(f"    OR: https://lambdalabs.com/ (web dashboard)")
    
    # Fallback to shutdown command (last resort)
    print("\n  Attempting fallback shutdown method...")
    try:
        subprocess.run(['sudo', 'shutdown', '-h', 'now'],
                      timeout=5, capture_output=True, text=True)
        print("  ✓ Shutdown command sent (may not fully terminate Lambda instance)")
        print("  ⚠ WARNING: This only shuts down the OS. The instance may still be billing.")
        print("  Please terminate from Lambda Labs dashboard to stop billing.")
    except:
        pass
    
    return False


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
    
    # Create dataset (using 252 days = 1 year with recent weighting)
    dataset = VolatilityDataset(raw_signal, target, seq_length=252, horizon=20, use_recent_weighting=True)
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
    
    # Auto-terminate Lambda instance if running on Lambda Labs
    print(f"\n{'='*60}")
    print("Checking for Lambda Labs instance auto-termination...")
    print(f"{'='*60}")
    if is_lambda_instance():
        instance_id = get_lambda_instance_id()
        terminate_lambda_instance(instance_id)
    else:
        print("Not running on Lambda Labs instance - skipping auto-termination")


if __name__ == '__main__':
    main()
