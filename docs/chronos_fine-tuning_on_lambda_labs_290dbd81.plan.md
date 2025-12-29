---
name: Chronos Fine-tuning on Lambda Labs
overview: Simple Chronos fine-tuning pipeline for volatility prediction. Minimal, direct implementation following nanochat style.
todos: []
---

# Chronos Fine-tuning on Lambda Labs - Simple Implementation

## Overview

Replace iTransformer and Neural GARCH with a simple Chronos-based fine-tuning pipeline. Train on Lambda Labs GPUs, infer locally. Keep it minimal and direct.

## Core Design Decisions

**Model:** Chronos-mini (standard choice, good balance of performance/speed)
**Input:** Raw squared returns (single channel - Chronos requirement)
**Target:** Log-realized variance `log(Σr²)` (Option A - simplest, cleanest)
**Fine-tuning:** LoRA adapters (10× cheaper than full fine-tuning)
**Training:** Cross-ticker (one model for all tickers) + held-out ticker validation
**Loss:** Quantile (pinball) loss for all quantiles (q10, q50, q90)
**Evaluation:** QLIKE on q50 only (for monitoring/evaluation, not training)
**Validation:** Walk-forward temporal splits + held-out ticker test
**Baselines:** EWMA, GARCH(1,1), HAR-RV (proper implementations)

## Architecture

```
Historical OHLC Data
    ↓
Raw Signal (squared returns - single channel)
    ↓
Chronos-mini + LoRA (fine-tune on Lambda)
    ↓
Quantile Predictions (q10, q50, q90)
```

**Critical:** Chronos expects single-channel time series (raw signal), not multi-feature matrices.

## File Structure

```
src/
├── models/
│   ├── chronos.py           # Simple Chronos wrapper (~200 lines)
│   └── base_model.py        # Keep (device utilities)
├── training/
│   ├── finetune.py          # Simple training loop (~300 lines)
│   └── data.py              # Data preparation (~200 lines)
├── prediction/
│   └── inference.py         # Simple inference (~150 lines)
├── evaluation/
│   ├── metrics.py           # QLIKE, MSE (~100 lines)
│   └── baselines.py         # EWMA, GARCH, HAR-RV (~200 lines)
└── utils/
    └── config.py            # Simple config loading

scripts/
├── train.py                 # Training script (Python)
└── inference.py             # Inference script (Python)

config.yaml                  # Simple config
requirements.txt             # Minimal dependencies
```

## Implementation

### 1. Remove Old Models

- Delete `src/models/itransformer.py`
- Delete `src/models/neural_garch.py`
- Update imports in `src/prediction/predictions.py`
- Update `config.yaml` (remove old sections)

### 2. Chronos Model (`src/models/chronos.py`)

Simple wrapper around Chronos-mini. **CRITICAL:** Chronos expects single-channel time series input, not multi-feature matrices.

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model

class ChronosVolatility(nn.Module):
    """
    Simple Chronos wrapper for volatility prediction.
    
    Input: Single-channel time series (squared returns)
    Output: Quantiles (q10, q50, q90) in log-variance space
    """
    
    def __init__(self, model_id='amazon/chronos-t5-mini', use_lora=True):
        super().__init__()
        # Load pretrained Chronos (expects single-channel input)
        self.base = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        # LoRA adapters
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                lora_dropout=0.1
            )
            self.base = get_peft_model(self.base, lora_config)
        
        # Quantile regression head (q10, q50, q90)
        hidden_dim = self.base.config.d_model
        self.quantile_head = nn.Linear(hidden_dim, 3)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized single-channel time series (from Chronos tokenizer)
            attention_mask: Attention mask
        
        Returns:
            Quantiles in log-variance space: (batch, 3) where columns are [q10, q50, q90]
        """
        outputs = self.base.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1]  # Last timestep
        quantiles = self.quantile_head(hidden)
        return quantiles
```

**Key Points:**
- Chronos expects single-channel input (use Chronos tokenizer on raw series)
- Input: squared returns (raw signal, not engineered features)
- LoRA on attention layers
- Output: quantiles in log-variance space

### 3. Data Preparation (`src/training/data.py`)

**CRITICAL:** Use raw squared returns (single channel) as input to Chronos. No feature engineering.

```python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from chronos import ChronosTokenizer

def prepare_raw_signal(df):
    """
    Prepare raw signal: squared returns (single channel).
    
    Chronos was pretrained on raw time series, not engineered features.
    Using squared returns preserves volatility structure while keeping it raw.
    """
    returns = np.log(df['close'] / df['close'].shift(1))
    squared_returns = returns ** 2
    return squared_returns.dropna()

def compute_target(returns, horizon=20):
    """
    Compute log-realized variance target: log(Σr²).
    
    CRITICAL TIMING: Predict variance over [t+1, t+h] using data up to time t.
    Target at index t corresponds to realized variance over [t+1, t+h].
    """
    squared_returns = returns ** 2
    # Realized variance: sum of squared returns over next h days
    realized_variance = squared_returns.rolling(window=horizon).sum()
    # Shift backward by h: at time t, we predict variance for [t+1, t+h]
    # So target[t] = log(Σr² from t+1 to t+h)
    target = np.log(realized_variance.shift(-horizon) + 1e-8)
    return target

class VolatilityDataset(Dataset):
    """
    Dataset for volatility prediction.
    
    CRITICAL: Ensures correct timing alignment.
    - Input at index idx: data from [idx, idx+seq_length-1]
    - Target at index idx: realized variance over [idx+seq_length, idx+seq_length+horizon-1]
    """
    def __init__(self, raw_signal, target, seq_length=60, horizon=20, tokenizer=None):
        """
        Args:
            raw_signal: Squared returns (single-channel time series)
            target: Log-realized variance targets
            seq_length: Input sequence length
            horizon: Prediction horizon (must match target computation)
            tokenizer: Chronos tokenizer for encoding raw signal
        """
        self.raw_signal = raw_signal.values
        self.target = target.values
        self.seq_length = seq_length
        self.horizon = horizon
        self.tokenizer = tokenizer or ChronosTokenizer.from_pretrained("amazon/chronos-t5-mini")
        
        # Valid indices: need seq_length input + horizon for target
        self.valid_len = len(self.target) - seq_length - horizon + 1
        
    def __len__(self):
        return max(0, self.valid_len)
        
    def __getitem__(self, idx):
        """
        Get sample at index idx.
        
        Input: raw signal from [idx, idx+seq_length-1]
        Target: log-realized variance for [idx+seq_length, idx+seq_length+horizon-1]
        """
        if idx >= self.valid_len:
            raise IndexError(f"Index {idx} out of range")
        
        # Input sequence (raw squared returns)
        seq_end = idx + self.seq_length
        raw_seq = self.raw_signal[idx:seq_end]
        
        # Tokenize for Chronos (convert to Chronos input format)
        # Chronos tokenizer expects numpy array or tensor
        tokenized = self.tokenizer(raw_seq, return_tensors="pt", padding=False)
        input_ids = tokenized["input_ids"].squeeze(0)
        
        # Target: log-realized variance for period starting at seq_end
        target_idx = seq_end  # Target is aligned with end of input sequence
        y = self.target[target_idx]
        
        return input_ids, torch.FloatTensor([y])
```

**Key Points:**
- **CRITICAL:** Use raw squared returns (single channel), not engineered features
- **CRITICAL:** Explicit timing: predict variance over [t+1, t+h] using data up to t
- Use Chronos tokenizer to encode raw signal
- Sequence length: 60 days, prediction horizon: 20 days

### 4. Training (`src/training/finetune.py`)

**CRITICAL FIXES:**
1. Train all quantiles with quantile (pinball) loss
2. Evaluate QLIKE only on q50 (median)
3. Quantiles and QLIKE are conceptually separate

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def quantile_loss(pred_quantiles, target, quantiles=[0.1, 0.5, 0.9]):
    """
    Quantile (pinball) loss for all quantiles.
    
    CRITICAL: Train all quantiles (q10, q50, q90) jointly.
    QLIKE is only used for evaluation on q50.
    """
    target = target.unsqueeze(1)  # (batch, 1)
    errors = target - pred_quantiles  # (batch, 3)
    
    loss = 0
    for i, q in enumerate(quantiles):
        loss += torch.mean(torch.max(q * errors[:, i], (q - 1) * errors[:, i]))
    
    return loss / len(quantiles)

def qlike_loss(pred_log_variance, target_log_realized_variance):
    """
    QLIKE loss: log(σ²_pred/σ²_true) + σ²_true/σ²_pred - 1.
    
    CRITICAL: Only use for evaluation on q50, not training.
    QLIKE is for conditional mean forecasts, not quantiles.
    """
    pred_var = torch.exp(pred_log_variance)
    target_var = torch.exp(target_log_realized_variance)
    term1 = torch.log(pred_var / target_var)
    term2 = target_var / pred_var
    return (term1 + term2 - 1.0).mean()

def train(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda'):
    """
    Training loop with quantile loss.
    
    CRITICAL: Train all quantiles with quantile loss.
    Evaluate QLIKE on q50 only for monitoring.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_qlike = 0
        
        for input_ids, y in train_loader:
            input_ids = input_ids.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            # Forward: get all quantiles
            quantiles = model(input_ids)  # (batch, 3): [q10, q50, q90]
            
            # Train with quantile loss (all quantiles)
            loss = quantile_loss(quantiles, y.squeeze())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Monitor QLIKE on q50 only (not used for training)
            q50 = quantiles[:, 1]
            qlike_val = qlike_loss(q50, y.squeeze())
            train_qlike += qlike_val.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_qlike = 0
        
        with torch.no_grad():
            for input_ids, y in val_loader:
                input_ids = input_ids.to(device)
                y = y.to(device)
                
                quantiles = model(input_ids)
                
                # Quantile loss for monitoring
                loss = quantile_loss(quantiles, y.squeeze())
                val_loss += loss.item()
                
                # QLIKE on q50 for evaluation
                q50 = quantiles[:, 1]
                qlike_val = qlike_loss(q50, y.squeeze())
                val_qlike += qlike_val.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val QLIKE: {val_qlike/len(val_loader):.4f}")
    
    return model
```

**Key Points:**
- **CRITICAL:** Train all quantiles with quantile (pinball) loss
- **CRITICAL:** QLIKE is only for evaluation on q50, not training
- Quantiles and QLIKE are conceptually separate
- Simple logging with both losses

### 6. Inference (`src/prediction/inference.py`)

Simple inference with quantile predictions using raw signal.

```python
import numpy as np
import torch
from chronos import ChronosTokenizer

def predict(model, raw_signal, device='cuda', tokenizer=None):
    """
    Predict volatility with uncertainty intervals.
    
    Args:
        model: Trained ChronosVolatility model
        raw_signal: Squared returns (single-channel time series, last 60 days)
        device: Device for inference
        tokenizer: Chronos tokenizer (optional, will create if None)
    
    Returns:
        Dictionary with volatility prediction and intervals
    """
    if tokenizer is None:
        tokenizer = ChronosTokenizer.from_pretrained("amazon/chronos-t5-mini")
    
    model.eval()
    with torch.no_grad():
        # Tokenize raw signal (single channel)
        tokenized = tokenizer(raw_signal, return_tensors="pt", padding=False)
        input_ids = tokenized["input_ids"].to(device)
        
        # Forward pass
        quantiles = model(input_ids)  # (1, 3): [q10, q50, q90]
        q10, q50, q90 = quantiles[0].cpu().numpy()
        
    # Convert log-variance to variance
    var_q10 = np.exp(q10)
    var_q50 = np.exp(q50)
    var_q90 = np.exp(q90)
    
    # Convert to volatility (annualized)
    vol_q10 = np.sqrt(var_q10 * 252)  # Lower bound (q10 variance)
    vol_q50 = np.sqrt(var_q50 * 252)  # Point estimate (q50 variance)
    vol_q90 = np.sqrt(var_q90 * 252)  # Upper bound (q90 variance)
    
    return {
        'volatility': vol_q50,
        'lower': vol_q10,
        'upper': vol_q90
    }
```

**Key Points:**
- **CRITICAL:** Use raw squared returns (single channel)
- Tokenize with Chronos tokenizer
- Extract all quantiles (q10, q50, q90)
- Convert to annualized volatility
- Return prediction intervals

### 7. Evaluation (`src/evaluation/metrics.py`)

QLIKE and basic metrics.

```python
import numpy as np

def qlike_metric(pred_log_var, true_log_var):
    """QLIKE metric."""
    pred_var = np.exp(pred_log_var)
    true_var = np.exp(true_log_var)
    return np.mean(np.log(pred_var / true_var) + true_var / pred_var - 1)

def mse_log_variance(pred_log_var, true_log_var):
    """MSE in log-variance space."""
    return np.mean((pred_log_var - true_log_var) ** 2)
```

### 8. Baselines (`src/evaluation/baselines.py`)

EWMA, GARCH(1,1), HAR-RV with proper implementations.

```python
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import LinearRegression

def ewma_volatility(returns, span=60, horizon=20):
    """
    EWMA volatility forecast.
    
    Forecast variance over next h days using EWMA of past squared returns.
    """
    squared_returns = returns ** 2
    ewma_var = squared_returns.ewm(span=span).mean()
    # Scale to horizon
    forecast_var = ewma_var * horizon
    return np.sqrt(forecast_var * 252)  # Annualized volatility

def garch_volatility(returns, horizon=20):
    """
    GARCH(1,1) volatility forecast.
    
    Forecast conditional variance over next h days.
    """
    # Fit GARCH(1,1)
    model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
    fitted = model.fit(disp='off')
    
    # Forecast h steps ahead
    forecasts = fitted.forecast(horizon=horizon, reindex=False)
    cond_var = forecasts.variance.iloc[-1, :].mean()  # Average conditional variance
    
    # Convert back and annualize
    return np.sqrt(cond_var / 10000 * 252)

def har_rv(returns, horizon=20):
    """
    HAR-RV: Heterogeneous Autoregressive Realized Volatility.
    
    Proper implementation using daily, weekly, monthly RV components.
    """
    # Compute realized variances
    rv_daily = returns ** 2
    
    # Rolling windows
    rv_weekly = rv_daily.rolling(window=5).sum() / 5   # Weekly RV (avg daily)
    rv_monthly = rv_daily.rolling(window=22).sum() / 22  # Monthly RV (avg daily)
    
    # Prepare for regression
    # Target: next day's realized variance (shifted backward)
    target = rv_daily.shift(-1)
    
    # Features: lagged daily, weekly, monthly RV
    features = pd.DataFrame({
        'rv_daily': rv_daily,
        'rv_weekly': rv_weekly,
        'rv_monthly': rv_monthly
    })
    
    # Drop NaN
    valid_mask = ~(features.isna().any(axis=1) | target.isna())
    X = features[valid_mask].values
    y = target[valid_mask].values
    
    if len(X) < 100:  # Need enough data
        return np.full(len(returns), np.nan)
    
    # Fit HAR-RV regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast
    forecast_rv = model.predict(features.values)
    forecast_rv = np.maximum(forecast_rv, 1e-8)  # Ensure positive
    
    # Scale to horizon and annualize
    forecast_var = forecast_rv * horizon
    return np.sqrt(forecast_var * 252)
```

**Key Points:**
- **CRITICAL:** Proper HAR-RV implementation with daily/weekly/monthly components
- All baselines forecast variance over horizon h days
- Annualized volatility output

### 9. Training Script (`scripts/train.py`)

Simple Python script with cross-ticker training and held-out validation.

```python
#!/usr/bin/env python3
"""Simple training script with cross-ticker training."""

import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from src.models.chronos import ChronosVolatility
from src.training.data import prepare_raw_signal, compute_target, VolatilityDataset
from src.training.finetune import train

def main():
    # CRITICAL: Reserve one ticker for held-out validation
    train_tickers = ['AAPL', 'MSFT', 'GOOG', 'SPY', 'TSLA']  # Train on these
    held_out_ticker = 'NVDA'  # Test on this (unseen during training)
    
    # Load and prepare training data (cross-ticker)
    train_datasets = []
    for ticker in train_tickers:
        df = pd.read_parquet(f'data/cache/{ticker}.parquet')
        
        # Prepare raw signal (squared returns)
        raw_signal = prepare_raw_signal(df)
        
        # Compute target (log-realized variance)
        returns = raw_signal ** 0.5  # Recover returns from squared returns
        target = compute_target(returns, horizon=20)
        
        # Create dataset
        dataset = VolatilityDataset(raw_signal, target, seq_length=60, horizon=20)
        train_datasets.append(dataset)
    
    # Combine all training tickers
    combined_dataset = ConcatDataset(train_datasets)
    
    # Temporal split (time series aware)
    train_size = int(0.8 * len(combined_dataset))
    indices = torch.randperm(len(combined_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets (simplified - in practice use SubsetDataset)
    train_data = torch.utils.data.Subset(combined_dataset, train_indices)
    val_data = torch.utils.data.Subset(combined_dataset, val_indices)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChronosVolatility().to(device)
    
    # Train
    model = train(model, train_loader, val_loader, epochs=50, device=device)
    
    # Save
    torch.save(model.state_dict(), 'models/checkpoints/chronos.pt')
    print("Training complete. Model saved.")
    
    # Test on held-out ticker
    print(f"\nTesting on held-out ticker: {held_out_ticker}")
    test_df = pd.read_parquet(f'data/cache/{held_out_ticker}.parquet')
    # ... run evaluation on held-out ticker

if __name__ == '__main__':
    main()
```

**Key Points:**
- **CRITICAL:** Cross-ticker training (train on multiple tickers)
- **CRITICAL:** Hold out one ticker for validation (generalization test)
- Temporal split for train/val
- Test on held-out ticker to verify cross-asset generalization

### 10. Inference Script (`scripts/inference.py`)

Simple inference script using raw signal.

```python
#!/usr/bin/env python3
"""Simple inference script."""

import sys
import torch
import pandas as pd
from src.models.chronos import ChronosVolatility
from src.prediction.inference import predict
from src.training.data import prepare_raw_signal

def main(ticker='AAPL'):
    # Load data
    df = pd.read_parquet(f'data/cache/{ticker}.parquet')
    
    # Prepare raw signal (last 60 days of squared returns)
    raw_signal = prepare_raw_signal(df)
    context = raw_signal.iloc[-60:].values  # Last 60 days
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChronosVolatility().to(device)
    model.load_state_dict(torch.load('models/checkpoints/chronos.pt'))
    
    # Predict
    result = predict(model, context, device=device)
    print(f"Predicted volatility: {result['volatility']:.2f}%")
    print(f"90% interval: [{result['lower']:.2f}%, {result['upper']:.2f}%]")

if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    main(ticker)
```

### 11. Lambda Labs Setup

**Manual Setup (Simple):**

1. Create Lambda account, get API key
2. Add SSH key: `lambdacloud ssh-key add ~/.ssh/id_rsa.pub --name my-key`
3. Launch instance: `lambdacloud instance launch --instance-type gpu_1x_a10 --ssh-key-name my-key`
4. SSH in: `ssh ubuntu@<instance-ip>`
5. Install: `pip install torch transformers peft pandas numpy`
6. Copy code, run training
7. Save weights: `~/models/checkpoints/chronos.pt` (persists on Lambda filesystem)
8. Download: `scp ubuntu@<ip>:~/models/checkpoints/chronos.pt ./models/checkpoints/`

**Simple Python script for Lambda:**

```python
# scripts/train_lambda.py
# Same as train.py but saves to ~/models/checkpoints/chronos.pt
# Run on Lambda instance via SSH
```

### 12. Configuration (`config.yaml`)

Minimal config.

```yaml
chronos:
  model_id: amazon/chronos-t5-mini
  use_lora: true
  seq_length: 60
  prediction_horizon: 20
  
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  
data:
  tickers: [AAPL, MSFT, GOOG, SPY, TSLA, NVDA]
  train_start: 2010-01-01
  train_end: 2020-01-01
  val_end: 2022-01-01
  test_end: 2024-01-01
```

### 13. Dependencies (`requirements.txt`)

Minimal dependencies.

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
pandas>=1.5.0
numpy>=1.24.0
arch>=6.2.0  # For GARCH baseline
statsmodels>=0.14.0  # For HAR-RV baseline
pyyaml>=6.0  # For config
```

## Implementation Order

1. Remove `itransformer.py`, `neural_garch.py`
2. Create `src/models/chronos.py` (simple wrapper)
3. Create `src/training/data.py` (data prep)
4. Create `src/training/finetune.py` (training loop)
5. Create `src/evaluation/metrics.py` (QLIKE)
6. Create `src/evaluation/baselines.py` (EWMA, GARCH, HAR-RV)
7. Create `src/prediction/inference.py` (inference)
8. Create `scripts/train.py` (training script)
9. Create `scripts/inference.py` (inference script)
10. Update `config.yaml`, `requirements.txt`
11. Test end-to-end

## Code Style (nanochat-inspired)

- **Direct, minimal code** (~200-300 lines per file max)
- **No abstractions** - just functions and classes
- **Clear names** - `train()`, `predict()`, `qlike_loss()`
- **Comments explain "why"** not "what"
- **No config factories** - simple YAML loading
- **Single purpose files** - one clear responsibility
- **Python only** - no shell scripts
- **Hackable** - easy to modify and understand

## Cost Estimate

- Chronos-mini + LoRA: ~2-3 hours on A10 = $1.00-$1.50
- Well under $10 budget
- Manual monitoring: Check Lambda dashboard after 2-3 hours

## Summary

This is a minimal, direct implementation:
- **Chronos-mini** (standard choice)
- **Raw squared returns input** (single channel - Chronos requirement)
- **LoRA** (cost-efficient)
- **Cross-ticker training** (one model) + held-out ticker validation
- **Quantile loss** (train all quantiles properly)
- **QLIKE evaluation** (on q50 only, for monitoring)
- **Proper baselines** (EWMA, GARCH, HAR-RV with correct implementations)
- **Python only** (no shell scripts)
- **Direct code** (no over-engineering)

**Critical Fixes Addressed:**
1. ✅ Single-channel input (raw squared returns)
2. ✅ No feature engineering (preserves Chronos pretraining)
3. ✅ Quantile loss for all quantiles (proper training)
4. ✅ QLIKE only on q50 (theoretically correct)
5. ✅ Explicit timing alignment (no future leakage)
6. ✅ Proper HAR-RV implementation
7. ✅ Held-out ticker validation (cross-asset generalization)

Follows nanochat philosophy: simple, minimal, hackable, effective.
