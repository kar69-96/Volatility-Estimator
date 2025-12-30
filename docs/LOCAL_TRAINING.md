# Running Training Locally

This guide explains how to run `scripts/train_sample.py` on your local machine.

## Prerequisites

1. **Python 3.10+** (you have `venv310` which should be Python 3.10)
2. **Virtual environment activated**
3. **All dependencies installed**

## Setup Steps

### 1. Activate Virtual Environment

```bash
cd /Users/karthikreddy/Downloads/GitHub/Demos/Volatility-Estimator
source venv310/bin/activate
```

### 2. Verify/Install Dependencies

The script requires:
- PyTorch (already installed: 2.9.1)
- transformers
- peft (for LoRA)
- pandas, numpy
- yfinance (for data download)
- Other dependencies from `requirements.txt`

If you need to install missing dependencies:

```bash
# Install PyTorch (if not already installed)
# For CPU only (macOS):
pip install torch torchvision torchaudio

# For CUDA (Linux/Windows with GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 3. Run Training

Simply execute the script:

```bash
python scripts/train_sample.py
```

Or make it executable and run directly:

```bash
chmod +x scripts/train_sample.py
./scripts/train_sample.py
```

## What the Script Does

1. **Downloads data** for 5 training tickers: `AAPL`, `MSFT`, `GOOG`, `SPY`, `TSLA`
   - Data is cached in `data/cache/` as Parquet files
   - Uses last 10 years of data

2. **Prepares training data**:
   - Computes log returns
   - Creates volatility targets (20-day horizon)
   - Creates sequences of 252 days (1 year)

3. **Trains the model**:
   - Uses Chronos-T5-mini with LoRA adapters
   - 50 epochs with learning rate 1e-5
   - 80/20 train/validation split
   - Batch size: 32

4. **Saves model** to `models/checkpoints/chronos_5ticker.pt`

5. **Tests on held-out ticker**: `NVDA`

## Expected Output

The script will:
- Print progress for each ticker being loaded
- Show total training samples
- Display training progress (epochs, loss, etc.)
- Save the trained model checkpoint
- Load test data for NVDA

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Install missing dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory" or slow training
**Solution**: The script automatically uses CPU if CUDA is not available. On macOS, it will use MPS (Metal Performance Shaders) if available, otherwise CPU. Training on CPU will be slower but should work.

### Issue: Data download fails
**Solution**: Check internet connection. The script uses `yfinance` to download data. If it fails, you can manually download data or check if cached data exists in `data/cache/`.

### Issue: "No training datasets loaded"
**Solution**: Ensure data files exist in `data/cache/` or that the script can download them. Check that tickers are valid and data is available for the date range.

## Customization

You can modify the script to:
- Change training tickers (line 246)
- Change test ticker (line 247)
- Adjust epochs (line 299)
- Adjust learning rate (line 299)
- Change batch size (line 287)
- Modify date range (lines 257-258)

## Notes

- The script is designed to work on both Lambda Labs instances and local machines
- It will automatically detect if running on Lambda and handle instance termination
- On local machines, it will skip Lambda-specific features
- Training time depends on your hardware (CPU/GPU) and data size
- The model uses LoRA for efficient fine-tuning, reducing memory requirements

