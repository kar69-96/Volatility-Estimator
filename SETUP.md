# Setup Guide

For detailed command documentation and output examples, see [USAGE.md](USAGE.md).

## Requirements

- **Python 3.10+**
- Dependencies are listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Volatility-Estimator

# Create virtual environment
python3.10 -m venv venv310
source venv310/bin/activate  # On Windows: venv310\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional but recommended for deep learning features)
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended for training):
pip install torch torchvision torchaudio
```

### GPU Setup

- **Check CUDA Installation**:
  ```bash
  nvidia-smi  # Check NVIDIA driver and GPU
  python -c "import torch; print(torch.cuda.is_available())"  # Check PyTorch CUDA support
  python -c "import torch; print(torch.cuda.get_device_name(0))"  # Check GPU name
  ```

- **CUDA Version Compatibility**:
  - CUDA 11.8: Use PyTorch with `cu118` index
  - CUDA 12.1: Use PyTorch with `cu121` index
  - Match your CUDA toolkit version with the PyTorch CUDA build

- **Device Selection**:
  - The toolkit automatically uses CUDA when available (`device='auto'` is default)
  - Explicitly specify device in code if needed: `device='cuda'` for GPU, `device='cpu'` for CPU

- **Performance Expectations**:
  - **Inference (predictions)**: GPU: seconds, CPU: minutes
  - **Training (5 tickers)**: GPU (A100): ~1-2 hours, CPU: days
  - **Training (S&P 500)**: GPU (A100): several hours, CPU: weeks

## Quick Start

**Basic volatility estimation:**
```bash
python cli/run.py --symbol SPY --estimator yang_zhang --window 60
```

**Compare all estimators:**
```bash
python cli/run.py --symbol SPY --compare --window 60
```

**Full analysis with outputs:**
```bash
python cli/run.py --symbol SPY --compare --events --predict --output_dir ./outputs
```

## CLI Usage

### Main Command

```bash
python cli/run.py --symbol <SYMBOL> [OPTIONS]
```

### Required Arguments

- `--symbol`: Asset symbol (e.g., SPY, AAPL, TSLA)

### Common Options

| Option | Description |
|--------|-------------|
| `--estimator` | Volatility estimator: `close_to_close`, `ewma`, `parkinson`, `rogers_satchell`, `yang_zhang` |
| `--compare` | Run all estimators and compare results |
| `--window` | Rolling window size in trading days (default: 60) |
| `--events` | Run event analysis around economic calendar events |
| `--event-window` | Days before/after event (default: 5) |
| `--predict` | Generate pattern-based volatility predictions |
| `--predict-chronos` | Use Chronos transformer model for predictions |
| `--prediction-window` | Prediction horizon in days (default: 20) |
| `--output_dir` | Directory for CSV, Excel, and chart outputs |
| `--excel` | Generate Excel report |
| `--config` | Path to config file (default: config.yaml) |
| `--verbose` | Enable detailed logging |
| `--lambda` | EWMA decay parameter (default: 0.94) |

### Usage Examples

**Single estimator with custom window:**
```bash
python cli/run.py --symbol AAPL --estimator ewma --window 30 --lambda 0.96
```

**Event impact analysis:**
```bash
python cli/run.py --symbol SPY --events --event-window 10 --window 60
```

**Chronos deep learning predictions:**
```bash
python cli/run.py --symbol TSLA --predict-chronos --prediction-window 30
```

**Complete workflow with all outputs:**
```bash
python cli/run.py --symbol QQQ \
  --compare \
  --events \
  --predict \
  --output_dir ./results \
  --excel \
  --verbose
```

## Training Models

The codebase includes scripts for training Chronos models on custom datasets:

### Pre-trained Model (5 Stocks)

A pre-trained model is available on Hugging Face for immediate use:
- **Model**: [karkar69/chronos-volatility](https://huggingface.co/karkar69/chronos-volatility)
- **Training Data**: Fine-tuned on sample dataset (AAPL, GOOG, MSFT, SPY, TSLA)
- **Base Model**: amazon/chronos-t5-mini with LoRA adaptation
- **Input**: 60 days of historical squared returns
- **Output**: Quantile predictions (q10, q50, q90) for 20-day volatility forecast
- **Usage**: The `--predict-chronos` CLI option automatically downloads and uses this model

### Training Custom Models

**Quick training (5 tickers)** - Recommended for testing:
```bash
python scripts/train_sample.py
```
This trains on the same dataset as the pre-trained model (AAPL, GOOG, MSFT, SPY, TSLA).

**Full S&P 500 training** - For production models:
```bash
python scripts/train_sp500.py
```
Requires CUDA-enabled GPU (A100 recommended). Training takes several hours.

**Inference with trained model:**
```bash
python scripts/inference.py AAPL
```

**Export model to Hugging Face:**
```bash
python scripts/export_to_huggingface.py \
  --checkpoint models/checkpoints/chronos_5ticker.pt \
  --output models/huggingface/chronos-volatility \
  --hub-repo-id your-username/model-name
```

**Note**: Training requires CUDA-enabled GPU for practical use. CPU training is possible but extremely slow (days to weeks). See `docs/LOCAL_TRAINING.md` and `docs/LAMBDA_SETUP.md` for detailed training guides.

## Configuration

Settings are managed in `config.yaml`:

```yaml
data:
  start_date: 2004-01-01
  end_date: 2024-12-31
  cache_dir: ./data/cache

volatility:
  default_estimator: yang_zhang
  default_window: 60
  ewma_lambda: 0.94

events:
  csv_path: ./data/events/economic_calendar.csv
  pre_window: 5
  post_window: 5

chronos:
  model_id: amazon/chronos-t5-mini
  use_lora: true
  seq_length: 252
  prediction_horizon: 20
```

## Testing

Run tests:
```bash
pytest -q
```

With coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

