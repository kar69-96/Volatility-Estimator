# Volatility Estimator Stack

Built this as a compact volatility analytics “workbench”: pull OHLC data, run multiple volatility estimators over configurable horizons, compare them side-by-side, and (optionally) look at event windows and simple pattern-based forecasts.

Meant to be easy to explore interactively (Streamlit) but also usable from the command line for repeatable runs.

## What’s inside

- **Data pipeline**: fetch OHLC data via `yfinance`, validate it, and cache locally as Parquet for fast reruns.
- **Volatility estimators (5)**:
  - **Close-to-Close**
  - **EWMA** (RiskMetrics-style smoothing)
  - **Parkinson**
  - **Rogers–Satchell**
  - **Yang–Zhang**
- **Comparison framework**: run all estimators on the same dataset and compute helpful diagnostics (e.g., correlations / MSE).
- **Event analysis & predictions**: utilities to study volatility around known events and generate simple forward-looking paths.
- **Outputs**: CSV/Excel exports and plots for quick sharing.

## Requirements

- **Python 3.10+**
  - On macOS, Python 3.9 can break Streamlit’s WebSocket connection due to a known asyncio/kqueue selector issue. Upgrading to 3.10+ fixes it.
- Dependencies are listed in `requirements.txt`.

## Quick start (Streamlit)

```bash
python3.10 -m venv venv310
source venv310/bin/activate
pip install -r requirements.txt

streamlit run frontend/app.py
```

Or use the provided script:

```bash
./scripts/run_app.sh
```

Then open `http://localhost:8501`.

## Quick start (CLI)

Single estimator:

```bash
python src/run.py --symbol SPY --estimator yang_zhang --window 60
```

Compare all estimators:

```bash
python src/run.py --symbol SPY --compare --window 60
```

Event window analysis:

```bash
python src/run.py --symbol SPY --events --event_window 5 --window 60
```

Run “full” workflow (comparison + events + predictions + reports):

```bash
python src/run.py --symbol SPY --compare --events --predict --window 60 --output_dir ./outputs
```

## How the pipeline is structured

At a high level:

1. **Retrieve** market data (API)
2. **Validate** (shape, missing values, date range consistency)
3. **Cache** to `data/cache/` (Parquet)
4. **Compute** returns + estimator-specific rolling measures
5. **Compare / analyze / export**

The estimator interface is intentionally small: each estimator implements `calculate()` and inherits validation + annualization behavior from a shared base class.


## Notes on the estimators

- **Close-to-Close**: baseline realized volatility from close-to-close returns.
- **EWMA**: exponentially weighted variance/volatility (good for “recent shock” sensitivity).
- **Parkinson / Rogers–Satchell / Yang–Zhang**: range-based estimators that use OHLC structure to improve efficiency vs. close-only approaches.

## Deep Learning Models

The stack now includes PyTorch-based deep learning models for advanced volatility prediction:

### Models Available

1. **iTransformer Volatility Predictor**
   - Inverted transformer architecture (tokens = features, not time steps)
   - Multi-horizon prediction (1, 5, 10, 20 days ahead)
   - Automatic feature extraction from OHLC data

2. **Neural GARCH**
   - Neural network-based conditional variance estimation
   - Learns nonlinear GARCH dynamics
   - Extends BaseEstimator for seamless integration

3. **Fed Rate Predictor** (LSTM/Transformer)
   - Predicts Fed rate direction (Increase/Decrease/No Change)
   - Optional magnitude prediction in basis points

### GPU Support

- Automatic device detection (CUDA > MPS > CPU)
- Mixed precision training for faster GPU training
- Works on CPU if no GPU available

### Quick Start (Deep Learning)

```python
from src.predictions import predict_volatility_dl, predict_neural_garch

# iTransformer prediction
result = predict_volatility_dl(df, device='auto')
print(result['predictions'])  # {'1d': 15.2, '5d': 16.8, ...}

# Neural GARCH
result = predict_neural_garch(df, p=1, q=1, epochs=100)
print(result['current_volatility'])  # 18.5%
```

### Training Models

```python
from src.predictions import train_dl_models

# Train volatility predictor
result = train_dl_models(
    df, 
    model_type='volatility',
    epochs=100,
    device='cuda'
)
print(f"Model saved to: {result['model_path']}")
```

### Configuration

Deep learning settings are in `config.yaml`:

```yaml
deep_learning:
  device: auto
  batch_size: 64
  mixed_precision: true

volatility_predictor:
  model_type: itransformer
  d_model: 128
  nhead: 8
  num_layers: 4
```

### Requirements

For deep learning features, install PyTorch:

```bash
# CPU only
pip install torch

# With CUDA (GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation of the repository organization.

## Testing

```bash
pytest -q
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

## Credits & Attribution

Deep learning implementation leverages concepts from:
- iTransformer paper (arXiv:2310.06625)
- PyTorch ecosystem
- Stock Transformers repository (architecture reference)
- Transformers Predictions dashboard (UI reference)

