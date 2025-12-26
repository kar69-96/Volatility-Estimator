# Volatility Estimator Stack

Built this as a compact volatility analytics “workbench”: pull OHLC data, run multiple volatility estimators over configurable horizons, compare them side-by-side, and (optionally) look at event windows and simple pattern-based forecasts.

Meant to be easy to explore interactively (Streamlit) but also usable from the command line for repeatable runs.

## What’s inside

- **Data pipeline**: fetch OHLC data via `yfinance`, validate it, and cache locally as **Parquet** for fast reruns.
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

streamlit run src/app.py
```

Or:

```bash
./run_app.sh
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

## Project layout

```
data/
  cache/                  # Cached market data (parquet)
  events/                 # Economic calendar CSV
src/
  app.py                  # Streamlit UI
  data_loader.py          # API integration, caching, validation
  estimators/             # Volatility estimators (base + implementations)
  comparison.py           # Run-all + correlation/MSE/stat summaries
  event_analysis.py       # Event window analytics
  predictions.py          # Pattern-based prediction utilities
  reporting.py            # CSV/Excel report generation
  run.py                  # CLI entry point
tests/
  unit/                   # Unit tests for core components
  integration/            # End-to-end pipeline tests
config.yaml               # Defaults (assets, date ranges, caching, logging)
requirements.txt          # Dependencies
```

## Notes on the estimators

- **Close-to-Close**: baseline realized volatility from close-to-close returns.
- **EWMA**: exponentially weighted variance/volatility (good for “recent shock” sensitivity).
- **Parkinson / Rogers–Satchell / Yang–Zhang**: range-based estimators that use OHLC structure to improve efficiency vs. close-only approaches.

## Testing

```bash
pytest -q
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

## Troubleshooting

- **Port already in use**: `streamlit run src/app.py --server.port 8502`
- **First run is slow**: expected (it’s downloading + caching data). Subsequent runs use Parquet cache.
- **macOS + Python 3.9**: upgrade to Python 3.10+ for Streamlit stability (the CLI/core libraries work, but the UI can fail).

