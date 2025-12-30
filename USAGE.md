# Usage Guide

This document provides comprehensive documentation for all commands available in the Volatility Estimator Stack, including their arguments, outputs, and return values.

## Main CLI Command

### `cli/run.py`

The main command-line interface for volatility estimation, analysis, and prediction.

#### Basic Syntax

```bash
python cli/run.py --symbol <SYMBOL> [OPTIONS]
```

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--symbol` | string | Asset symbol (e.g., SPY, AAPL, TSLA, QQQ) |

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--estimator` | string | `yang_zhang` | Volatility estimator: `close_to_close`, `ewma`, `parkinson`, `rogers_satchell`, `yang_zhang` |
| `--compare` | flag | False | Run all estimators and compare results |
| `--window` | integer | 60 | Rolling window size in trading days |
| `--lambda` | float | 0.94 | EWMA decay parameter (only used with EWMA estimator) |
| `--events` | flag | False | Run event analysis around economic calendar events |
| `--event-window` | integer | 5 | Days before/after event for analysis |
| `--predict` | flag | False | Generate pattern-based volatility predictions for upcoming events |
| `--predict-chronos` | flag | False | Use Chronos transformer model for volatility predictions |
| `--prediction-window` | integer | 20 | Prediction horizon in days (used with `--predict-chronos`) |
| `--output_dir` | string | None | Directory for CSV, Excel, and chart outputs |
| `--excel` | flag | False | Generate Excel report (auto-enabled if `--output_dir` specified) |
| `--config` | string | `config.yaml` | Path to configuration file |
| `--verbose` | flag | False | Enable detailed logging |

---

## Command Examples and Outputs

### 1. Single Estimator Calculation

**Command:**
```bash
python cli/run.py --symbol SPY --estimator yang_zhang --window 60
```

**Console Output:**
```
============================================================
Volatility Estimation Results: SPY
============================================================
Estimator: yang_zhang
Window: 60 days
Date Range: 2004-01-15 to 2024-12-31
Total Estimates: 5257

Summary Statistics (Annualized %):
  Mean:   18.45%
  Std:    8.32%
  Min:    4.21%
  Max:    89.34%
============================================================
```

**Files Generated (if `--output_dir` specified):**
- `{symbol}_{estimator}_volatility.csv`: Time series of volatility estimates with columns: `date`, `volatility`

**Returns:** Console summary statistics only (unless `--output_dir` is specified)

---

### 2. Compare All Estimators

**Command:**
```bash
python cli/run.py --symbol SPY --compare --window 60
```

**Console Output:**
```
======================================================================
Volatility Estimator Comparison: SPY
======================================================================
Window: 60 days
Date Range: 2004-01-15 to 2024-12-31
Total Estimates: 5257

----------------------------------------------------------------------
Summary Statistics (Annualized %):
----------------------------------------------------------------------
Estimator              Mean      Std      Min      Max
----------------------------------------------------------------------
close_to_close       18.32     8.45     4.12    89.21
ewma                 18.51     8.28     4.25    88.95
parkinson            18.28     8.39     4.18    89.05
rogers_satchell      18.35     8.41     4.20    89.12
yang_zhang           18.45     8.32     4.21    89.34

----------------------------------------------------------------------
Correlation Matrix:
----------------------------------------------------------------------
                   close_to_close    ewma  parkinson  rogers_satchell  yang_zhang
close_to_close              1.000  0.987     0.992             0.988       0.995
ewma                        0.987  1.000     0.983             0.985       0.991
parkinson                   0.992  0.983     1.000             0.997       0.998
rogers_satchell             0.988  0.985     0.997             1.000       0.996
yang_zhang                  0.995  0.991     0.998             0.996       1.000

----------------------------------------------------------------------
Mean Squared Error Matrix:
----------------------------------------------------------------------
                   close_to_close    ewma  parkinson  rogers_satchell  yang_zhang
close_to_close              0.000  0.125     0.089             0.112       0.045
ewma                        0.125  0.000     0.134             0.098       0.087
parkinson                   0.089  0.134     0.000             0.023       0.034
rogers_satchell             0.112  0.098     0.023             0.000       0.041
yang_zhang                  0.045  0.087     0.034             0.041       0.000

======================================================================
```

**Files Generated (if `--output_dir` specified):**
- `{symbol}_comparison_volatility.csv`: DataFrame with columns `date`, `close_to_close`, `ewma`, `parkinson`, `rogers_satchell`, `yang_zhang`

**Returns:** 
- Correlation matrix comparing all estimators
- MSE (Mean Squared Error) matrix
- Summary statistics for each estimator

---

### 3. Event Analysis

**Command:**
```bash
python cli/run.py --symbol SPY --events --event-window 10 --window 60
```

**Console Output:**
```
------------------------------------------------------------
Event Impact Analysis:
------------------------------------------------------------
Total Events Analyzed: 142
Significant Events: 89 (62.7%)
Average Volatility Change: 12.34%

Top 5 Most Impactful Events:
  2020-03-15 (FOMC Meeting): 45.67%*
  2008-10-08 (Emergency Rate Cut): 38.92%*
  2020-03-23 (Fed QE Announcement): 32.15%*
  2011-08-05 (S&P Downgrade): 28.43%*
  2008-09-15 (Lehman Collapse): 27.89%*
------------------------------------------------------------
```

**Files Generated (if `--output_dir` specified):**
- `{symbol}_event_analysis.csv`: Event analysis results with columns: `event_date`, `event_type`, `pre_vol`, `post_vol`, `volatility_change_pct`, `significant`

**Returns:**
- Number of events analyzed
- Count and percentage of significant events
- Average volatility change
- Top 5 most impactful events with volatility change percentages

---

### 4. Pattern-Based Predictions

**Command:**
```bash
python cli/run.py --symbol SPY --predict --events --window 60
```

**Console Output:**
```
------------------------------------------------------------
Prediction Accuracy (Backtesting):
------------------------------------------------------------
Mean Absolute Error: 3.45%
Root Mean Squared Error: 5.12%
Correlation: 0.782
------------------------------------------------------------
```

**Files Generated (if `--output_dir` specified):**
- `{symbol}_predictions.csv`: Predicted volatility paths for upcoming events
- `charts/{symbol}_predictions.png`: Visualization of predictions

**Returns:**
- Backtest metrics (MAE, RMSE, correlation)
- Predicted volatility paths for upcoming events (if any)

---

### 5. Chronos Deep Learning Predictions

**Command:**
```bash
python cli/run.py --symbol TSLA --predict-chronos --prediction-window 30
```

**Console Output:**
```
======================================================================
Chronos Volatility Prediction: TSLA
======================================================================
Prediction Window: 30 days
Model: karkar69/chronos-volatility
Device: cuda

----------------------------------------------------------------------
Predicted Volatility (Annualized %):
----------------------------------------------------------------------
Date         Volatility Lower (q10) Upper (q90)
----------------------------------------------------------------------
2024-12-16       45.23        38.12        54.87
2024-12-17       44.89        37.95        54.21
2024-12-18       44.56        37.78        53.65
...
2025-01-15       42.15        35.42        51.23
======================================================================
```

**Files Generated (if `--output_dir` specified):**
- `{symbol}_chronos_predictions.csv`: DataFrame with columns `date`, `volatility`, `lower`, `upper`

**Returns:**
- Probabilistic forecasts (q10, q50, q90 quantiles) for specified prediction window
- Model device information (cuda/cpu/mps)

**Note:** This command exits early and does not run other estimators or analyses.

---

### 6. Complete Workflow

**Command:**
```bash
python cli/run.py --symbol QQQ \
  --compare \
  --events \
  --predict \
  --output_dir ./results \
  --excel \
  --verbose
```

**Console Output:**
- All outputs from comparison, event analysis, and predictions (as shown above)

**Files Generated:**
- `{symbol}_comparison_volatility.csv`: All estimators comparison
- `{symbol}_event_analysis.csv`: Event impact analysis
- `{symbol}_predictions.csv`: Pattern-based predictions
- `{symbol}_volatility_report.xlsx`: Multi-sheet Excel report containing:
  - Volatility estimates (all estimators)
  - Event analysis results
  - Prediction results
  - Backtest metrics
- `charts/{symbol}_volatility_comparison.png`: Comparison chart
- `charts/{symbol}_predictions.png`: Predictions visualization
- `summaries/{symbol}_summary.txt`: Text summary report

**Returns:** Comprehensive analysis combining all features

---

## Training Scripts

### `scripts/train_sample.py`

Trains Chronos model on a sample dataset (5 tickers: AAPL, MSFT, GOOG, SPY, TSLA).

**Command:**
```bash
python scripts/train_sample.py
```

**Console Output:**
```
Training on 5 tickers: ['AAPL', 'MSFT', 'GOOG', 'SPY', 'TSLA']
Test ticker: NVDA

Loading training data...
  ✓ AAPL: 2,523 samples
  ✓ MSFT: 2,523 samples
  ✓ GOOG: 2,523 samples
  ✓ SPY: 2,523 samples
  ✓ TSLA: 2,523 samples

✓ Total training samples: 12,615

Using device: cuda

Starting training...
[Training progress logs...]

Training complete. Model saved to models/checkpoints/chronos_5ticker.pt
```

**Files Generated:**
- `models/checkpoints/chronos_5ticker.pt`: Trained model checkpoint

**Returns:** Training progress logs and final checkpoint path

---

### `scripts/train_sp500.py`

Trains Chronos model on full S&P 500 dataset.

**Command:**
```bash
python scripts/train_sp500.py
```

**Console Output:** Similar to `train_sample.py` but with progress for all S&P 500 tickers

**Files Generated:**
- `models/checkpoints/chronos_sp500.pt`: Trained model checkpoint

**Returns:** Training progress for all S&P 500 stocks

---

## Inference Script

### `scripts/inference.py`

Runs inference using a trained model checkpoint.

**Command:**
```bash
python scripts/inference.py AAPL
```

**Console Output:**
```
Predicted volatility for AAPL:
  Point estimate (q50): 18.45%
  90% interval: [15.23%, 22.67%]
  Log-variance (q50): 2.9154
```

**Returns:** 
- Point estimate (median q50)
- 90% confidence interval (q10, q90)
- Log-variance value

**Note:** Requires cached data file at `data/cache/{ticker}.parquet` and trained model at `models/checkpoints/chronos.pt`

---

## Model Export Script

### `scripts/export_to_huggingface.py`

Exports trained model to Hugging Face format.

**Command:**
```bash
python scripts/export_to_huggingface.py \
  --checkpoint models/checkpoints/chronos_5ticker.pt \
  --output models/huggingface/chronos-volatility \
  --hub-repo-id username/model-name
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | string | `models/checkpoints/chronos_5ticker.pt` | Path to model checkpoint |
| `--output` | string | `models/huggingface/chronos-volatility` | Output directory |
| `--name` | string | None | Model name (default: infer from output dir) |
| `--description` | string | None | Model description |
| `--no-push` | flag | False | Skip pushing to Hugging Face Hub |
| `--hub-repo-id` | string | `karkar69/chronos-volatility` | Hugging Face Hub repository ID |

**Console Output:**
```
Loading model from models/checkpoints/chronos_5ticker.pt...
✓ Model loaded
Saving LoRA adapters to models/huggingface/chronos-volatility/adapter...
✓ LoRA adapters saved
Saving custom heads to models/huggingface/chronos-volatility/heads.pt...
✓ Custom heads saved
Saving config to models/huggingface/chronos-volatility/config.json...
✓ Config saved
Saving model card to models/huggingface/chronos-volatility/README.md...
✓ Model card saved
✓ Example usage script saved

============================================================
Export Complete!
============================================================
Model exported to: models/huggingface/chronos-volatility

Contents:
  - adapter/       : LoRA adapter weights
  - heads.pt       : Custom head weights (quantile_head, value_embedding)
  - config.json    : Model configuration
  - README.md      : Model card
  - example_usage.py: Example loading script

Pushing to Hugging Face Hub: username/model-name...
✓ Successfully pushed to https://huggingface.co/username/model-name
```

**Files Generated:**
- `{output_dir}/adapter/`: LoRA adapter weights
- `{output_dir}/heads.pt`: Custom head weights
- `{output_dir}/config.json`: Model configuration
- `{output_dir}/README.md`: Model card documentation
- `{output_dir}/example_usage.py`: Example loading script

**Returns:** Export status and Hugging Face Hub URL (if pushed)

---

## Output File Formats

### CSV Files

All CSV outputs use comma separation and include a header row:

- **Volatility Time Series**: `date`, `volatility` (annualized %)
- **Comparison Results**: `date`, `close_to_close`, `ewma`, `parkinson`, `rogers_satchell`, `yang_zhang`
- **Event Analysis**: `event_date`, `event_type`, `pre_vol`, `post_vol`, `volatility_change_pct`, `significant`
- **Chronos Predictions**: `date`, `volatility`, `lower`, `upper` (all annualized %)

### Excel Reports

Multi-sheet Excel files (`.xlsx`) containing:
- **Volatility Estimates**: All estimator results (if comparison mode)
- **Event Analysis**: Event impact results
- **Predictions**: Forecast results
- **Backtest Metrics**: Prediction accuracy metrics

### Chart Files

PNG format charts generated using Plotly:
- Volatility comparison over time
- Predictions with confidence intervals
- Event markers on time series

### Summary Reports

Plain text files containing:
- Summary statistics
- Key findings
- Model information
- Configuration used

---

## Error Handling

All commands return appropriate exit codes:

- **Exit 0**: Success
- **Exit 1**: Error (invalid input, missing data, model failure, etc.)

Error messages are printed to stderr and logged (if `--verbose` is enabled).

Common errors:
- **Missing data**: Symbol not found or insufficient historical data
- **Invalid estimator**: Estimator name not recognized
- **Missing model**: Required model checkpoint not found (for inference/predictions)
- **Configuration error**: Invalid config file or missing required settings

---

## Configuration Overrides

Command-line arguments override settings in `config.yaml`. For example:

- `--window 30` overrides `config.volatility.default_window`
- `--lambda 0.96` overrides `config.volatility.ewma_lambda`
- `--event-window 10` overrides `config.events.pre_window`

This allows flexibility without modifying configuration files.

