# Volatility Estimator Stack

Stock volatility analysis and forecasting. Calculates and analyzes stock market volatility using multiple estimators (Close-to-Close, EWMA, Parkinson, Rogers-Satchell, Yang-Zhang) and forecasts future volatility for horizon windows using a fine-tuned (NVIDIA A100) volatility model trained on stocks included in the S&P 500.

It combines traditional statistical estimators with modern deep learning approaches to analyze and forecast stock market volatility patterns.

## Features

### Deep Learning & Forecasting

- **Chronos Transformer Predictions**: Volatility forecasting using fine-tuned Chronos transformer models trained on S&P 500 data. Provides probabilistic forecasts (q10, q50, q90 quantiles) for configurable prediction horizons (default: 20 trading days). 

- **Pattern-Based Forecasts**: Generate volatility predictions by identifying and matching historical similar events from economic calendars. Uses pattern matching algorithms to find comparable market conditions and extrapolate volatility paths.

- **Training Infrastructure**: Full pipeline for fine-tuning Chronos models on custom datasets with:
  - LoRA (Low-Rank Adaptation) for parameter-efficient training
  - Support for NVIDIA A100 GPU instances (Lambda Labs integration) with CUDA acceleration
  - Model export to Hugging Face Hub format
  - Weighted sampling to emphasize recent market data
  - Pre-trained sample model (only trained on 8 stocks): [karkar69/chronos-volatility](https://huggingface.co/karkar69/chronos-volatility)


### Volatility Estimators

Five traditional statistical estimators are implemented for historical volatility estimation:

1. **Close-to-Close**: Standard realized volatility from daily close-to-close returns
2. **EWMA**: Exponentially weighted moving average (RiskMetrics-style) with configurable decay parameter
3. **Parkinson**: Range-based estimator using high/low prices (more efficient than close-to-close)
4. **Rogers-Satchell**: Range-based estimator accounting for drift bias
5. **Yang-Zhang**: Comprehensive range-based estimator combining multiple components (recommended for accuracy)

### Analysis Capabilities

- **Estimator Comparison**: Run all estimators simultaneously and compute correlation matrices, MSE comparisons, and summary statistics to understand estimator relationships and performance
- **Event Analysis**: Analyze volatility impact around economic calendar events with statistical significance testing to quantify how events affect market volatility

### Output Formats

- Console output with summary statistics
- CSV exports for time series data
- Excel reports with multiple sheets (volatility estimates, events, predictions)
- Plotly-generated charts (saved as PNG)
- Text-based summary reports

## Architecture

### Data Pipeline

The system follows a clean data pipeline architecture:

1. **Fetch**: Download OHLC data via `yfinance` API
2. **Validate**: Check data quality (missing values, date consistency, shape validation)
3. **Cache**: Store validated data as Parquet files for fast subsequent runs
4. **Compute**: Calculate returns and estimator-specific rolling measures
5. **Analyze**: Run comparisons, event analysis, and predictions
6. **Export**: Generate reports and visualizations

### Estimator Design

All volatility estimators inherit from a common `BaseEstimator` class that provides:
- Input validation (window size, data requirements)
- Annualization handling (configurable trading days per year)
- Consistent interface (`calculate()` method)
- Error handling and data quality checks

This design allows for easy extension and comparison across estimators while maintaining code consistency.

### Deep Learning Architecture

The deep learning components use:
- **Base Model**: Amazon Chronos T5 (transformer-based time series foundation)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) for parameter-efficient training
- **Task Head**: Custom quantile regression head (q10, q50, q90) for probabilistic forecasts
- **Training Data**: S&P 500 stocks with weighted sampling to emphasize recent data
- **Sequence Length**: 252 trading days (1 year) for context
- **Prediction Horizon**: 20 trading days ahead


## Notes

### Estimator Selection

- **Yang-Zhang** is recommended as the default estimator due to its comprehensive use of OHLC data and superior efficiency compared to close-to-close methods
- **EWMA** is useful when recent volatility shocks should be weighted more heavily (financial crisis scenarios)
- **Parkinson** and **Rogers-Satchell** are intermediate options that use range data but may have limitations with drift

### Data Considerations

- Market data is cached locally as Parquet files to avoid repeated API calls
- Default date range (2004-2024) can be configured in `config.yaml`
- Data quality validation ensures missing values and inconsistencies are detected early
- Economic calendar events are loaded from CSV (can be extended to FRED API)

### Model Limitations

- The Chronos model is fine-tuned on S&P 500 stocks and may not generalize well to:
  - International markets
  - Different asset classes (bonds, commodities)
  - Extremely volatile periods outside training distribution
- Pattern-based predictions rely on historical similarity matching and assume past patterns will repeat
- All predictions are probabilistic (quantiles) and should be interpreted with appropriate uncertainty

### Performance

- Chronos predictions require GPU for reasonable inference time:
  - **CUDA (NVIDIA GPU)**: Recommended for best performance 
  - **CPU**: Works but slow 
- Training on full S&P 500 requires GPU (I trained on A100) and several hours (12+):
  - CUDA-enabled GPUs allow mixed precision training (FP16) for 2x speedup
  - Training benefits significantly from GPU memory bandwidth and parallel processing
- Data caching significantly speeds up repeated analysis on the same symbols

### GPU/CUDA Requirements

Deep learning features (Chronos predictions and training) benefit significantly from CUDA-enabled NVIDIA GPUs:

- **CUDA Support**: The toolkit automatically detects and uses CUDA when available
- **CUDA Versions**: Compatible with CUDA 11.8+ and 12.1+ (install matching PyTorch version)
- **Performance**: GPU inference is 10-100x faster than CPU, and training requires GPU for practical use
- **Mixed Precision**: Automatic FP16 mixed precision training when CUDA is available, reducing memory usage and increasing speed
- **Device Selection**: Use `device='auto'` in code (default) for automatic CUDA detection, or explicitly set `device='cuda'` for GPU or `device='cpu'` for CPU-only

## AI Usage

AI was used for supporting and supplementary tasks throughout the project:

- **Debugging**: AI (primarily Sonnet 4.5) was used for debugging all types of failures. Very useful in debugging issues with finetuning scripts, which was new to me.
- **Error Handling**: Used to create edge cases and fallbacks if primary approach does not work. LLMs (mainly Codex-mini) were very good at thinking through potential errors and providing fallbacks. 
- **Docs**: Used to compile information and create docs (README, SETUP.md, USAGE.md, LAMBDA_SETUP.md) and init files. Used Gemini Flash for this. 
- **Configurations**: Had trouble configuring complicated files, such as setup_lambda.sh or run.py for CLI commands. Opus 4.5 and Codex Max were useful in taking care of tedious tasks so I can focus on backend logic. 

## Drawbacks

1. **Limited Asset Coverage**: Training data focuses on S&P 500 stocks; predictions on international or smaller-cap stocks may be less reliable

2. **Historical Bias**: Pattern-based predictions assume historical patterns repeat, which may not hold during unprecedented market conditions

3. **Event Dependency**: Event analysis requires accurate economic calendar data; missing or incorrect event dates will affect results

4. **Computational Requirements**: Deep learning features require GPU for practical use; CPU inference is possible but slow

5. **Single-Symbol Workflow**: CLI processes one symbol at a time; batch processing requires scripting multiple calls

6. **Model Maintenance**: Fine-tuned models may need periodic retraining as market dynamics evolve

7. **API Dependency**: Relies on `yfinance` for market data; API changes or rate limits could affect functionality

## See Also

- [SETUP.md](SETUP.md) - Installation and setup instructions
- [USAGE.md](USAGE.md) - Comprehensive command reference and usage guide
- [docs/LAMBDA_SETUP.md](docs/LAMBDA_SETUP.md) - Lambda Labs GPU setup guide
