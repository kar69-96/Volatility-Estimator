# Web Interface Guide

## Quick Start

1. **Install Streamlit** (if not already installed):
   ```bash
   pip install streamlit
   ```

2. **Launch the web interface**:
   ```bash
   streamlit run src/app.py
   ```

   Or use the convenience script:
   ```bash
   ./run_app.sh
   ```

3. **Open your browser** to `http://localhost:8501`

## Features

### 🎯 Analysis Modes

1. **Single Estimator**: Run one volatility estimator with customizable parameters
2. **Compare All**: Compare all 5 estimators side-by-side with correlation analysis
3. **Event Analysis**: Analyze how economic events impact volatility
4. **Predictions**: Generate volatility predictions for upcoming events
5. **Full Analysis**: Complete analysis with all features combined

### 📊 Interactive Controls

- **Asset Symbol**: Select from popular assets (SPY, QQQ, AAPL, etc.)
- **Date Range**: Choose start and end dates for analysis
- **Estimator**: Select from 5 available estimators
- **Window Size**: Adjust rolling window (10-252 days)
- **EWMA Lambda**: Tune EWMA decay factor (0.80-0.99)
- **Event Window**: Set days before/after events for analysis

### 📈 Visualizations

- Real-time volatility charts
- Estimator comparison overlays
- Prediction confidence bands
- Event impact visualizations

### 📥 Downloads

- CSV exports for all results
- Event analysis tables
- Prediction data
- Comparison matrices

## Usage Tips

1. **First Run**: The first analysis may take longer as it downloads and caches market data
2. **Caching**: Data is cached automatically - subsequent runs are faster
3. **Date Range**: Larger date ranges provide more historical context but take longer to process
4. **Event Analysis**: Requires events CSV file in `data/events/economic_calendar.csv`
5. **Predictions**: Best results with sufficient historical event data

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
- **Data Loading Issues**: Check internet connection (yfinance requires online access)
- **Port Already in Use**: Change port with `streamlit run src/app.py --server.port 8502`
- **Memory Issues**: Use smaller date ranges or window sizes for large datasets

