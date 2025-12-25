"""
Ultraminimalistic Streamlit front-end for Volatility Estimator Stack.

This web interface provides dropdown selectors to demonstrate all
capabilities of the platform.
"""

import os
import sys
from pathlib import Path

# Note: This Streamlit app requires Python 3.10+ on macOS
# Python 3.9 on macOS has a known kqueue selector bug that breaks WebSocket connections

import pandas as pd
import streamlit as st
import yaml

# Set environment variables for Streamlit configuration
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.comparison import (
    run_all_estimators,
    calculate_correlation_matrix,
    calculate_mse_matrix,
    generate_comparison_statistics
)
from src.data_loader import get_market_data, check_data_quality
from src.event_analysis import analyze_all_events
from src.estimators import get_estimator, list_estimators
from src.predictions import (
    build_pattern_database,
    find_similar_events,
    predict_volatility_path,
    backtest_predictions
)
from src.reporting import (
    export_to_excel,
    plot_volatility_comparison,
    plot_predictions,
    generate_summary_report
)
from src.utils import load_events, setup_logging


# Page configuration - MUST be the very first Streamlit command
# No code should execute before this
st.set_page_config(
    page_title="Volatility Estimator Stack",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultraminimalistic design
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    h1 {
        font-size: 2.5rem;
        font-weight: 300;
        letter-spacing: -0.02em;
    }
    h2 {
        font-size: 1.5rem;
        font-weight: 300;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config(config_path='config.yaml'):
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except:
        return {}


@st.cache_data(ttl=3600)
def load_market_data(symbol, start_date, end_date, use_cache=True):
    """Load market data with caching."""
    try:
        config = load_config()
        data_config = config.get('data', {})
        cache_dir = data_config.get('cache_dir', './data/cache')
        cache_format = data_config.get('cache_format', 'parquet')
        
        # Convert dates to string format if needed
        if hasattr(start_date, 'strftime'):
            start_date = start_date.strftime('%Y-%m-%d')
        if hasattr(end_date, 'strftime'):
            end_date = end_date.strftime('%Y-%m-%d')
        
        df = get_market_data(
            symbol=symbol,
            start_date=str(start_date),
            end_date=str(end_date),
            use_cache=use_cache,
            cache_dir=cache_dir,
            cache_format=cache_format
        )
        return df
    except Exception as e:
        import traceback
        error_msg = f"Error loading data: {str(e)}\n{traceback.format_exc()}"
        st.error(error_msg)
        return None


# Main function - Streamlit runs this automatically
def main():
    """Main Streamlit app."""
    
    # Title
    st.title("Volatility Estimator Stack")
    st.markdown("---")
    
    # Load configuration
    config = load_config()
    vol_config = config.get('volatility', {})
    events_config = config.get('events', {})
    predictions_config = config.get('predictions', {})
    data_config = config.get('data', {})
    
    # Sidebar - Input Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Asset Selection
        default_assets = data_config.get('assets', ['SPY', 'QQQ', 'AAPL'])
        symbol = st.selectbox(
            "Asset Symbol",
            options=default_assets + ['MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            index=0
        )
        
        # Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime(data_config.get('start_date', '2020-01-01')).date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime(data_config.get('end_date', '2024-12-31')).date()
            )
        
        st.markdown("---")
        
        # Analysis Mode
        analysis_mode = st.radio(
            "Analysis Mode",
            options=["Single Estimator", "Compare All", "Event Analysis", "Predictions", "Full Analysis"],
            index=0
        )
        
        st.markdown("---")
        
        # Estimator Selection (for single mode)
        if analysis_mode == "Single Estimator":
            estimator_name = st.selectbox(
                "Estimator",
                options=list_estimators(),
                index=list_estimators().index(vol_config.get('default_estimator', 'yang_zhang'))
            )
            
            # EWMA Lambda (if EWMA selected)
            if estimator_name == 'ewma':
                lambda_param = st.slider(
                    "EWMA Lambda",
                    min_value=0.80,
                    max_value=0.99,
                    value=vol_config.get('ewma_lambda', 0.94),
                    step=0.01
                )
            else:
                lambda_param = vol_config.get('ewma_lambda', 0.94)
        else:
            estimator_name = None
            lambda_param = vol_config.get('ewma_lambda', 0.94)
        
        # Window Size
        window = st.slider(
            "Rolling Window (days)",
            min_value=10,
            max_value=252,
            value=vol_config.get('default_window', 60),
            step=10
        )
        
        # Event Analysis Options
        if analysis_mode in ["Event Analysis", "Predictions", "Full Analysis"]:
            st.markdown("---")
            event_window = st.slider(
                "Event Window (days)",
                min_value=1,
                max_value=30,
                value=events_config.get('pre_window', 5),
                step=1
            )
        else:
            event_window = events_config.get('pre_window', 5)
        
        # Action Button
        st.markdown("---")
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main Content Area
    if run_analysis:
        # Show loading
        try:
            with st.spinner(f"Loading data for {symbol}..."):
                df = load_market_data(symbol, start_date, end_date)
            
            if df is None or len(df) == 0:
                st.error("Failed to load market data. Please check your internet connection and try again.")
                st.info("💡 Tip: Make sure you have an internet connection. The first run downloads data from yfinance API.")
                return
            
            st.success(f"✓ Loaded {len(df)} days of data for {symbol}")
            
            # Data Quality Check
            try:
                quality_report = check_data_quality(df)
                with st.expander("📊 Data Quality Report"):
                    st.json({
                        'Total Rows': quality_report['total_rows'],
                        'Date Range': f"{quality_report['date_range'][0]} to {quality_report['date_range'][1]}" if quality_report.get('date_range') else 'N/A',
                        'Missing Values': quality_report.get('missing_values', {})
                    })
            except Exception as e:
                st.warning(f"Could not generate quality report: {str(e)}")
            
            # Run Analysis Based on Mode
            try:
                if analysis_mode == "Single Estimator":
                    if estimator_name is None:
                        st.error("Please select an estimator in the sidebar.")
                        return
                    run_single_estimator(df, estimator_name, window, lambda_param, symbol)
                
                elif analysis_mode == "Compare All":
                    run_comparison(df, window, lambda_param, symbol)
                
                elif analysis_mode == "Event Analysis":
                    run_event_analysis(df, window, event_window, symbol)
                
                elif analysis_mode == "Predictions":
                    run_predictions(df, window, event_window, symbol)
                
                elif analysis_mode == "Full Analysis":
                    run_full_analysis(df, window, lambda_param, event_window, symbol)
            except Exception as e:
                import traceback
                st.error(f"Error running analysis: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
        except Exception as e:
            import traceback
            st.error(f"Unexpected error: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    else:
        # Welcome message
        st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis' to begin.")
        
        # Show available features
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Estimators", "5")
        with col2:
            st.metric("Events", "500+")
        with col3:
            st.metric("Assets", "Unlimited")
        with col4:
            st.metric("Analysis Modes", "5")


def run_single_estimator(df, estimator_name, window, lambda_param, symbol):
    """Run single estimator analysis."""
    st.header(f"📈 {estimator_name.replace('_', ' ').title()} Estimator")
    
    try:
        if estimator_name == 'ewma':
            estimator = get_estimator(estimator_name, window, 252, lambda_param=lambda_param)
        else:
            estimator = get_estimator(estimator_name, window, 252)
        
        volatility = estimator.compute(df, annualize=True)
        
        results = pd.DataFrame({
            'date': df['date'],
            'volatility': volatility
        }).dropna()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Volatility", f"{results['volatility'].mean():.2f}%")
        with col2:
            st.metric("Std Deviation", f"{results['volatility'].std():.2f}%")
        with col3:
            st.metric("Minimum", f"{results['volatility'].min():.2f}%")
        with col4:
            st.metric("Maximum", f"{results['volatility'].max():.2f}%")
        
        # Plot
        st.line_chart(results.set_index('date')['volatility'])
        
        # Data table
        with st.expander("📋 View Data"):
            st.dataframe(results, use_container_width=True)
        
        # Download
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{symbol}_{estimator_name}_volatility.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")


def run_comparison(df, window, lambda_param, symbol):
    """Run comparison of all estimators."""
    st.header("🔀 Estimator Comparison")
    
    try:
        results = run_all_estimators(df, window, 252, lambda_param)
        results = results.dropna(how='all')
        
        # Statistics
        stats = generate_comparison_statistics(results)
        stats_df = pd.DataFrame(stats).T
        stats_df.columns = ['Mean', 'Std', 'Min', 'Max', 'Count']
        
        st.subheader("Summary Statistics")
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
        
        # Correlation Matrix
        correlation = calculate_correlation_matrix(results)
        st.subheader("Correlation Matrix")
        st.dataframe(correlation.style.format("{:.4f}").background_gradient(cmap='RdYlGn', vmin=-1, vmax=1), use_container_width=True)
        
        # Plot comparison
        st.subheader("Volatility Comparison")
        vol_cols = [col for col in results.columns if col != 'date']
        chart_data = results.set_index('date')[vol_cols]
        st.line_chart(chart_data)
        
        # Download
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison CSV",
            data=csv,
            file_name=f"{symbol}_comparison.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")


def run_event_analysis(df, window, event_window, symbol):
    """Run event analysis."""
    st.header("📅 Event Impact Analysis")
    
    try:
        # Get estimator
        estimator = get_estimator('yang_zhang', window, 252)
        volatility = estimator.compute(df, annualize=True)
        
        results_df = pd.DataFrame({
            'date': df['date'],
            'volatility': volatility
        }).dropna()
        
        # Load events
        events_df = load_events('./data/events/economic_calendar.csv')
        
        # Run analysis
        event_results = analyze_all_events(
            volatility_series=results_df['volatility'],
            volatility_dates=results_df['date'],
            events_df=events_df,
            pre_window=event_window,
            post_window=event_window
        )
        event_results = event_results.dropna(subset=['pre_vol', 'post_vol'])
        
        if len(event_results) == 0:
            st.warning("No events found in the data range.")
            return
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Events Analyzed", len(event_results))
        with col2:
            significant = event_results['significant'].sum()
            st.metric("Significant Events", f"{significant} ({100*significant/len(event_results):.1f}%)")
        with col3:
            avg_change = event_results['volatility_change_pct'].mean()
            st.metric("Avg Change", f"{avg_change:.2f}%")
        
        # Top events
        st.subheader("Most Impactful Events")
        top_events = event_results.nlargest(10, 'volatility_change_pct')[
            ['event_date', 'event_type', 'description', 'volatility_change_pct', 'significant']
        ]
        st.dataframe(top_events.style.format({'volatility_change_pct': '{:.2f}%'}), use_container_width=True)
        
        # Full results
        with st.expander("📋 All Event Results"):
            st.dataframe(event_results, use_container_width=True)
        
        # Download
        csv = event_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Event Analysis CSV",
            data=csv,
            file_name=f"{symbol}_event_analysis.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")


def run_predictions(df, window, event_window, symbol):
    """Run volatility predictions."""
    st.header("🔮 Volatility Predictions")
    
    try:
        # Get estimator
        estimator = get_estimator('yang_zhang', window, 252)
        volatility = estimator.compute(df, annualize=True)
        
        results_df = pd.DataFrame({
            'date': df['date'],
            'volatility': volatility
        }).dropna()
        
        # Load events
        events_df = load_events('./data/events/economic_calendar.csv')
        
        # Build pattern database
        pattern_db = build_pattern_database(
            events_df, results_df['volatility'], results_df['date'],
            pre_window=event_window, post_window=event_window
        )
        
        # Find upcoming events
        current_date = results_df['date'].max()
        upcoming_events = events_df[pd.to_datetime(events_df['date']) > pd.to_datetime(current_date)]
        
        if len(upcoming_events) == 0:
            st.warning("No upcoming events found.")
            return
        
        # Generate prediction for first upcoming event
        upcoming_event = upcoming_events.iloc[0]
        similar = find_similar_events(
            upcoming_event, events_df, results_df['volatility'], results_df['date']
        )
        
        if len(similar) == 0:
            st.warning("No similar historical events found for prediction.")
            return
        
        predictions_df = predict_volatility_path(
            pd.to_datetime(upcoming_event['date']),
            pattern_db, results_df['volatility'], results_df['date'],
            similar, lookback_window=10
        )
        
        # Display prediction
        st.subheader(f"Prediction for {upcoming_event['event_type']} on {upcoming_event['date']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Similar Events Found", len(similar))
        with col2:
            avg_pred = predictions_df['predicted'].mean()
            st.metric("Avg Predicted Volatility", f"{avg_pred:.2f}%")
        with col3:
            st.metric("Confidence Level", "95%")
        
        # Plot prediction
        st.subheader("Predicted Volatility Path")
        chart_data = predictions_df.set_index('date')[['predicted', 'lower_bound', 'upper_bound']]
        st.line_chart(chart_data)
        
        # Backtesting
        st.subheader("Backtesting Results")
        backtest_metrics = backtest_predictions(
            events_df, results_df['volatility'], results_df['date'],
            pattern_db, event_window, event_window
        )
        
        if backtest_metrics['total_predictions'] > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", backtest_metrics['total_predictions'])
            with col2:
                st.metric("MAE", f"{backtest_metrics['prediction_mae']:.2f}%")
            with col3:
                st.metric("Correlation", f"{backtest_metrics['prediction_correlation']:.3f}")
        
        # Download
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name=f"{symbol}_predictions.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")


def run_full_analysis(df, window, lambda_param, event_window, symbol):
    """Run full analysis with all features."""
    st.header("🎯 Full Analysis")
    
    # Comparison
    with st.spinner("Running estimator comparison..."):
        results = run_all_estimators(df, window, 252, lambda_param)
        results = results.dropna(how='all')
    
    st.subheader("📊 Estimator Comparison")
    vol_cols = [col for col in results.columns if col != 'date']
    st.line_chart(results.set_index('date')[vol_cols])
    
    # Event Analysis
    with st.spinner("Running event analysis..."):
        estimator = get_estimator('yang_zhang', window, 252)
        volatility = estimator.compute(df, annualize=True)
        
        results_df = pd.DataFrame({
            'date': df['date'],
            'volatility': volatility
        }).dropna()
        
        events_df = load_events('./data/events/economic_calendar.csv')
        event_results = analyze_all_events(
            volatility_series=results_df['volatility'],
            volatility_dates=results_df['date'],
            events_df=events_df,
            pre_window=event_window,
            post_window=event_window
        )
        event_results = event_results.dropna(subset=['pre_vol', 'post_vol'])
    
    st.subheader("📅 Event Impact Summary")
    if len(event_results) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Events", len(event_results))
        with col2:
            significant = event_results['significant'].sum()
            st.metric("Significant", f"{significant}")
        with col3:
            avg_change = event_results['volatility_change_pct'].mean()
            st.metric("Avg Change", f"{avg_change:.2f}%")
    
    # Predictions
    with st.spinner("Generating predictions..."):
        pattern_db = build_pattern_database(
            events_df, results_df['volatility'], results_df['date'],
            pre_window=event_window, post_window=event_window
        )
        
        current_date = results_df['date'].max()
        upcoming_events = events_df[pd.to_datetime(events_df['date']) > pd.to_datetime(current_date)]
        
        if len(upcoming_events) > 0:
            upcoming_event = upcoming_events.iloc[0]
            similar = find_similar_events(
                upcoming_event, events_df, results_df['volatility'], results_df['date']
            )
            
            if len(similar) > 0:
                predictions_df = predict_volatility_path(
                    pd.to_datetime(upcoming_event['date']),
                    pattern_db, results_df['volatility'], results_df['date'],
                    similar, lookback_window=10
                )
                
                st.subheader("🔮 Predictions")
                st.line_chart(predictions_df.set_index('date')[['predicted', 'lower_bound', 'upper_bound']])
    
    # Download all
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Comparison CSV",
            results.to_csv(index=False),
            f"{symbol}_comparison.csv",
            "text/csv"
        )
    with col2:
        if len(event_results) > 0:
            st.download_button(
                "Event Analysis CSV",
                event_results.to_csv(index=False),
                f"{symbol}_events.csv",
                "text/csv"
            )
    with col3:
        if len(upcoming_events) > 0 and len(similar) > 0:
            st.download_button(
                "Predictions CSV",
                predictions_df.to_csv(index=False),
                f"{symbol}_predictions.csv",
                "text/csv"
            )


# Call main function to run the Streamlit app
# Streamlit executes the script directly, so we just call main() once
main()

