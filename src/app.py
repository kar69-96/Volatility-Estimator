"""
Hyperminimalistic Volatility Estimator Frontend.

Core Design Principle:
Separate "What am I analyzing?" from "How am I analyzing it?"

Navigation Structure:
[ Single Asset ] [ Asset Comparison ]
  ├─ Volatility Analysis
  └─ Volatility Forecasting
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Plotly import (optional)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

# Set environment variables
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis import (
    run_all_estimators,
    calculate_correlation_matrix,
    generate_comparison_statistics,
    calculate_sharpe_ratio,
    calculate_beta,
    calculate_treynor_ratio,
    calculate_max_drawdown,
)
from src.data import get_market_data, get_implied_volatility, calculate_returns
from src.estimators import get_estimator, list_estimators
from src.prediction import (
    is_deep_learning_available,
    predict_volatility_dl,
    predict_neural_garch,
    analyze_fed_rate_scenario,
)
from src.visualization import (
    calculate_historical_volatility,
    classify_volatility_regime,
    calculate_historical_averages,
    calculate_percentile_ranking,
    estimate_confidence_intervals,
    get_alert_level,
    prepare_forecast_dataframe,
    calculate_forecast_statistics,
    calculate_y_axis_range,
    clean_series_for_plotly,
    create_forecast_chart,
)

# Check deep learning availability
DL_AVAILABLE = is_deep_learning_available()

# Page configuration
st.set_page_config(
    page_title="Volatility Estimator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure theme to remove red colors
try:
    # Try to set theme via config (if available in Streamlit version)
    import streamlit as st
    # This might not work in all Streamlit versions, but worth trying
except:
    pass

# Hyperminimalistic CSS
st.markdown("""
    <style>
    /* Remove default Streamlit styling */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        padding: 1rem 2rem;
    }
    
    /* Force all text to be black */
    * {
        color: #000000 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 400;
        color: #000000 !important;
        margin: 0.5rem 0;
    }
    
    /* Black and white theme */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Minimal tabs - remove all underlines and red selectors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #000000;
    }
    
    /* Add padding between tab texts */
    .stTabs [data-baseweb="tab"],
    .stTabs button[data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
        margin-right: 2rem;
    }
    
    /* Remove ALL borders and underlines from tabs */
    .stTabs [data-baseweb="tab"],
    .stTabs button[data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] * {
        border: none !important;
        border-bottom: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
        color: #000000 !important;
    }
    
    /* Specifically target active tabs to remove red underline */
    .stTabs [aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs button[data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] *,
    .stTabs [data-baseweb="tab"][aria-selected="true"] > *,
    .stTabs [data-baseweb="tab"][aria-selected="true"] > * > * {
        border: none !important;
        border-bottom: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        box-shadow: none !important;
        outline: none !important;
        background-color: transparent !important;
        color: #000000 !important;
    }
    
    /* Remove any pseudo-elements that might create underlines */
    .stTabs [data-baseweb="tab"]::after,
    .stTabs [data-baseweb="tab"]::before,
    .stTabs [data-baseweb="tab"][aria-selected="true"]::after,
    .stTabs [data-baseweb="tab"][aria-selected="true"]::before,
    .stTabs button[data-baseweb="tab"]::after,
    .stTabs button[data-baseweb="tab"]::before,
    .stTabs button[data-baseweb="tab"][aria-selected="true"]::after,
    .stTabs button[data-baseweb="tab"][aria-selected="true"]::before {
        display: none !important;
        content: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Target BaseWeb's internal styling */
    [data-baseweb="tab-list"] [aria-selected="true"],
    [data-baseweb="tab-list"] [aria-selected="true"] * {
        border-bottom: none !important;
        box-shadow: none !important;
    }
    
    /* Remove tab panel borders */
    .stTabs [role="tabpanel"] {
        border: none !important;
    }
    
    /* Checkboxes */
    .stCheckbox label {
        font-size: 0.9rem;
        color: #000000 !important;
    }
    
    .stCheckbox label p {
        color: #000000 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #000000 !important;
    }
    
    .stRadio label p {
        color: #000000 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #000000;
        color: #FFFFFF !important;
        border: 1px solid #000000;
        border-radius: 0;
        font-weight: 400;
    }
    
    .stButton > button,
    .stButton > button *,
    .stButton > button span {
        color: #FFFFFF !important;
    }
    
    .stButton > button:hover {
        background-color: #333333;
        color: #FFFFFF !important;
    }
    
    .stButton > button:hover *,
    .stButton > button:hover span {
        color: #FFFFFF !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid #000000;
        border-radius: 0;
        color: #000000 !important;
    }
    
    .stTextInput label,
        .stSelectbox label {
        color: #000000 !important;
        }
    
    /* Date input */
    .stDateInput label {
        color: #000000 !important;
    }
    
    .stDateInput > div > div > input {
        color: #000000 !important;
    }
    
    /* Slider */
    .stSlider label {
        color: #000000 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #000000 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #000000 !important;
    }
    
    /* Info, warning, error messages */
    .stInfo,
    .stWarning,
    .stError,
    .stSuccess {
        color: #000000 !important;
    }
    
    /* Dataframe text */
    .dataframe {
        color: #000000 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove any Streamlit default active tab colors */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #000000 !important;
        background-color: transparent !important;
        border-bottom: none !important;
    }
    
    /* Target BaseWeb's box-shadow underline (this is how they create the red line) */
    [data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"] * {
        box-shadow: none !important;
        -webkit-box-shadow: none !important;
        -moz-box-shadow: none !important;
        border-bottom-color: transparent !important;
        border-bottom: 0px solid transparent !important;
    }
    
    /* Hide any element that might be creating the red line */
    [data-baseweb="tab-list"] > * > *[aria-selected="true"],
    [data-baseweb="tab-list"] [aria-selected="true"]::after {
        border-bottom: 0px !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    </style>
    <script>
    // Aggressively remove red underlines using continuous monitoring
    function forceRemoveUnderlines() {
        // Target all possible selectors
        const selectors = [
            '[data-baseweb="tab"][aria-selected="true"]',
            'button[data-baseweb="tab"][aria-selected="true"]',
            '[role="tab"][aria-selected="true"]',
            '[data-baseweb="tab-list"] [aria-selected="true"]'
        ];
        
        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                // Remove all possible underline styles
                el.style.setProperty('border-bottom', '0px solid transparent', 'important');
                el.style.setProperty('box-shadow', 'none', 'important');
                el.style.setProperty('-webkit-box-shadow', 'none', 'important');
                el.style.setProperty('-moz-box-shadow', 'none', 'important');
                el.style.setProperty('outline', 'none', 'important');
                el.style.setProperty('border-bottom-color', 'transparent', 'important');
                el.style.setProperty('border-bottom-width', '0px', 'important');
                
                // Remove from all nested elements and pseudo-elements
                el.querySelectorAll('*').forEach(child => {
                    child.style.setProperty('border-bottom', '0px solid transparent', 'important');
                    child.style.setProperty('box-shadow', 'none', 'important');
                    child.style.setProperty('-webkit-box-shadow', 'none', 'important');
                    child.style.setProperty('-moz-box-shadow', 'none', 'important');
                    child.style.setProperty('border-bottom-color', 'transparent', 'important');
                });
                
                // Also try to remove computed styles
                const computed = window.getComputedStyle(el);
                if (computed.boxShadow && computed.boxShadow !== 'none') {
                    el.style.setProperty('box-shadow', 'none', 'important');
                }
                if (computed.borderBottomWidth && computed.borderBottomWidth !== '0px') {
                    el.style.setProperty('border-bottom', '0px solid transparent', 'important');
                }
            });
        });
    }
    
    // Run immediately and continuously
    forceRemoveUnderlines();
    setInterval(forceRemoveUnderlines, 50);
    
    // Also run on DOM changes
    const observer = new MutationObserver(forceRemoveUnderlines);
    if (document.body) {
        observer.observe(document.body, { childList: true, subtree: true, attributes: true });
    }
    </script>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data(ttl=3600)
def load_market_data(symbol: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load and cache market data.
    
    Returns:
        Tuple of (DataFrame or None, error message or None)
    """
    try:
        config = load_config()
        data_config = config.get('data', {})
        cache_dir = data_config.get('cache_dir', './data/cache')
        cache_format = data_config.get('cache_format', 'parquet')
        
        df = get_market_data(
            symbol=symbol,
            start_date=str(start_date),
            end_date=str(end_date),
            use_cache=True,
            cache_dir=cache_dir,
            cache_format=cache_format
        )
        return df, None
    except Exception as e:
        import traceback
        error_msg = f"Error loading {symbol}: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
        return None, error_msg


@st.cache_data
def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except:
        return {}


def format_estimator_name(name: str) -> str:
    """Format estimator name for display."""
    return name.replace('_', ' ').title()


def calculate_volatility_stats(volatility: pd.Series) -> dict:
    """Calculate comprehensive volatility statistics."""
    vol_clean = volatility.dropna()
    if len(vol_clean) == 0:
        return {}
    
    # Calculate skewness and kurtosis manually (no scipy dependency)
    mean = vol_clean.mean()
    std = vol_clean.std()
    
    if len(vol_clean) > 2 and std > 0:
        normalized = (vol_clean - mean) / std
        skewness = (normalized ** 3).mean()
        kurtosis = (normalized ** 4).mean() - 3  # Excess kurtosis
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    return {
        'mean': mean,
        'std': std,
        'min': vol_clean.min(),
        'max': vol_clean.max(),
        'median': vol_clean.median(),
        'skewness': skewness,
        'kurtosis': kurtosis,
    }


# ============================================================================
# Single Asset Views
# ============================================================================

def render_single_asset_analysis():
    """Render Single Asset → Volatility Analysis view."""
    st.header("Volatility Analysis")
    
    # Asset selection
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        symbol = st.text_input("Asset Symbol", value="AAPL", key="single_asset_symbol").upper()
    with col2:
        start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01').date())
    with col3:
        end_date = st.date_input("End Date", value=date.today())
    
    # Load data
    if st.button("Load Data", type="primary"):
        with st.spinner(f"Loading {symbol}..."):
            df, error_msg = load_market_data(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            st.session_state[f'data_{symbol}'] = df
            st.success(f"Loaded {len(df)} days of data")
        else:
            if error_msg:
                st.error(f"Failed to load data: {error_msg.split(chr(10))[0]}")
                with st.expander("Error Details"):
                    st.code(error_msg)
            else:
                st.error("Failed to load data: No data returned")
            return
                    
    if f'data_{symbol}' not in st.session_state:
        st.info("Enter asset symbol and click 'Load Data'")
        return
                
    df = st.session_state[f'data_{symbol}']
    
    # Method selection
    st.subheader("Methods")
    available_methods = list_estimators()
    if DL_AVAILABLE:
        available_methods.append('neural_garch')
    
    method_cols = st.columns(min(7, len(available_methods)))
    selected_methods = []
    for i, method in enumerate(available_methods):
        with method_cols[i % len(method_cols)]:
            if st.checkbox(format_estimator_name(method), key=f"method_{method}", value=(method == 'yang_zhang')):
                selected_methods.append(method)
    
    if not selected_methods:
        st.warning("Select at least one method")
        return
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        window = st.slider("Rolling Window (days)", 10, 252, 60)
    with col2:
        lambda_param = st.slider("EWMA Lambda", 0.80, 0.99, 0.94, 0.01) if 'ewma' in selected_methods else 0.94
    
    # Output mode
    st.subheader("Output Mode")
    output_mode = st.radio("", ["Time Series", "Distribution", "Summary Stats"], horizontal=True, key="output_mode")
    
    # Calculate volatilities
    if st.button("Calculate", type="primary"):
        results = {}
        config = load_config()
        vol_config = config.get('volatility', {})
        annualization = vol_config.get('annualization_factor', 252)
        
        for method in selected_methods:
            try:
                if method == 'ewma':
                    estimator = get_estimator(method, window, annualization, lambda_param=lambda_param)
                elif method == 'neural_garch':
                    if not DL_AVAILABLE:
                        continue
                    # Neural GARCH requires special handling
                    result = predict_neural_garch(df, p=1, q=1, epochs=50, device='auto')
                    if 'volatility' in result and result['volatility'] is not None:
                        results[method] = {
                            'values': result['volatility'].set_index('date')['volatility'],
                            'stats': calculate_volatility_stats(result['volatility']['volatility'])
                        }
                    continue
                else:
                    estimator = get_estimator(method, window, annualization)
                
                    volatility = estimator.compute(df, annualize=True)
                results[method] = {
                    'values': volatility,
                    'stats': calculate_volatility_stats(volatility)
                }
            except Exception as e:
                st.warning(f"Error calculating {method}: {str(e)}")
        
        st.session_state['analysis_results'] = results
    
    # Display results
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        
        if output_mode == "Time Series":
            if not PLOTLY_AVAILABLE:
                st.error("Plotly required for charts. Install with: pip install plotly")
            else:
                # Time series chart
                fig = go.Figure()
                colors = ['#000000', '#666666', '#999999', '#CCCCCC', '#333333', '#555555', '#777777']
                for i, (method, data) in enumerate(results.items()):
                    values = data['values']
                    x, y = clean_series_for_plotly(values)
                    if x:
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            name=format_estimator_name(method),
                            line=dict(color=colors[i % len(colors)], width=1)
                        ))
                fig.update_layout(
                    title="",
                    xaxis_title="",
                    yaxis_title="Volatility (%)",
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black', size=11),
                    showlegend=True,
                    margin=dict(l=40, r=20, t=20, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif output_mode == "Distribution":
            if not PLOTLY_AVAILABLE:
                st.error("Plotly required for charts. Install with: pip install plotly")
            else:
                # Distribution chart
                fig = go.Figure()
                colors = ['#000000', '#666666', '#999999', '#CCCCCC', '#333333']
                for i, (method, data) in enumerate(results.items()):
                    values = data['values'].dropna()
                    fig.add_trace(go.Histogram(
                        x=values.values,
                        name=format_estimator_name(method),
                        opacity=0.7,
                        nbinsx=30,
                        marker_color=colors[i % len(colors)]
                    ))
                fig.update_layout(
                    title="",
                    xaxis_title="Volatility (%)",
                    yaxis_title="Frequency",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black', size=11),
                    barmode='overlay',
                    margin=dict(l=40, r=20, t=20, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif output_mode == "Summary Stats":
            # Summary statistics table
            stats_data = []
            for method, data in results.items():
                stats = data.get('stats', {})
                stats_data.append({
                    'Method': format_estimator_name(method),
                    'Mean Vol (%)': f"{stats.get('mean', 0):.2f}",
                    'Std Dev': f"{stats.get('std', 0):.2f}",
                    'Min': f"{stats.get('min', 0):.2f}",
                    'Max': f"{stats.get('max', 0):.2f}",
                    'Skewness': f"{stats.get('skewness', 0):.2f}",
                    'Kurtosis': f"{stats.get('kurtosis', 0):.2f}",
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)


def render_single_asset_forecasting():
    """Render Single Asset → Volatility Forecasting view."""
    st.header("Volatility Forecasting")
    
    # Asset selection
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        symbol = st.text_input("Asset Symbol", value="AAPL", key="forecast_symbol").upper()
    with col2:
        start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01').date(), key="forecast_start")
    with col3:
        end_date = st.date_input("End Date", value=date.today(), key="forecast_end")
    
    if st.button("Load Data", type="primary", key="forecast_load"):
        with st.spinner(f"Loading {symbol}..."):
            df, error_msg = load_market_data(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            st.session_state[f'forecast_data_{symbol}'] = df
            # Note: symbol is already stored in session_state by the text_input widget
            st.success(f"Loaded {len(df)} days of data")
        else:
            if error_msg:
                st.error(f"Failed to load data: {error_msg.split(chr(10))[0]}")
                with st.expander("Error Details"):
                    st.code(error_msg)
            else:
                st.error("Failed to load data: No data returned")
            return
        
    if f'forecast_data_{symbol}' not in st.session_state:
        st.info("Enter asset symbol and click 'Load Data'")
        return
    df = st.session_state[f'forecast_data_{symbol}']
    
    if not DL_AVAILABLE:
        st.warning("Deep learning not available. Install PyTorch for forecasting.")
        return
    
    # Forecast options
    st.subheader("Forecast Configuration")
    col1, col2 = st.columns(2)
    with col1:
        forecast_type = st.selectbox("Forecast Type", ["iTransformer", "Neural GARCH"], key="forecast_type_select")
    with col2:
        # Show horizon input for both forecast types
        horizon = st.number_input("Forecast Horizon (days)", min_value=1, max_value=252, value=20, step=1, key="forecast_horizon_input")
    
    # Fed Rate Effect Configuration
    st.subheader("Fed Rate Effect")
    include_fed_rate = st.checkbox("Include Fed Rate Effect", value=False, key="include_fed_rate_check")
    fed_rate_change_bps = 0
    if include_fed_rate:
        fed_rate_change_bps = st.slider(
            "Fed Rate Change Scenario (basis points)", 
            min_value=-100, 
            max_value=100, 
            value=0, 
            step=25,
            key="fed_rate_slider",
            help="Positive = rate increase, Negative = rate decrease. 25 bps = 0.25%"
        )
    
    # Generate forecast button
    if st.button("Generate Forecast", type="primary", key="forecast_generate_btn"):
        with st.spinner("Generating forecast..."):
            if include_fed_rate:
                # Use Fed rate scenario analysis
                result = analyze_fed_rate_scenario(
                    stock_df=df,
                    fed_rate_change_bps=fed_rate_change_bps,
                    prediction_horizon=horizon if horizon else 20,
                    device='auto'
                )
                if 'error' not in result and result.get('results') is not None:
                    st.session_state['forecast_result'] = result
                    st.session_state['forecast_type'] = 'fed_rate'
                    st.session_state['forecast_horizon'] = horizon if horizon else 20
                    st.session_state['fed_rate_change'] = fed_rate_change_bps
                else:
                    # Check for error first
                    if 'error' in result:
                        error_msg = result.get('error', 'Unknown error')
                        st.error(f"Forecast failed: {error_msg.split(chr(10))[0]}")
                        with st.expander("Error Details"):
                            st.code(error_msg)
                    else:
                        # No error key but also no results - this shouldn't happen
                        st.error(f"Forecast failed: Unknown error - result keys: {list(result.keys())}")
                        with st.expander("Debug Info"):
                            st.json(result)
            elif forecast_type == "iTransformer":
                result = predict_volatility_dl(
                    df=df,
                    prediction_horizon=horizon,
                    device='auto'
                )
                if 'error' not in result:
                    st.session_state['forecast_result'] = result
                    st.session_state['forecast_type'] = 'itransformer'
                    st.session_state['forecast_horizon'] = horizon
                else:
                    error_msg = result.get('error', 'Unknown error')
                    st.error(f"Forecast failed: {error_msg}")
                    with st.expander("Error Details"):
                        st.code(error_msg)
            elif forecast_type == "Neural GARCH":
                result = predict_neural_garch(
                    df, 
                    p=1, 
                    q=1, 
                    epochs=50, 
                    device='auto',
                    prediction_horizon=horizon
                )
                # Check for both 'error' and 'forecast_error' keys
                if 'error' not in result and 'forecast_error' not in result:
                    st.session_state['forecast_result'] = result
                    st.session_state['forecast_type'] = 'neural_garch'
                    st.session_state['forecast_horizon'] = horizon
                else:
                    # Check for forecast_error first (forecast failed but historical worked)
                    if 'forecast_error' in result:
                        error_msg = result.get('forecast_error', 'Unknown forecast error')
                        st.error(f"Forecast failed: {error_msg}")
                        with st.expander("Error Details"):
                            st.code(error_msg)
                        # Still show historical results if available
                        if 'volatility' in result:
                            st.warning("Historical volatility calculated, but forecast failed.")
                            st.session_state['forecast_result'] = result
                            st.session_state['forecast_type'] = 'neural_garch'
                            st.session_state['forecast_horizon'] = horizon
                    else:
                        # Main error (model training/prediction failed)
                        error_msg = result.get('error', 'Unknown error')
                        st.error(f"Forecast failed: {error_msg}")
                        with st.expander("Error Details"):
                            st.code(error_msg)
    
    # Display results
    if 'forecast_result' in st.session_state:
        result = st.session_state['forecast_result']
        forecast_type_display = st.session_state.get('forecast_type', 'itransformer')
        horizon_display = st.session_state.get('forecast_horizon', 20)
        
        # Calculate historical volatility for context using Black-Scholes implied volatility
        symbol = st.session_state.get('forecast_symbol', '').upper()
        horizon_display = st.session_state.get('forecast_horizon', 20)
        
        try:
            # Use Black-Scholes implied volatility for estimation
            hist_vol = calculate_historical_volatility(
                df, 
                window=20, 
                method='yang_zhang',
                use_implied_vol=True,
                symbol=symbol,
                horizon_days=horizon_display
            )
            
            # Ensure we have data and proper index
            if len(hist_vol) == 0:
                # Final fallback: calculate simple rolling volatility from returns to show fluctuation
                try:
                    from src.data.returns import calculate_returns
                    returns = calculate_returns(df)
                    hist_vol = returns.rolling(window=20).std() * np.sqrt(252)
                    
                    # Try to calibrate to current IV if available
                    try:
                        # symbol is already defined at line 775
                        current_iv = get_implied_volatility(symbol, df, horizon_days=horizon_display, use_api=True)
                        if current_iv is not None and len(hist_vol.dropna()) > 0:
                            current_iv_decimal = current_iv / 100.0
                            current_realized = hist_vol.dropna().iloc[-1]
                            if current_realized > 0:
                                calibration_factor = current_iv_decimal / current_realized
                                hist_vol = hist_vol * calibration_factor
                    except:
                        pass
                except:
                    # Absolute final fallback: calculate from close prices
                    try:
                        close_prices = df['close']
                        returns = np.log(close_prices / close_prices.shift(1))
                        hist_vol = returns.rolling(window=20).std() * np.sqrt(252)
                    except:
                        hist_vol = pd.Series(dtype=float)
            
            # Ensure the index is datetime if df has date column
            if 'date' in df.columns and not isinstance(hist_vol.index, pd.DatetimeIndex):
                hist_vol.index = pd.to_datetime(df['date'].iloc[:len(hist_vol)])
            elif not isinstance(hist_vol.index, pd.DatetimeIndex) and len(hist_vol) > 0:
                # Create date index from df if available
                if 'date' in df.columns:
                    hist_vol.index = pd.to_datetime(df['date'].iloc[:len(hist_vol)])
                else:
                    # Use df index if it's datetime
                    hist_vol.index = df.index[:len(hist_vol)]
        except Exception as e:
            # Fallback: calculate simple rolling volatility from returns to show fluctuation
            try:
                from src.data.returns import calculate_returns
                returns = calculate_returns(df)
                hist_vol = returns.rolling(window=20).std() * np.sqrt(252)
                
                # Try to calibrate to current IV if available
                try:
                    # symbol is already defined at line 775
                    current_iv = get_implied_volatility(symbol, df, horizon_days=horizon_display, use_api=True)
                    if current_iv is not None and len(hist_vol.dropna()) > 0:
                        current_iv_decimal = current_iv / 100.0
                        current_realized = hist_vol.dropna().iloc[-1]
                        if current_realized > 0:
                            calibration_factor = current_iv_decimal / current_realized
                            hist_vol = hist_vol * calibration_factor
                except:
                    pass
                
                # Set index from df
                if 'date' in df.columns:
                    hist_vol.index = pd.to_datetime(df['date'].iloc[:len(hist_vol)])
                else:
                    hist_vol.index = df.index[:len(hist_vol)]
            except:
                # Absolute final fallback
                try:
                    close_prices = df['close']
                    returns = np.log(close_prices / close_prices.shift(1))
                    hist_vol = returns.rolling(window=20).std() * np.sqrt(252)
                    if 'date' in df.columns:
                        hist_vol.index = pd.to_datetime(df['date'].iloc[:len(hist_vol)])
                except:
                    hist_vol = pd.Series(dtype=float)
        # Get implied volatility matching forecast horizon
        symbol = st.session_state.get('forecast_symbol', '').upper()
        current_vol = get_implied_volatility(symbol, df, horizon_days=horizon_display, use_api=True)
        if current_vol is None:
            # Fallback to historical volatility if IV unavailable
            # Historical volatility is in decimal form, convert to percentage
            current_vol = float(hist_vol.iloc[-1]) * 100 if len(hist_vol) > 0 else 0.0
        
        if forecast_type_display == 'fed_rate' and 'results' in result:
            # Fed rate scenario results
            results = result['results']
            baseline_vol = results.get('baseline_volatility', 0)
            scenario_vol = results.get('scenario_volatility', 0)
            
            # Debug: Check if values are actually different
            if abs(baseline_vol - scenario_vol) < 0.01:  # Less than 0.01% difference
                # Values are too close - might indicate model issue
                fed_rate_change = st.session_state.get('fed_rate_change', 0)
                if fed_rate_change != 0:
                    st.warning(f"⚠️ Baseline and scenario forecasts are very similar ({baseline_vol:.2f}% vs {scenario_vol:.2f}%). The model may not be sensitive to Fed rate changes, or the change ({fed_rate_change} bps) may be too small to affect volatility.")
            
            # Use implied volatility for current volatility (matching forecast horizon)
            # This should remain constant regardless of Fed rate changes
            if current_vol is None or current_vol == 0:
                current_vol = baseline_vol
            # Calculate impact from actual current volatility to scenario forecast
            impact = scenario_vol - current_vol
            
            # Key Metrics Dashboard
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Volatility", f"{current_vol:.2f}%")
            with col2:
                st.metric("Forecast Value", f"{scenario_vol:.2f}%")
            with col3:
                st.metric("Change", f"{impact:+.2f}%", delta=f"{impact:+.2f}%")
            with col4:
                regime, _ = classify_volatility_regime(scenario_vol, hist_vol)
                st.metric("Forecast Regime", regime)
            
            # Fed Rate Scenario Comparison Chart
            if not PLOTLY_AVAILABLE:
                st.warning("Plotly required for charts. Install with: pip install plotly")
            elif len(hist_vol) == 0:
                st.warning("Unable to calculate historical volatility for chart display.")
            else:
                # Convert historical volatility from decimal to percentage
                hist_vol_pct = hist_vol * 100
                # Prepare data
                hist_vol_recent = hist_vol_pct.iloc[-90:] if len(hist_vol_pct) >= 90 else hist_vol_pct
                hist_x, hist_y = clean_series_for_plotly(hist_vol_recent)
                
                if hist_x:
                    try:
                        # Normalize x-axis: today - forecast_horizon to today + forecast_horizon
                        today = pd.Timestamp.today().normalize()
                        x_axis_start = today - pd.Timedelta(days=horizon_display)
                        x_axis_end = today + pd.Timedelta(days=horizon_display)
                        
                        # Filter historical data to x_axis_start to today
                        hist_vol_pct.index = pd.to_datetime(hist_vol_pct.index)
                        hist_vol_filtered = hist_vol_pct[(hist_vol_pct.index >= x_axis_start) & (hist_vol_pct.index <= today)]
                        
                        # If we don't have enough historical data, pad with the earliest available value
                        if len(hist_vol_filtered) < horizon_display:
                            earliest_value = hist_vol_pct.iloc[0] if len(hist_vol_pct) > 0 else 0
                            padding_dates = pd.date_range(start=x_axis_start, end=hist_vol_filtered.index[0] - pd.Timedelta(days=1), freq='D') if len(hist_vol_filtered) > 0 else pd.date_range(start=x_axis_start, end=today, freq='D')
                            padding_series = pd.Series([earliest_value] * len(padding_dates), index=padding_dates)
                            hist_vol_filtered = pd.concat([padding_series, hist_vol_filtered]).sort_index()
                        
                        # Ensure historical data extends exactly to today (the separator line)
                        # If the last point is not exactly at today, add it using the last available value
                        if len(hist_vol_filtered) > 0:
                            last_date = hist_vol_filtered.index[-1]
                            if last_date < today:
                                # Add today's point using the last available value to extend to the separator line
                                last_value = hist_vol_filtered.iloc[-1]
                                hist_vol_filtered = pd.concat([hist_vol_filtered, pd.Series([last_value], index=[today])])
                            elif last_date > today:
                                # If somehow we have data beyond today, trim it
                                hist_vol_filtered = hist_vol_filtered[hist_vol_filtered.index <= today]
                        else:
                            # If no historical data, create a single point at today
                            hist_vol_filtered = pd.Series([0], index=[today])
                        
                        hist_x, hist_y = clean_series_for_plotly(hist_vol_filtered)
                        today_date = today
                        
                        # Forecast dates from today to today+horizon_display (includes today for seamless connection)
                        forecast_dates = pd.date_range(start=today_date, periods=horizon_display + 1, freq='D')
                        
                        fig = go.Figure()
                        
                        # Historical volatility
                        fig.add_trace(go.Scatter(
                            x=hist_x,
                            y=hist_y,
                            mode='lines',
                            name='Historical Volatility',
                            line=dict(color='#000000', width=1.5)
                        ))
                        
                        # Current marker
                        fig.add_trace(go.Scatter(
                            x=[hist_x[-1]],
                            y=[hist_y[-1]],
                            mode='markers',
                            name='Current',
                            marker=dict(color='#000000', size=10, symbol='circle')
                        ))
                        
                        # Ensure forecast values are not NaN
                        s_vol = scenario_vol if not (pd.isna(scenario_vol) or np.isnan(scenario_vol)) else hist_y[-1]
                        
                        # Generate fluctuating forecast path for scenario only
                        from src.visualization.forecast_viz import generate_forecast_path
                        start_vol = hist_y[-1] if len(hist_y) > 0 else s_vol
                        
                        # Scenario forecast path - make it more volatile, starting from today (no gap)
                        scenario_path = generate_forecast_path(
                            start_value=start_vol,
                            end_value=s_vol,
                            horizon=horizon_display + 1,  # +1 to include today for seamless connection
                            volatility=max(0.5, abs(s_vol - start_vol) * 0.5),  # Increased volatility for more fluctuation
                            mean_reversion=0.1  # Reduced mean reversion for more natural movement
                        )
                        # Ensure first point exactly matches last historical point for seamless connection
                        if len(scenario_path) > 0 and len(hist_y) > 0:
                            scenario_path[0] = start_vol
                        
                        # Scenario forecast only (baseline removed)
                        # Use all points including today to connect seamlessly with historical data
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=scenario_path,
                            mode='lines',
                            name='Scenario Forecast',
                            line=dict(color='#333333', width=2, dash='dot')
                        ))
                        
                        # Vertical separator at today (middle of x-axis)
                        fig.add_vline(
                            x=today_date,
                            line_dash="dot",
                            line_color="#999999",
                            line_width=1
                        )
                        
                        # Fed rate annotation
                        fed_rate_change = st.session_state.get('fed_rate_change', 0)
                        if fed_rate_change != 0:
                            fig.add_annotation(
                                x=forecast_dates[len(forecast_dates)//2],
                                y=s_vol * 1.1,
                                text=f"Fed Rate: {fed_rate_change:+.0f} bps",
                                showarrow=False,
                                font=dict(color='#000000', size=10)
                            )
                        
                        # Calculate y-axis range to show all data
                        y_min, y_max = calculate_y_axis_range(hist_y + scenario_path)
                        
                        fig.update_layout(
                            title="",
                            xaxis_title="",
                            yaxis_title="Volatility (%)",
                            yaxis=dict(range=[y_min, y_max]),
                            xaxis=dict(range=[x_axis_start, x_axis_end]),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='black', size=11),
                            margin=dict(l=40, r=20, t=20, b=40),
                            hovermode='x unified',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating Fed Rate scenario chart: {str(e)}")
                        with st.expander("Error Details"):
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.warning("No historical data available for chart display.")
            
            # Statistical Summary Table
            with st.expander("Statistical Summary", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Historical Context**")
                    stats = calculate_forecast_statistics(hist_vol, scenario_vol, current_vol)
                    stats_data = {
                        'Metric': ['30-day Average', '60-day Average', '90-day Average', 'Current Volatility', 'Forecast Volatility'],
                        'Value (%)': [
                            f"{stats['avg_30d']:.2f}",
                            f"{stats['avg_60d']:.2f}",
                            f"{stats['avg_90d']:.2f}",
                            f"{stats['current_volatility']:.2f}",
                            f"{stats['forecast_volatility']:.2f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                
                with col2:
                    st.write("**Forecast Details**")
                    percentile_info = calculate_percentile_ranking(scenario_vol, hist_vol)
                    details_data = {
                        'Metric': ['Forecast Value', 'Percentile Ranking', 'Change from Current', 'Change Percentage'],
                        'Value': [
                            f"{scenario_vol:.2f}%",
                            f"{percentile_info['percentile']:.1f}th percentile",
                            f"{stats['change_abs']:+.2f}%",
                            f"{stats['change_pct']:+.2f}%"
                        ]
                    }
                    st.dataframe(pd.DataFrame(details_data), use_container_width=True, hide_index=True)
        
        # Show Fed rate change info
        fed_rate_change = st.session_state.get('fed_rate_change', 0)
        if fed_rate_change != 0:
            st.info(f"Scenario: Fed rate {'increase' if fed_rate_change > 0 else 'decrease'} of {abs(fed_rate_change)} basis points ({abs(fed_rate_change)/100:.2f}%)")
            
            # Model Information Panel
            with st.expander("Model Information", expanded=False):
                model_info = result.get('model_info', {})
                st.write(f"**Model Type:** Fed Rate Scenario Analysis")
                st.write(f"**Prediction Horizon:** {horizon_display} days")
                if 'device' in model_info:
                    st.write(f"**Device:** {model_info['device']}")
            
            # Export Functionality
            if st.button("Download Forecast", type="primary"):
                export_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Scenario_Forecast': [scenario_vol] * len(forecast_dates),
                    'Impact': [impact] * len(forecast_dates)
                })
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"forecast_{symbol}_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
        elif forecast_type_display == 'itransformer' and 'predictions' in result:
            preds = result['predictions']
            
            # Single horizon - use the selected horizon or first available
            if f'{horizon_display}d' in preds:
                forecast_value = preds[f'{horizon_display}d']
            else:
                # Fallback to first available prediction
                first_key = list(preds.keys())[0] if preds else None
                if first_key:
                    forecast_value = preds[first_key]
                else:
                    forecast_value = 0
            
            # Get implied volatility matching forecast horizon
            symbol = st.session_state.get('forecast_symbol', '').upper()
            current_vol = get_implied_volatility(symbol, df, horizon_days=horizon_display, use_api=True)
            if current_vol is None:
                # Fallback to historical volatility if IV unavailable
                # Historical volatility is in decimal form, convert to percentage
                current_vol = float(hist_vol.iloc[-1]) * 100 if len(hist_vol) > 0 else 0.0
            main_forecast = forecast_value
            
            # Key Metrics Dashboard
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Volatility", f"{current_vol:.2f}%")
            with col2:
                st.metric("Forecast Value", f"{main_forecast:.2f}%")
            with col3:
                change = main_forecast - current_vol
                st.metric("Change", f"{change:+.2f}%", delta=f"{change:+.2f}%")
            with col4:
                # Convert hist_vol to percentage for regime classification
                hist_vol_pct_for_regime = hist_vol * 100 if len(hist_vol) > 0 else pd.Series(dtype=float)
                regime, _ = classify_volatility_regime(main_forecast, hist_vol_pct_for_regime)
                alert_level, _ = get_alert_level(main_forecast, hist_vol_pct_for_regime)
                st.metric("Forecast Regime", regime)
            
            # Forecast Chart
            if not PLOTLY_AVAILABLE:
                st.warning("Plotly required for charts. Install with: pip install plotly")
            elif len(hist_vol) == 0:
                st.warning("Unable to calculate historical volatility for chart display.")
            else:
                # Convert historical volatility from decimal to percentage
                hist_vol_pct = hist_vol * 100
                
                # Calculate confidence intervals (using percentage values)
                conf_intervals = estimate_confidence_intervals(main_forecast, hist_vol_pct)
                
                # Create forecast chart using visualization function
                try:
                    fig = create_forecast_chart(
                        historical_volatility=hist_vol_pct,
                        forecast_value=main_forecast,
                        forecast_horizon=horizon_display,
                        confidence_intervals=conf_intervals,
                        historical_days=90,
                        title=""
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating forecast chart: {str(e)}")
                    with st.expander("Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
            
            # Statistical Summary Table
            with st.expander("Statistical Summary", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Historical Context**")
                    stats = calculate_forecast_statistics(hist_vol, main_forecast, current_vol)
                    stats_data = {
                        'Metric': ['30-day Average', '60-day Average', '90-day Average', 'Current Volatility', 'Forecast Volatility'],
                        'Value (%)': [
                            f"{stats['avg_30d']:.2f}",
                            f"{stats['avg_60d']:.2f}",
                            f"{stats['avg_90d']:.2f}",
                            f"{stats['current_volatility']:.2f}",
                            f"{stats['forecast_volatility']:.2f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                
                with col2:
                    st.write("**Forecast Details**")
                    percentile_info = calculate_percentile_ranking(main_forecast, hist_vol)
                    details_data = {
                        'Metric': [f'{horizon_display}d Forecast', 'Percentile Ranking', 'Change from Current', 'Change Percentage'],
                        'Value': [
                            f"{main_forecast:.2f}%",
                            f"{percentile_info['percentile']:.1f}th percentile",
                            f"{stats['change_abs']:+.2f}%",
                            f"{stats['change_pct']:+.2f}%"
                        ]
                    }
                    details_df = pd.DataFrame(details_data)
                    st.dataframe(details_df, use_container_width=True, hide_index=True)
            
            # Volatility Percentile Ranking
            with st.expander("Percentile Ranking", expanded=False):
                percentile_info = calculate_percentile_ranking(main_forecast, hist_vol)
                percentile_val = percentile_info['percentile']
                
                # Horizontal bar
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[percentile_val],
                    y=['Percentile'],
                    orientation='h',
                    marker=dict(color='#000000'),
                    text=[f"{percentile_val:.1f}th percentile"],
                    textposition='inside'
                ))
                fig.update_layout(
                    title="",
                    xaxis_title="Percentile (%)",
                    xaxis_range=[0, 100],
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black', size=11),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=100
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"**{percentile_info['context']}**")
            
            # Alert Thresholds
            alert_level, alert_color = get_alert_level(main_forecast, hist_vol)
            if alert_level == "extreme":
                st.warning(f"⚠️ Extreme Volatility Forecast: {main_forecast:.2f}% is in the 90th+ percentile of historical volatility")
            elif alert_level == "elevated":
                st.info(f"ℹ️ Elevated Volatility Forecast: {main_forecast:.2f}% is in the 75th-90th percentile of historical volatility")
            
            # Model Information Panel
            with st.expander("Model Information", expanded=False):
                model_info = result.get('model_info', {})
                st.write(f"**Model Type:** iTransformer")
                st.write(f"**Prediction Horizon:** {horizon_display}d")
                if 'seq_length' in model_info:
                    st.write(f"**Sequence Length:** {model_info['seq_length']}")
                if 'device' in model_info:
                    st.write(f"**Device:** {model_info['device']}")
                if 'n_features' in model_info:
                    st.write(f"**Input Features:** {model_info['n_features']}")
            
            # Export Functionality
            if st.button("Download Forecast", type="primary"):
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon_display,
                    freq='D'
                )
                export_data = {
                    'Date': forecast_dates,
                    f'Forecast_{horizon_display}d': [main_forecast] * len(forecast_dates)
                }
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"forecast_{symbol}_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        elif forecast_type_display == 'neural_garch' and 'volatility' in result:
            # Show forecast if available
            horizon_display = st.session_state.get('forecast_horizon', 20)
            
            if 'forecast' in result and result['forecast'] is not None:
                # Show forecasted volatility
                forecast_df = result['forecast']
                forecasted_vol = result.get('forecasted_volatility', 0)
                # Get implied volatility matching forecast horizon
                symbol = st.session_state.get('forecast_symbol', '')
                current_vol = get_implied_volatility(symbol, df, horizon_days=horizon_display, use_api=True)
                if current_vol is None:
                    # Fallback to result's current_volatility if IV unavailable
                    current_vol = result.get('current_volatility', 0)
                
                # Get forecast dates for export
                if 'date' in forecast_df.columns:
                    forecast_dates = pd.to_datetime(forecast_df['date']).tolist()
                else:
                    # Generate forecast dates from last historical date
                    vol_df = result['volatility']
                    if len(vol_df) > 0:
                        last_date = pd.to_datetime(vol_df['date'].iloc[-1])
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_df), freq='D').tolist()
                    else:
                        forecast_dates = []
                
                # Key Metrics Dashboard
                st.subheader("Key Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Volatility", f"{current_vol:.2f}%")
                with col2:
                    st.metric("Forecast Value", f"{forecasted_vol:.2f}%")
                with col3:
                    change = forecasted_vol - current_vol
                    st.metric("Change", f"{change:+.2f}%", delta=f"{change:+.2f}%")
                with col4:
                    regime, _ = classify_volatility_regime(forecasted_vol, hist_vol)
                    st.metric("Forecast Regime", regime)
                
                # Historical + Forecast Chart
                if PLOTLY_AVAILABLE:
                    vol_df = result['volatility']
                    
                    # Get recent historical data
                    hist_vol_series = pd.Series(vol_df['volatility'].values, index=pd.to_datetime(vol_df['date']))
                    hist_vol_recent = hist_vol_series.iloc[-90:] if len(hist_vol_series) >= 90 else hist_vol_series
                    hist_x, hist_y = clean_series_for_plotly(hist_vol_recent)
                    
                    # Calculate confidence intervals (needed for both chart and export)
                    conf_intervals = estimate_confidence_intervals(forecasted_vol, hist_vol_series)
                    c_low = conf_intervals['lower']
                    c_high = conf_intervals['upper']
                    
                    if hist_x:
                        fig = go.Figure()
                        
                        # Historical volatility
                        fig.add_trace(go.Scatter(
                            x=hist_x,
                            y=hist_y,
                            mode='lines',
                            name='Historical Volatility',
                            line=dict(color='#000000', width=1.5)
                        ))
                        
                        # Current marker
                        fig.add_trace(go.Scatter(
                            x=[hist_x[-1]],
                            y=[hist_y[-1]],
                            mode='markers',
                            name='Current',
                            marker=dict(color='#000000', size=10, symbol='circle')
                        ))
                        
                        # Forecast - use daily values from forecast_df
                        if 'date' in forecast_df.columns and len(forecast_df) > 0:
                            forecast_series = pd.Series(forecast_df['volatility'].values, index=pd.to_datetime(forecast_df['date']))
                            f_x, f_y = clean_series_for_plotly(forecast_series)
                            # Ensure last value matches forecasted_vol
                            if len(f_y) > 0 and forecasted_vol is not None:
                                f_y[-1] = forecasted_vol
                        else:
                            # Generate dates if not present and create fluctuating path
                            last_hist_date = pd.to_datetime(hist_x[-1]) if hist_x else pd.Timestamp.today()
                            forecast_dates_for_chart = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), periods=horizon_display, freq='D')
                            from src.visualization.forecast_viz import generate_forecast_path
                            start_vol = hist_y[-1] if len(hist_y) > 0 else forecasted_vol
                            forecast_path = generate_forecast_path(
                                start_value=start_vol,
                                end_value=forecasted_vol if forecasted_vol is not None else start_vol,
                                horizon=horizon_display,
                                volatility=abs((forecasted_vol if forecasted_vol is not None else start_vol) - start_vol) * 0.15 / horizon_display if horizon_display > 0 else 0.02,
                                mean_reversion=0.15
                            )
                            f_x = forecast_dates_for_chart.tolist()
                            f_y = forecast_path
                        
                        if f_x:
                            fig.add_trace(go.Scatter(
                                x=f_x,
                                y=f_y,
                                mode='lines',
                                name=f'{horizon_display}-Day Forecast',
                                line=dict(color='#666666', width=2, dash='dash')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=list(f_x) + list(f_x[::-1]),
                                y=[c_low] * len(f_x) + [c_high] * len(f_x),
                                fill='toself',
                                fillcolor='rgba(100,100,100,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% Confidence Interval',
                                showlegend=True
                            ))
                            
                            # Vertical separator
                            fig.add_vline(
                                x=hist_x[-1],
                                line_dash="dot",
                                line_color="#999999",
                                line_width=1
                            )
                            
                            # Calculate y-axis range
                            y_min, y_max = calculate_y_axis_range(hist_y + f_y + [c_low, c_high])
                        else:
                            # Calculate y-axis range without forecast
                            y_min, y_max = calculate_y_axis_range(hist_y)
                        
                        fig.update_layout(
                            title="",
                            xaxis_title="",
                            yaxis_title="Volatility (%)",
                            yaxis=dict(range=[y_min, y_max]),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='black', size=11),
                            margin=dict(l=40, r=20, t=20, b=40),
                            hovermode='x unified',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Plotly required for charts. Install with: pip install plotly")
                
                # Statistical Summary Table
                with st.expander("Statistical Summary", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Historical Context**")
                        stats = calculate_forecast_statistics(hist_vol_series, forecasted_vol, current_vol)
                        stats_data = {
                            'Metric': ['30-day Average', '60-day Average', '90-day Average', 'Current Volatility', 'Forecast Volatility'],
                            'Value (%)': [
                                f"{stats['avg_30d']:.2f}",
                                f"{stats['avg_60d']:.2f}",
                                f"{stats['avg_90d']:.2f}",
                                f"{stats['current_volatility']:.2f}",
                                f"{stats['forecast_volatility']:.2f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.write("**Forecast Details**")
                        percentile_info = calculate_percentile_ranking(forecasted_vol, hist_vol_series)
                        details_data = {
                            'Metric': ['Forecast Value', 'Percentile Ranking', 'Change from Current', 'Change Percentage'],
                            'Value': [
                                f"{forecasted_vol:.2f}%",
                                f"{percentile_info['percentile']:.1f}th percentile",
                                f"{stats['change_abs']:+.2f}%",
                                f"{stats['change_pct']:+.2f}%"
                            ]
                        }
                        st.dataframe(pd.DataFrame(details_data), use_container_width=True, hide_index=True)
                
                # Model Information Panel
                with st.expander("Model Information", expanded=False):
                    model_info = result.get('model_info', {})
                    st.write(f"**Model Type:** Neural GARCH")
                    st.write(f"**Prediction Horizon:** {horizon_display} days")
                    if 'p' in model_info:
                        st.write(f"**GARCH(p,q):** ({model_info['p']}, {model_info.get('q', 1)})")
                    if 'device' in model_info:
                        st.write(f"**Device:** {model_info['device']}")
                
                # Export Functionality
                if st.button("Download Forecast", type="primary"):
                    export_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Historical_Vol': [current_vol] + [None] * (len(forecast_dates) - 1),
                        'Forecast_Vol': forecast_df['volatility'].values,
                        'Confidence_Lower': [conf_intervals['lower']] * len(forecast_dates),
                        'Confidence_Upper': [conf_intervals['upper']] * len(forecast_dates)
                    })
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"forecast_{symbol}_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                # Show historical only (no forecast)
                if not PLOTLY_AVAILABLE:
                    st.error("Plotly required for charts. Install with: pip install plotly")
                else:
                    vol_df = result['volatility']
                    fig = go.Figure()
                    
                    # Clean data
                    vol_series = pd.Series(vol_df['volatility'].values, index=pd.to_datetime(vol_df['date']))
                    x, y = clean_series_for_plotly(vol_series)
                    
                    if x:
                        fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            name="Neural GARCH",
                            line=dict(color='#000000', width=1)
                        ))
                        
                        # Calculate y-axis range
                        y_min, y_max = calculate_y_axis_range(y)
                        
                        fig.update_layout(
                            title="",
                            xaxis_title="",
                            yaxis_title="Volatility (%)",
                            yaxis=dict(range=[y_min, y_max]),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='black', size=11),
                            margin=dict(l=40, r=20, t=20, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Show current volatility metric if no forecast was shown
            if 'forecast' not in result or result['forecast'] is None:
                # Get implied volatility matching forecast horizon
                symbol = st.session_state.get('forecast_symbol', '')
                horizon_display = st.session_state.get('forecast_horizon', 20)
                current_vol = get_implied_volatility(symbol, df, horizon_days=horizon_display, use_api=True)
                if current_vol is None:
                    # Fallback to result's current_volatility if IV unavailable
                    current_vol = result.get('current_volatility', 0)
                if current_vol > 0:
                    st.metric("Current Volatility", f"{current_vol:.2f}%")


# ============================================================================
# Asset Comparison Views
# ============================================================================

def render_asset_comparison_volatility():
    """Render Asset Comparison → Volatility Comparison view."""
    st.header("Volatility Comparison")
    
    # Asset selection
    col1, col2, col3 = st.columns([2, 2, 2])
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        symbol_a = st.text_input("Asset A", value="AAPL", key="comp_a").upper()
    with col2:
        symbol_b = st.text_input("Asset B", value="MSFT", key="comp_b").upper()
    with col3:
        benchmark = st.text_input("Benchmark (optional)", value="SPY", key="comp_bench").upper()
    col1, col2 = st.columns(2)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01').date(), key="comp_start")
    with col2:
        end_date = st.date_input("End Date", value=date.today(), key="comp_end")
    if st.button("Load Data", type="primary", key="comp_load"):
        with st.spinner("Loading data..."):
            df_a, error_a = load_market_data(symbol_a, start_date, end_date)
            df_b, error_b = load_market_data(symbol_b, start_date, end_date)
            df_bench, error_bench = load_market_data(benchmark, start_date, end_date) if benchmark else (None, None)
            
            if df_a is not None and df_b is not None:
                st.session_state['comp_data_a'] = df_a
                st.session_state['comp_data_b'] = df_b
                st.session_state['comp_symbol_a'] = symbol_a
                st.session_state['comp_symbol_b'] = symbol_b
                if df_bench is not None:
                    st.session_state['comp_data_bench'] = df_bench
                st.success("Data loaded")
            else:
                errors = []
                if error_a:
                    errors.append(f"{symbol_a}: {error_a.split(chr(10))[0]}")
                if error_b:
                    errors.append(f"{symbol_b}: {error_b.split(chr(10))[0]}")
                if error_bench:
                    errors.append(f"{benchmark}: {error_bench.split(chr(10))[0]}")
                
                if errors:
                    st.error(f"Failed to load data:\n" + "\n".join(errors))
                    with st.expander("Error Details"):
                        if error_a:
                            st.code(f"{symbol_a}:\n{error_a}")
                        if error_b:
                            st.code(f"{symbol_b}:\n{error_b}")
                        if error_bench:
                            st.code(f"{benchmark}:\n{error_bench}")
                else:
                    st.error("Failed to load data: No data returned")
                    return
    
    if 'comp_data_a' not in st.session_state or 'comp_data_b' not in st.session_state:
        st.info("Enter asset symbols and click 'Load Data'")
        return
        
    df_a = st.session_state['comp_data_a']
    df_b = st.session_state['comp_data_b']
    symbol_a = st.session_state['comp_symbol_a']
    symbol_b = st.session_state['comp_symbol_b']
    
    # Method selection
    st.subheader("Method")
    available_methods = list_estimators()
    selected_method = st.selectbox(
        "Select Method",
        available_methods,
        format_func=format_estimator_name,
        index=available_methods.index('yang_zhang') if 'yang_zhang' in available_methods else 0,
        key="comp_method"
    )
    
    # Parameters
    window = st.slider("Rolling Window (days)", 10, 252, 60, key="comp_window")
    lambda_param = st.slider("EWMA Lambda", 0.80, 0.99, 0.94, 0.01, key="comp_lambda") if selected_method == 'ewma' else 0.94
    
    if st.button("Calculate", type="primary", key="comp_calc"):
        config = load_config()
        vol_config = config.get('volatility', {})
        annualization = vol_config.get('annualization_factor', 252)
        
        try:
            if selected_method == 'ewma':
                estimator = get_estimator(selected_method, window, annualization, lambda_param=lambda_param)
            else:
                estimator = get_estimator(selected_method, window, annualization)
            
            vol_a = estimator.compute(df_a, annualize=True)
            vol_b = estimator.compute(df_b, annualize=True)
            
            st.session_state['comp_vol_a'] = vol_a
            st.session_state['comp_vol_b'] = vol_b
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")
    
    if 'comp_vol_a' in st.session_state and 'comp_vol_b' in st.session_state:
        vol_a = st.session_state['comp_vol_a']
        vol_b = st.session_state['comp_vol_b']
        
        # Align indices
        common_idx = vol_a.index.intersection(vol_b.index)
        vol_a_aligned = vol_a.loc[common_idx].dropna()
        vol_b_aligned = vol_b.loc[common_idx].dropna()
        
        # Chart
        if not PLOTLY_AVAILABLE:
            st.error("Plotly required for charts. Install with: pip install plotly")
        else:
                fig = go.Figure()
                
                # Clean data for Asset A
                x_a, y_a = clean_series_for_plotly(vol_a_aligned)
                if x_a:
                    fig.add_trace(go.Scatter(
                        x=x_a,
                        y=y_a,
                        name=symbol_a,
                        line=dict(color='#000000', width=1)
                    ))
                
                # Clean data for Asset B
                x_b, y_b = clean_series_for_plotly(vol_b_aligned)
                if x_b:
                    fig.add_trace(go.Scatter(
                        x=x_b,
                        y=y_b,
                        name=symbol_b,
                        line=dict(color='#666666', width=1)
                    ))
                
                # Calculate y-axis range
                all_y = []
                if y_a:
                    all_y.extend(y_a)
                if y_b:
                    all_y.extend(y_b)
                y_min, y_max = calculate_y_axis_range(all_y)
                
                fig.update_layout(
                    title="",
                    xaxis_title="",
                    yaxis_title="Volatility (%)",
                    yaxis=dict(range=[y_min, y_max]),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black', size=11),
                    hovermode='x unified',
                    margin=dict(l=40, r=20, t=20, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("Volatility Relationship")
        correlation = 0.0
        covariance = 0.0
        
        if len(vol_a_aligned) > 0 and len(vol_b_aligned) > 0:
            # Align for correlation
            final_idx = vol_a_aligned.index.intersection(vol_b_aligned.index)
            if len(final_idx) > 1:
                vol_a_final = vol_a_aligned.loc[final_idx]
                vol_b_final = vol_b_aligned.loc[final_idx]
                
                correlation = vol_a_final.corr(vol_b_final)
                covariance = np.cov(vol_a_final, vol_b_final)[0, 1]
            else:
                st.warning("Insufficient data for correlation calculation (need at least 2 overlapping points)")
        else:
            st.warning("No aligned volatility data available")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vol Correlation", f"{correlation:.3f}")
        with col2:
            st.metric("Covariance", f"{covariance:.6f}")


def render_asset_comparison_risk_return():
    """Render Asset Comparison → Risk & Return Metrics view."""
    st.header("Risk & Return Metrics")
    
    # Asset selection (reuse from volatility comparison if available)
    if 'comp_data_a' in st.session_state:
        symbol_a = st.session_state.get('comp_symbol_a', 'AAPL')
        symbol_b = st.session_state.get('comp_symbol_b', 'MSFT')
        df_a = st.session_state['comp_data_a']
        df_b = st.session_state['comp_data_b']
        df_bench = st.session_state.get('comp_data_bench', None)
        benchmark = st.session_state.get('comp_bench', 'SPY')
    else:
        st.info("Load data in 'Volatility Comparison' tab first")
        return
            
    returns_a = calculate_returns(df_a['close'], method='log')
    returns_b = calculate_returns(df_b['close'], method='log')
    returns_bench = calculate_returns(df_bench['close'], method='log') if df_bench is not None else None
    
    # Calculate metrics
    metrics_data = []
    
    # Average return
    avg_ret_a = returns_a.mean() * 252 * 100  # Annualized %
    avg_ret_b = returns_b.mean() * 252 * 100
    
    # Volatility (using Yang-Zhang)
    config = load_config()
    vol_config = config.get('volatility', {})
    annualization = vol_config.get('annualization_factor', 252)
    estimator = get_estimator('yang_zhang', 60, annualization)
    vol_a = estimator.compute(df_a, annualize=True).mean()
    vol_b = estimator.compute(df_b, annualize=True).mean()
    
    # Beta
    beta_a = calculate_beta(returns_a, returns_bench) if returns_bench is not None else 0.0
    beta_b = calculate_beta(returns_b, returns_bench) if returns_bench is not None else 0.0
    
    # Sharpe ratio
    sharpe_a = calculate_sharpe_ratio(returns_a)
    sharpe_b = calculate_sharpe_ratio(returns_b)
    
    # Treynor ratio
    treynor_a = calculate_treynor_ratio(returns_a, returns_bench) if returns_bench is not None and beta_a != 0 else 0.0
    treynor_b = calculate_treynor_ratio(returns_b, returns_bench) if returns_bench is not None and beta_b != 0 else 0.0
    
    max_dd_a = calculate_max_drawdown(df_a['close'])
    max_dd_b = calculate_max_drawdown(df_b['close'])
    
    # Metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['Avg Return (%)', 'Volatility (%)', f'Beta (vs {benchmark})', 'Sharpe Ratio', 'Treynor Ratio', 'Max Drawdown (%)'],
        symbol_a: [f"{avg_ret_a:.2f}", f"{vol_a:.2f}", f"{beta_a:.2f}", f"{sharpe_a:.2f}", f"{treynor_a:.2f}", f"{max_dd_a:.2f}"],
        symbol_b: [f"{avg_ret_b:.2f}", f"{vol_b:.2f}", f"{beta_b:.2f}", f"{sharpe_b:.2f}", f"{treynor_b:.2f}", f"{max_dd_b:.2f}"],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Correlation matrix
    if returns_bench is not None:
        st.subheader("Correlation Matrix")
        # Align all returns
        common_idx = returns_a.index.intersection(returns_b.index).intersection(returns_bench.index)
        if len(common_idx) > 1:
            returns_a_clean = returns_a.loc[common_idx].dropna()
            returns_b_clean = returns_b.loc[common_idx].dropna()
            returns_bench_clean = returns_bench.loc[common_idx].dropna()
            
            # Final alignment
            final_idx = returns_a_clean.index.intersection(returns_b_clean.index).intersection(returns_bench_clean.index)
            if len(final_idx) > 1:
                corr_matrix = pd.DataFrame({
                    symbol_a: [1.0, returns_a_clean.loc[final_idx].corr(returns_b_clean.loc[final_idx]), returns_a_clean.loc[final_idx].corr(returns_bench_clean.loc[final_idx])],
                    symbol_b: [returns_b_clean.loc[final_idx].corr(returns_a_clean.loc[final_idx]), 1.0, returns_b_clean.loc[final_idx].corr(returns_bench_clean.loc[final_idx])],
                    benchmark: [returns_bench_clean.loc[final_idx].corr(returns_a_clean.loc[final_idx]), returns_bench_clean.loc[final_idx].corr(returns_b_clean.loc[final_idx]), 1.0],
                }, index=[symbol_a, symbol_b, benchmark])
                
                # Heatmap
                if not PLOTLY_AVAILABLE:
                    st.dataframe(corr_matrix, use_container_width=True)
        else:
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='Greys',
                        text=corr_matrix.values,
                        texttemplate='%{text:.2f}',
                        textfont={"color": "black"},
                        showscale=False
                    ))
                    fig.update_layout(
                        title="",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='black', size=11),
                        margin=dict(l=40, r=20, t=20, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)


def render_asset_comparison_forecast():
    """Render Asset Comparison → Forecast Comparison view."""
    st.header("Forecast Comparison")
    
    if not DL_AVAILABLE:
        st.warning("Deep learning not available. Install PyTorch for forecasting.")
        return
    
    # Reuse data from comparison
    if 'comp_data_a' in st.session_state:
        symbol_a = st.session_state.get('comp_symbol_a', 'AAPL')
        symbol_b = st.session_state.get('comp_symbol_b', 'MSFT')
        df_a = st.session_state['comp_data_a']
        df_b = st.session_state['comp_data_b']
    else:
        st.info("Load data in 'Volatility Comparison' tab first")
        return
    
    # Forecast configuration
    st.subheader("Forecast Configuration")
    horizon = st.number_input("Forecast Horizon (days)", min_value=1, max_value=252, value=20, step=1, key="comp_forecast_horizon")
    
    # Fed Rate Effect Configuration
    st.subheader("Fed Rate Effect")
    include_fed_rate = st.checkbox("Include Fed Rate Effect", value=False, key="comp_include_fed_rate")
    fed_rate_change_bps = 0
    if include_fed_rate:
        fed_rate_change_bps = st.slider(
            "Fed Rate Change Scenario (basis points)", 
            min_value=-100, 
            max_value=100, 
            value=0, 
            step=25,
            key="comp_fed_rate_slider",
            help="Positive = rate increase, Negative = rate decrease. 25 bps = 0.25%"
        )
    
    if st.button("Generate Forecasts", type="primary", key="comp_forecast_generate"):
        with st.spinner("Generating forecasts..."):
            if include_fed_rate:
                # Use Fed rate scenario analysis for both assets
                result_a = analyze_fed_rate_scenario(df_a, fed_rate_change_bps, prediction_horizon=horizon, device='auto')
                result_b = analyze_fed_rate_scenario(df_b, fed_rate_change_bps, prediction_horizon=horizon, device='auto')
                
                if 'error' not in result_a and 'error' not in result_b and result_a.get('results') and result_b.get('results'):
                    st.session_state['forecast_a'] = result_a
                    st.session_state['forecast_b'] = result_b
                    st.session_state['comp_forecast_type'] = 'fed_rate'
                    st.session_state['comp_last_forecast_horizon'] = horizon
                    st.session_state['comp_fed_rate_change'] = fed_rate_change_bps
                else:
                    error_msg = result_a.get('error', result_b.get('error', 'Unknown error'))
                    st.error(f"Forecast generation failed: {error_msg}")
                return
            else:
                # Standard iTransformer forecasts
                result_a = predict_volatility_dl(df_a, prediction_horizon=horizon, device='auto')
                result_b = predict_volatility_dl(df_b, prediction_horizon=horizon, device='auto')
                
                if 'error' not in result_a and 'error' not in result_b:
                    st.session_state['forecast_a'] = result_a
                    st.session_state['forecast_b'] = result_b
                    st.session_state['comp_forecast_type'] = 'itransformer'
                    st.session_state['comp_last_forecast_horizon'] = horizon
                else:
                    st.error("Forecast generation failed")
                    return
    
    if 'forecast_a' not in st.session_state or 'forecast_b' not in st.session_state:
        st.info("Generate forecasts to see results")
        return
        
    result_a = st.session_state['forecast_a']
    result_b = st.session_state['forecast_b']
    forecast_type_comp = st.session_state.get('comp_forecast_type', 'itransformer')
    horizon_display = st.session_state.get('comp_last_forecast_horizon', horizon)
    
    if forecast_type_comp == 'fed_rate' and 'results' in result_a and 'results' in result_b:
        # Fed rate scenario results
        results_a = result_a['results']
        results_b = result_b['results']
        
        pred_a = results_a.get('scenario_volatility', 0)
        pred_b = results_b.get('scenario_volatility', 0)
        baseline_a = results_a.get('baseline_volatility', 0)
        baseline_b = results_b.get('baseline_volatility', 0)
        spread = pred_a - pred_b
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{symbol_a} Forecast", f"{pred_a:.2f}%", delta=f"{pred_a - baseline_a:+.2f}%")
        with col2:
            st.metric(f"{symbol_b} Forecast", f"{pred_b:.2f}%", delta=f"{pred_b - baseline_b:+.2f}%")
        with col3:
            st.metric("Forecast Spread", f"{spread:+.2f}%")
        
        fed_rate_change = st.session_state.get('comp_fed_rate_change', 0)
        if fed_rate_change != 0:
            st.info(f"Fed rate scenario: {'increase' if fed_rate_change > 0 else 'decrease'} of {abs(fed_rate_change)} bps ({abs(fed_rate_change)/100:.2f}%)")
    else:
        # Standard forecasts
        pred_a = result_a.get('predictions', {}).get(f'{horizon_display}d', 0)
        pred_b = result_b.get('predictions', {}).get(f'{horizon_display}d', 0)
        spread = pred_a - pred_b
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{symbol_a} Forecast", f"{pred_a:.2f}%")
        with col2:
            st.metric(f"{symbol_b} Forecast", f"{pred_b:.2f}%")
        with col3:
            st.metric("Forecast Spread", f"{spread:+.2f}%")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application entry point."""
    st.title("Volatility Estimator")
    
    # Top-level navigation
    tab1, tab2 = st.tabs(["Single Asset", "Asset Comparison"])
    
    with tab1:
        # Sub-tabs for Single Asset
        sub_tab1, sub_tab2 = st.tabs(["Volatility Analysis", "Volatility Forecasting"])
        with sub_tab1:
            render_single_asset_analysis()
        with sub_tab2:
            render_single_asset_forecasting()
    
    with tab2:
        # Sub-tabs for Asset Comparison
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Volatility Comparison", "Risk & Return Metrics", "Forecast Comparison"])
        with sub_tab1:
            render_asset_comparison_volatility()
        with sub_tab2:
            render_asset_comparison_risk_return()
        with sub_tab3:
            render_asset_comparison_forecast()


if __name__ == "__main__":
    main()
