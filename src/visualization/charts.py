"""
Chart generation utilities for volatility visualization.

Provides helper functions for creating consistent, styled charts.
"""

from typing import Dict, List, Optional

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None


def create_line_chart(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: Dict[str, str] = None
):
    """
    Create a line chart with multiple series.
    
    Args:
        data: DataFrame with data
        x_col: Column name for x-axis
        y_cols: List of column names for y-axis
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        colors: Dictionary mapping column names to colors
        
    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for chart generation")
    
    fig = go.Figure()
    
    for col in y_cols:
        color = colors.get(col) if colors else None
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[col],
            mode='lines',
            name=col,
            line=dict(color=color) if color else None
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified'
    )
    
    return fig


def create_comparison_chart(
    data: pd.DataFrame,
    methods: List[str],
    title: str = "Volatility Comparison"
):
    """
    Create a comparison chart for multiple volatility methods.
    
    Args:
        data: DataFrame with date and volatility columns
        methods: List of method column names
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for chart generation")
    
    fig = go.Figure()
    
    for method in methods:
        if method in data.columns:
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data[method],
                mode='lines',
                name=method
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode='x unified'
    )
    
    return fig


def clean_series_for_plotly(series: pd.Series) -> pd.Series:
    """
    Clean a series for Plotly visualization (remove inf/nan).
    
    Args:
        series: Input series
        
    Returns:
        Cleaned series
    """
    import numpy as np
    return series.replace([np.inf, -np.inf], None)


def create_forecast_chart(
    historical_volatility: pd.Series,
    forecast_value: float,
    forecast_horizon: int,
    confidence_intervals: Optional[Dict[str, float]] = None,
    historical_days: int = 90,
    title: str = ""
) -> go.Figure:
    """
    Create a forecast chart with historical volatility and forecast.
    
    Args:
        historical_volatility: Historical volatility series (in percentage form)
        forecast_value: Forecasted volatility value (in percentage)
        forecast_horizon: Number of days to forecast
        confidence_intervals: Optional dict with 'lower' and 'upper' keys
        historical_days: Number of historical days to show
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for chart generation")
    
    from src.visualization.forecast_viz import clean_series_for_plotly, calculate_y_axis_range
    
    # Normalize x-axis: today - forecast_horizon to today + forecast_horizon
    today = pd.Timestamp.today().normalize()
    x_axis_start = today - pd.Timedelta(days=forecast_horizon)
    x_axis_end = today + pd.Timedelta(days=forecast_horizon)
    
    # Filter historical data to x_axis_start to today
    historical_volatility.index = pd.to_datetime(historical_volatility.index)
    hist_vol_filtered = historical_volatility[(historical_volatility.index >= x_axis_start) & (historical_volatility.index <= today)]
    
    # If we don't have enough historical data, pad with the earliest available value
    if len(hist_vol_filtered) < forecast_horizon:
        earliest_value = historical_volatility.iloc[0] if len(historical_volatility) > 0 else 0
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
    
    if not hist_x:
        # Return empty figure if no historical data
        fig = go.Figure()
        fig.update_layout(title=title or "Forecast Chart", xaxis_title="Date", yaxis_title="Volatility (%)")
        return fig
    
    today_date = today
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
    
    # Forecast line - generate fluctuating path (starting from today, no gap)
    forecast_dates = pd.date_range(start=today_date, periods=forecast_horizon + 1, freq='D')
    # Ensure forecast_value is not NaN
    m_forecast = forecast_value if not (pd.isna(forecast_value) or pd.isna(forecast_value)) else hist_y[-1]
    
    # Generate fluctuating forecast path from last historical value to forecast value
    from src.visualization.forecast_viz import generate_forecast_path
    start_vol = hist_y[-1] if len(hist_y) > 0 else m_forecast
    forecast_path = generate_forecast_path(
        start_value=start_vol,
        end_value=m_forecast,
        horizon=forecast_horizon + 1,  # +1 to include today for seamless connection
        volatility=max(0.5, abs(m_forecast - start_vol) * 0.5),  # Increased volatility for more fluctuation
        mean_reversion=0.1  # Reduced mean reversion for more natural movement
    )
    # Keep all points - forecast starts from today (same point as historical end, no gap)
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_path,
        mode='lines',
        name=f'{forecast_horizon}d Forecast',
        line=dict(color='#666666', width=2, dash='dash')
    ))
    
    # Add confidence intervals if provided
    if confidence_intervals:
        c_low = confidence_intervals.get('lower', m_forecast * 0.9)
        c_high = confidence_intervals.get('upper', m_forecast * 1.1)
        
        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=[c_low] * len(forecast_dates) + [c_high] * len(forecast_dates),
            fill='toself',
            fillcolor='rgba(100,100,100,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{forecast_horizon}d 95% CI',
            showlegend=True
        ))
        
        # Calculate y-axis range with confidence intervals
        y_min, y_max = calculate_y_axis_range(hist_y + [m_forecast, c_low, c_high])
    else:
        y_min, y_max = calculate_y_axis_range(hist_y + [m_forecast])
    
    # Vertical separator at today (middle of x-axis)
    fig.add_vline(
        x=today_date,
        line_dash="dot",
        line_color="#999999",
        line_width=1
    )
    
    fig.update_layout(
        title=title,
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
    
    return fig

