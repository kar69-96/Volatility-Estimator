"""
Reporting module for Excel export and visualization.

This module generates professional Excel reports and charts for
volatility analysis results.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter

from src.utils import ensure_directory


def export_to_excel(
    volatility_data: Optional[pd.DataFrame] = None,
    events_data: Optional[pd.DataFrame] = None,
    predictions: Optional[pd.DataFrame] = None,
    backtest_results: Optional[pd.DataFrame] = None,
    output_path: str = './outputs/excel/volatility_report.xlsx'
) -> Path:
    """
    Export volatility analysis results to Excel with multiple sheets.

    Sheets:
    1. Volatility Time Series: All estimators side-by-side
    2. Estimator Comparison: Correlation matrix, statistics table
    3. Event Impact Summary: Event analysis results
    4. Predictions: Upcoming events with forecasts
    5. Historical Accuracy: Backtesting results

    Args:
        volatility_data: DataFrame with volatility estimates (date + estimator columns)
        events_data: DataFrame with event analysis results
        predictions: DataFrame with predicted volatility paths
        backtest_results: DataFrame with backtesting accuracy metrics
        output_path: Path to output Excel file

    Returns:
        Path to created Excel file
    """
    output_file = Path(output_path)
    ensure_directory(output_file.parent)

    # Create workbook
    workbook = xlsxwriter.Workbook(str(output_file))
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#366092',
        'font_color': 'white',
        'border': 1
    })
    number_format = workbook.add_format({'num_format': '0.00'})
    percent_format = workbook.add_format({'num_format': '0.00%'})

    # Sheet 1: Volatility Time Series
    if volatility_data is not None and len(volatility_data) > 0:
        worksheet = workbook.add_worksheet('Volatility Time Series')
        worksheet.write_row(0, 0, volatility_data.columns.tolist(), header_format)

        for row_idx, (_, row) in enumerate(volatility_data.iterrows(), start=1):
            for col_idx, value in enumerate(row):
                if isinstance(value, (int, float)) and not pd.isna(value):
                    worksheet.write(row_idx, col_idx, value, number_format)
                else:
                    worksheet.write(row_idx, col_idx, value)

        worksheet.set_column(0, 0, 12)  # Date column
        worksheet.set_column(1, len(volatility_data.columns) - 1, 15)  # Volatility columns

    # Sheet 2: Estimator Comparison (if multiple estimators)
    if volatility_data is not None and len(volatility_data.columns) > 2:
        worksheet = workbook.add_worksheet('Estimator Comparison')

        # Calculate statistics
        vol_cols = [col for col in volatility_data.columns if col != 'date']
        stats_data = []
        for col in vol_cols:
            vol_series = volatility_data[col].dropna()
            if len(vol_series) > 0:
                stats_data.append({
                    'Estimator': col,
                    'Mean': vol_series.mean(),
                    'Std': vol_series.std(),
                    'Min': vol_series.min(),
                    'Max': vol_series.max()
                })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            worksheet.write_row(0, 0, stats_df.columns.tolist(), header_format)
            for row_idx, (_, row) in enumerate(stats_df.iterrows(), start=1):
                for col_idx, value in enumerate(row):
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        worksheet.write(row_idx, col_idx, value, number_format)
                    else:
                        worksheet.write(row_idx, col_idx, value)

            worksheet.set_column(0, len(stats_df.columns) - 1, 15)

    # Sheet 3: Event Impact Summary
    if events_data is not None and len(events_data) > 0:
        worksheet = workbook.add_worksheet('Event Impact Summary')
        worksheet.write_row(0, 0, events_data.columns.tolist(), header_format)

        for row_idx, (_, row) in enumerate(events_data.iterrows(), start=1):
            for col_idx, value in enumerate(row):
                if isinstance(value, (int, float)) and not pd.isna(value):
                    if 'pct' in events_data.columns[col_idx].lower() or 'change' in events_data.columns[col_idx].lower():
                        worksheet.write(row_idx, col_idx, value / 100, percent_format)
                    else:
                        worksheet.write(row_idx, col_idx, value, number_format)
                else:
                    worksheet.write(row_idx, col_idx, value)

        worksheet.set_column(0, len(events_data.columns) - 1, 15)

    # Sheet 4: Predictions
    if predictions is not None and len(predictions) > 0:
        worksheet = workbook.add_worksheet('Predictions')
        worksheet.write_row(0, 0, predictions.columns.tolist(), header_format)

        for row_idx, (_, row) in enumerate(predictions.iterrows(), start=1):
            for col_idx, value in enumerate(row):
                if isinstance(value, (int, float)) and not pd.isna(value):
                    worksheet.write(row_idx, col_idx, value, number_format)
                else:
                    worksheet.write(row_idx, col_idx, value)

        worksheet.set_column(0, len(predictions.columns) - 1, 15)

    # Sheet 5: Historical Accuracy
    if backtest_results is not None and len(backtest_results) > 0:
        worksheet = workbook.add_worksheet('Historical Accuracy')
        worksheet.write_row(0, 0, backtest_results.columns.tolist(), header_format)

        for row_idx, (_, row) in enumerate(backtest_results.iterrows(), start=1):
            for col_idx, value in enumerate(row):
                if isinstance(value, (int, float)) and not pd.isna(value):
                    worksheet.write(row_idx, col_idx, value, number_format)
                else:
                    worksheet.write(row_idx, col_idx, value)

        worksheet.set_column(0, len(backtest_results.columns) - 1, 15)

    workbook.close()
    return output_file


def plot_volatility_comparison(
    volatility_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    output_path: str = './outputs/charts/volatility_comparison.png',
    crisis_periods: Optional[list] = None
) -> Path:
    """
    Plot volatility comparison chart with all estimators.

    Features:
    - Plot all estimators on one time series (different colors)
    - Add event annotations (vertical lines at event dates)
    - Zoom on crisis periods (2008, 2020) - subplot or inset
    - Professional styling (labels, legend, grid)

    Args:
        volatility_df: DataFrame with date and estimator columns
        events_df: Optional DataFrame with event dates
        output_path: Path to save chart
        crisis_periods: List of crisis period tuples [(start, end, label), ...]

    Returns:
        Path to saved chart
    """
    output_file = Path(output_path)
    ensure_directory(output_file.parent)

    # Prepare data
    if 'date' in volatility_df.columns:
        dates = pd.to_datetime(volatility_df['date'])
        vol_data = volatility_df.drop(columns=['date'])
    else:
        dates = volatility_df.index
        vol_data = volatility_df

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each estimator
    colors = plt.cm.tab10(np.linspace(0, 1, len(vol_data.columns)))
    for i, col in enumerate(vol_data.columns):
        ax.plot(dates, vol_data[col], label=col.replace('_', ' ').title(), color=colors[i], linewidth=1.5)

    # Add event annotations
    if events_df is not None and len(events_df) > 0:
        event_dates = pd.to_datetime(events_df['date'])
        for event_date in event_dates:
            ax.axvline(x=event_date, color='red', alpha=0.3, linestyle='--', linewidth=0.5)

    # Styling
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volatility (Annualized %)', fontsize=12, fontweight='bold')
    ax.set_title('Volatility Estimator Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_predictions(
    actual: Optional[pd.Series] = None,
    predicted: pd.DataFrame = None,
    events: Optional[pd.DataFrame] = None,
    output_path: str = './outputs/charts/predictions.png'
) -> Path:
    """
    Plot prediction visualization with confidence bands.

    Features:
    - Plot predicted vs actual volatility
    - Show confidence bands (shaded area)
    - Annotate events (vertical lines)
    - Different colors for predicted vs actual

    Args:
        actual: Series of actual volatility values
        predicted: DataFrame with predicted, lower_bound, upper_bound columns
        events: Optional DataFrame with event dates
        output_path: Path to save chart

    Returns:
        Path to saved chart
    """
    output_file = Path(output_path)
    ensure_directory(output_file.parent)

    fig, ax = plt.subplots(figsize=(14, 8))

    if predicted is not None and len(predicted) > 0:
        dates = pd.to_datetime(predicted['date'])

        # Plot confidence bands
        if 'lower_bound' in predicted.columns and 'upper_bound' in predicted.columns:
            ax.fill_between(
                dates,
                predicted['lower_bound'],
                predicted['upper_bound'],
                alpha=0.3,
                color='blue',
                label='95% Confidence Band'
            )

        # Plot predicted
        if 'predicted' in predicted.columns:
            ax.plot(dates, predicted['predicted'], 'b-', label='Predicted', linewidth=2)

    # Plot actual if provided
    if actual is not None and len(actual) > 0:
        if hasattr(actual, 'index'):
            actual_dates = pd.to_datetime(actual.index)
        else:
            actual_dates = pd.date_range(start=dates[0], periods=len(actual), freq='D')
        ax.plot(actual_dates, actual, 'r-', label='Actual', linewidth=2, alpha=0.7)

    # Add event annotations
    if events is not None and len(events) > 0:
        event_dates = pd.to_datetime(events['date'])
        for event_date in event_dates:
            ax.axvline(x=event_date, color='orange', alpha=0.5, linestyle='--', linewidth=1)

    # Styling
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volatility (Annualized %)', fontsize=12, fontweight='bold')
    ax.set_title('Volatility Predictions with Confidence Bands', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def generate_summary_report(
    volatility_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    predictions: Optional[pd.DataFrame] = None,
    backtest_metrics: Optional[dict] = None,
    output_path: str = './outputs/summaries/summary_report.txt'
) -> Path:
    """
    Generate text summary report with key statistics.

    Args:
        volatility_df: DataFrame with volatility estimates
        events_df: DataFrame with event analysis
        predictions: DataFrame with predictions
        backtest_metrics: Dictionary with backtesting metrics
        output_path: Path to save report

    Returns:
        Path to saved report
    """
    output_file = Path(output_path)
    ensure_directory(output_file.parent)

    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VOLATILITY ESTIMATOR STACK - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")

        # Volatility statistics
        if volatility_df is not None and len(volatility_df) > 0:
            f.write("VOLATILITY STATISTICS:\n")
            f.write("-"*70 + "\n")
            vol_cols = [col for col in volatility_df.columns if col != 'date']
            for col in vol_cols:
                vol_series = volatility_df[col].dropna()
                if len(vol_series) > 0:
                    f.write(f"{col.replace('_', ' ').title()}:\n")
                    f.write(f"  Mean: {vol_series.mean():.2f}%\n")
                    f.write(f"  Std:  {vol_series.std():.2f}%\n")
                    f.write(f"  Min:  {vol_series.min():.2f}%\n")
                    f.write(f"  Max:  {vol_series.max():.2f}%\n\n")

        # Event impact statistics
        if events_df is not None and len(events_df) > 0:
            f.write("EVENT IMPACT ANALYSIS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Events Analyzed: {len(events_df)}\n")
            significant = events_df['significant'].sum() if 'significant' in events_df.columns else 0
            f.write(f"Significant Events: {significant} ({100*significant/len(events_df):.1f}%)\n")
            if 'volatility_change_pct' in events_df.columns:
                avg_change = events_df['volatility_change_pct'].mean()
                f.write(f"Average Volatility Change: {avg_change:.2f}%\n\n")

        # Prediction accuracy
        if backtest_metrics is not None:
            f.write("PREDICTION ACCURACY:\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Predictions: {backtest_metrics.get('total_predictions', 0)}\n")
            f.write(f"Mean Absolute Error: {backtest_metrics.get('prediction_mae', np.nan):.2f}%\n")
            f.write(f"Root Mean Squared Error: {backtest_metrics.get('prediction_rmse', np.nan):.2f}%\n")
            f.write(f"Correlation: {backtest_metrics.get('prediction_correlation', np.nan):.3f}\n\n")

        f.write("="*70 + "\n")

    return output_file

