"""
Visualization module for volatility analysis.

Includes:
- Forecast visualization utilities
- Chart generation
- Excel export and reporting
"""

from src.visualization.forecast_viz import (
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
    generate_forecast_path,
)
from src.visualization.reporting import (
    export_to_excel,
    plot_volatility_comparison,
    plot_predictions,
    generate_summary_report,
)
from src.visualization.charts import (
    create_line_chart,
    create_comparison_chart,
    create_forecast_chart,
)

__all__ = [
    # Forecast visualization
    'calculate_historical_volatility',
    'classify_volatility_regime',
    'calculate_historical_averages',
    'calculate_percentile_ranking',
    'estimate_confidence_intervals',
    'get_alert_level',
    'prepare_forecast_dataframe',
    'calculate_forecast_statistics',
    'calculate_y_axis_range',
    'clean_series_for_plotly',
    'generate_forecast_path',
    # Reporting
    'export_to_excel',
    'plot_volatility_comparison',
    'plot_predictions',
    'generate_summary_report',
    # Charts
    'create_line_chart',
    'create_comparison_chart',
    'create_forecast_chart',
]

