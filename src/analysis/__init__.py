"""
Analysis module for volatility estimation and comparison.

Includes:
- Multi-estimator comparison
- Event impact analysis
- Statistical analysis (correlation, MSE)
- Risk metrics (Sharpe, Beta, Max Drawdown, VaR)
"""

from src.analysis.comparison import (
    run_all_estimators,
    calculate_correlation_matrix,
    generate_comparison_statistics,
)
from src.analysis.event_analysis import (
    analyze_all_events,
    analyze_event_impact,
)
from src.analysis.statistics import (
    calculate_mse_matrix,
    calculate_summary_statistics,
)
from src.analysis.risk_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_beta,
    calculate_treynor_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_all_risk_metrics,
)

__all__ = [
    # Comparison
    'run_all_estimators',
    'calculate_correlation_matrix',
    'generate_comparison_statistics',
    # Event analysis
    'analyze_all_events',
    'analyze_event_impact',
    # Statistics
    'calculate_mse_matrix',
    'calculate_summary_statistics',
    # Risk metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_beta',
    'calculate_treynor_ratio',
    'calculate_max_drawdown',
    'calculate_var',
    'calculate_cvar',
    'calculate_all_risk_metrics',
]

