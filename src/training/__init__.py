"""
Training Infrastructure for Deep Learning Models.

This package provides training pipelines using PyTorch Lightning,
including data modules, trainers, and evaluation metrics.
"""

from src.training.trainer import (
    VolatilityTrainer,
    train_volatility_model,
    train_fed_rate_model,
    train_neural_garch,
)
from src.training.data_module import (
    VolatilityDataModule,
    TimeSeriesDataset,
)
from src.training.metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
)

__all__ = [
    'VolatilityTrainer',
    'train_volatility_model',
    'train_fed_rate_model',
    'train_neural_garch',
    'VolatilityDataModule',
    'TimeSeriesDataset',
    'calculate_regression_metrics',
    'calculate_classification_metrics',
]







