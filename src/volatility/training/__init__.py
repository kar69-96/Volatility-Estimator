"""
Training Infrastructure for Deep Learning Models.

This package provides training pipelines using PyTorch Lightning,
including data modules, trainers, and evaluation metrics.
"""

from src.volatility.training.trainer import (
    VolatilityTrainer,
    train_volatility_model,
)
from src.volatility.training.data_module import (
    VolatilityDataModule,
    TimeSeriesDataset,
)
from src.volatility.training.metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
)
from src.volatility.training.data import (
    VolatilityDataset,
    prepare_raw_signal,
    compute_target,
    create_weighted_sampler_for_concat_dataset,
)

__all__ = [
    'VolatilityTrainer',
    'train_volatility_model',
    'VolatilityDataModule',
    'TimeSeriesDataset',
    'calculate_regression_metrics',
    'calculate_classification_metrics',
    'VolatilityDataset',
    'prepare_raw_signal',
    'compute_target',
    'create_weighted_sampler_for_concat_dataset',
]
