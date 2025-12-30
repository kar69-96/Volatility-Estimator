"""
Training Pipeline for Deep Learning Models.

Provides training functions for:
- Volatility prediction models
- Fed rate prediction models
- Neural GARCH models
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
    print_metrics_report,
)

# Lazy imports
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


class VolatilityTrainer:
    """
    Trainer for volatility prediction models.
    
    Handles training loop, validation, checkpointing, and early stopping.
    """
    
    def __init__(
        self,
        model,
        optimizer=None,
        loss_fn=None,
        device: str = 'auto',
        checkpoint_dir: str = './models/checkpoints',
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer (defaults to Adam)
            loss_fn: Loss function (defaults to Huber loss)
            device: Device for training
            checkpoint_dir: Directory for saving checkpoints
        """
        from src.models.base_model import get_device
        
        torch = _get_torch()
        
        self.device = get_device(device)
        self.model = model.to(self.device)
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        if loss_fn is None:
            self.loss_fn = torch.nn.HuberLoss()
        else:
            self.loss_fn = loss_fn
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
    
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training DataLoader
            
        Returns:
            Average training loss
        """
        torch = _get_torch()
        
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X, y in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(X)
            loss = self.loss_fn(predictions, y)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def validate(self, val_loader) -> Tuple[float, Dict]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation DataLoader
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        torch = _get_torch()
        
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                predictions = self.model(X)
                loss = self.loss_fn(predictions, y)
                
                total_loss += loss.item()
                n_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Calculate metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Handle different array shapes
        # Squeeze to remove single-dimensional entries
        predictions = np.squeeze(predictions)
        targets = np.squeeze(targets)
        
        # If both are 2D, use first column/row
        if predictions.ndim > 1 and targets.ndim > 1:
            # Both are 2D - use first column
            pred_flat = predictions[:, 0] if predictions.shape[1] > 1 else predictions.flatten()
            targ_flat = targets[:, 0] if targets.shape[1] > 1 else targets.flatten()
        elif predictions.ndim > 1:
            # Only predictions is 2D
            pred_flat = predictions[:, 0] if predictions.shape[1] > 1 else predictions.flatten()
            targ_flat = targets
        elif targets.ndim > 1:
            # Only targets is 2D
            pred_flat = predictions
            targ_flat = targets[:, 0] if targets.shape[1] > 1 else targets.flatten()
        else:
            # Both are 1D
            pred_flat = predictions
            targ_flat = targets
        
        # Ensure same length
        min_len = min(len(pred_flat), len(targ_flat))
        pred_flat = pred_flat[:min_len]
        targ_flat = targ_flat[:min_len]
        
        metrics = calculate_regression_metrics(targ_flat, pred_flat)
        
        return avg_loss, metrics
    
    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            verbose: Print training progress
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Check early stopping
                if early_stopping(val_loss):
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.6f} - "
                        f"Val Loss: {val_loss:.6f} - "
                        f"MAE: {val_metrics.get('mae', 0):.4f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch = _get_torch()
        
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        torch = _get_torch()
        
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)


def train_volatility_model(
    df: pd.DataFrame,
    model_type: str = 'itransformer',
    seq_length: int = 60,
    prediction_horizons: List[int] = None,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 0.001,
    early_stopping_patience: int = 10,
    device: str = 'auto',
    verbose: bool = True,
) -> Tuple:
    """
    Train volatility prediction model.
    
    Args:
        df: DataFrame with features (from FeatureExtractor)
        model_type: 'itransformer' or 'transformer'
        seq_length: Input sequence length
        prediction_horizons: List of prediction horizons
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of layers
        dropout: Dropout rate
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        early_stopping_patience: Patience for early stopping
        device: Device preference
        verbose: Print progress
        
    Returns:
        Tuple of (trained model, training history, metrics)
    """
    torch = _get_torch()
    
    from src.training.data_module import VolatilityDataModule
    from src.models.volatility_predictor import VolatilityPredictorWrapper
    
    if prediction_horizons is None:
        prediction_horizons = [1, 5, 10, 20]
    
    # Prepare data
    data_module = VolatilityDataModule(
        df=df,
        target_column='realized_vol_20d',
        seq_length=seq_length,
        prediction_horizons=prediction_horizons,
        batch_size=batch_size,
    )
    
    n_features = data_module.n_features
    
    # Create model
    predictor = VolatilityPredictorWrapper(
        n_features=n_features,
        seq_length=seq_length,
        model_type=model_type,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        prediction_horizons=prediction_horizons,
        device=device,
    )
    
    # Create trainer
    optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=lr, weight_decay=0.01)
    trainer = VolatilityTrainer(
        model=predictor.model,
        optimizer=optimizer,
        device=device,
    )
    
    # Train
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
    )
    
    # Evaluate on test set
    test_loader = data_module.get_test_dataloader()
    test_loss, test_metrics = trainer.validate(test_loader)
    
    if verbose:
        print_metrics_report(test_metrics, 'Test Set Metrics')
    
    return predictor, history, test_metrics


def train_fed_rate_model(
    df: pd.DataFrame,
    model_type: str = 'lstm',
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = 'auto',
    verbose: bool = True,
) -> Tuple:
    """
    Train Fed rate prediction model.
    
    Args:
        df: DataFrame with features
        model_type: 'lstm' or 'transformer'
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        device: Device preference
        verbose: Print progress
        
    Returns:
        Tuple of (trained model, history, metrics)
    """
    torch = _get_torch()
    
    from src.models.fed_rate_predictor import FedRatePredictorWrapper
    
    # Prepare data
    feature_cols = [c for c in df.columns if c not in ['date', 'direction', 'rate_change']]
    features = df[feature_cols].values
    targets = df['direction'].values if 'direction' in df.columns else np.zeros(len(df))
    
    n_features = features.shape[1]
    
    # Create model
    predictor = FedRatePredictorWrapper(
        n_features=n_features,
        model_type=model_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )
    
    # Training loop
    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=lr)
    class_weights = torch.tensor([1.0, 1.0, 1.0], device=predictor.device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    history = {'train_loss': []}
    
    predictor.model.train()
    
    for epoch in range(epochs):
        # Simple batch training
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(features) - 60, batch_size):
            batch_end = min(i + batch_size, len(features) - 60)
            
            X_batch = []
            y_batch = []
            
            for j in range(i, batch_end):
                if j + 60 < len(features):
                    X_batch.append(features[j:j+60])
                    y_batch.append(targets[j + 60])
            
            if len(X_batch) == 0:
                continue
            
            X = torch.tensor(np.array(X_batch), dtype=torch.float32, device=predictor.device)
            y = torch.tensor(np.array(y_batch), dtype=torch.long, device=predictor.device)
            
            optimizer.zero_grad()
            
            outputs = predictor.model(X)
            loss = criterion(outputs['class_logits'], y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")
    
    # Evaluate
    predictor.model.eval()
    
    # Get predictions for evaluation
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for i in range(len(features) - 60):
            X = features[i:i+60]
            y = targets[i + 60] if i + 60 < len(targets) else 0
            
            X_tensor = torch.tensor(X[np.newaxis], dtype=torch.float32, device=predictor.device)
            pred = predictor.predict(X_tensor.cpu().numpy())
            
            all_preds.append(pred['predicted_class'][0])
            all_true.append(int(y))
    
    metrics = calculate_classification_metrics(
        np.array(all_true),
        np.array(all_preds),
        class_labels=['Decrease', 'No Change', 'Increase']
    )
    
    if verbose:
        print_metrics_report(metrics, 'Fed Rate Prediction Metrics')
    
    return predictor, history, metrics


def train_neural_garch(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
    hidden_layers: List[int] = None,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = 'auto',
    verbose: bool = True,
) -> Tuple:
    """
    Train Neural GARCH model.
    
    Args:
        returns: Array of returns
        p: ARCH order
        q: GARCH order
        hidden_layers: Hidden layer sizes
        epochs: Number of epochs
        lr: Learning rate
        device: Device preference
        verbose: Print progress
        
    Returns:
        Tuple of (trained model, history)
    """
    from src.models.neural_garch import NeuralGARCHWrapper
    
    # Create and train model
    model = NeuralGARCHWrapper(
        p=p,
        q=q,
        hidden_layers=hidden_layers,
        device=device,
    )
    
    history = model.fit(
        returns=returns,
        epochs=epochs,
        lr=lr,
        verbose=verbose,
    )
    
    return model, history

