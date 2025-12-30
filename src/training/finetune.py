"""
Training loop for Chronos volatility prediction.

CRITICAL FIXES:
1. Train all quantiles with quantile (pinball) loss
2. Evaluate QLIKE only on q50 (median)
3. Quantiles and QLIKE are conceptually separate
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def quantile_loss(pred_quantiles, target, quantiles=[0.1, 0.5, 0.9]):
    """
    Quantile (pinball) loss for all quantiles.
    
    CRITICAL: Train all quantiles (q10, q50, q90) jointly.
    QLIKE is only used for evaluation on q50.
    
    Args:
        pred_quantiles: Predicted quantiles, shape (batch, 3) where columns are [q10, q50, q90]
        target: True values, shape (batch,)
        quantiles: List of quantile levels [q10, q50, q90]
        
    Returns:
        Average quantile loss across all quantiles
    """
    target = target.unsqueeze(1)  # (batch, 1)
    errors = target - pred_quantiles  # (batch, 3)
    
    loss = 0
    for i, q in enumerate(quantiles):
        loss += torch.mean(torch.max(q * errors[:, i], (q - 1) * errors[:, i]))
    
    return loss / len(quantiles)


def qlike_loss(pred_log_variance, target_log_realized_variance):
    """
    QLIKE loss: log(σ²_pred/σ²_true) + σ²_true/σ²_pred - 1.
    
    CRITICAL: Only use for evaluation on q50, not training.
    QLIKE is for conditional mean forecasts, not quantiles.
    
    Args:
        pred_log_variance: Predicted log-variance (from q50)
        target_log_realized_variance: True log-realized variance
        
    Returns:
        QLIKE loss value
    """
    pred_var = torch.exp(pred_log_variance)
    target_var = torch.exp(target_log_realized_variance)
    term1 = torch.log(pred_var / target_var + 1e-8)
    term2 = target_var / (pred_var + 1e-8)
    return (term1 + term2 - 1.0).mean()


def train(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda'):
    """
    Training loop with quantile loss.
    
    CRITICAL: Train all quantiles with quantile loss.
    Evaluate QLIKE on q50 only for monitoring.
    
    Args:
        model: ChronosVolatility model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device for training
        
    Returns:
        Trained model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_qlike = 0
        
        for input_seq, y in train_loader:
            input_seq = input_seq.to(device)
            y = y.to(device).squeeze()
            
            optimizer.zero_grad()
            
            # Forward: get all quantiles
            # Input_seq should be (batch, seq_len) from the dataset
            quantiles = model(input_seq)  # (batch, 3): [q10, q50, q90]
            
            # Train with quantile loss (all quantiles)
            loss = quantile_loss(quantiles, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Monitor QLIKE on q50 only (not used for training)
            q50 = quantiles[:, 1]
            qlike_val = qlike_loss(q50, y)
            train_qlike += qlike_val.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_qlike = 0
        
        with torch.no_grad():
            for input_seq, y in val_loader:
                input_seq = input_seq.to(device)
                y = y.to(device).squeeze()
                
                quantiles = model(input_seq)
                
                # Quantile loss for monitoring
                loss = quantile_loss(quantiles, y)
                val_loss += loss.item()
                
                # QLIKE on q50 for evaluation
                q50 = quantiles[:, 1]
                qlike_val = qlike_loss(q50, y)
                val_qlike += qlike_val.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val QLIKE: {val_qlike/len(val_loader):.4f}")
    
    return model

