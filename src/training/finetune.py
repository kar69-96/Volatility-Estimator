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
    # Clip predictions to prevent extreme values
    pred_quantiles = torch.clamp(pred_quantiles, min=-20, max=20)
    
    target = target.unsqueeze(1)  # (batch, 1)
    errors = target - pred_quantiles  # (batch, 3)
    
    loss = 0
    for i, q in enumerate(quantiles):
        loss_q = torch.mean(torch.max(q * errors[:, i], (q - 1) * errors[:, i]))
        # Check for NaN
        if torch.isnan(loss_q):
            loss_q = torch.tensor(0.0, device=loss_q.device)
        loss += loss_q
    
    loss = loss / len(quantiles)
    # Final NaN check
    if torch.isnan(loss):
        return torch.tensor(0.0, device=loss.device)
    return loss


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
    # Clip log-variance values to prevent exp() overflow
    pred_log_variance = torch.clamp(pred_log_variance, min=-20, max=20)
    target_log_realized_variance = torch.clamp(target_log_realized_variance, min=-20, max=20)
    
    pred_var = torch.exp(pred_log_variance)
    target_var = torch.exp(target_log_realized_variance)
    
    # Ensure positive values with better epsilon
    epsilon = 1e-6
    term1 = torch.log(pred_var / (target_var + epsilon) + epsilon)
    term2 = target_var / (pred_var + epsilon)
    loss = (term1 + term2 - 1.0).mean()
    
    # Check for NaN/inf and return safe value
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=loss.device)
    return loss


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
        n_batches = 0
        
        for input_seq, y in train_loader:
            input_seq = input_seq.to(device)
            y = y.to(device).squeeze()
            
            # Check for NaN/inf in inputs
            if torch.isnan(input_seq).any() or torch.isinf(input_seq).any():
                print(f"Warning: NaN/inf in input_seq, skipping batch")
                continue
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"Warning: NaN/inf in target y, skipping batch")
                continue
            
            optimizer.zero_grad()
            
            # Forward: get all quantiles
            # Input_seq should be (batch, seq_len) from the dataset
            quantiles = model(input_seq)  # (batch, 3): [q10, q50, q90]
            
            # Check for NaN in model output
            if torch.isnan(quantiles).any() or torch.isinf(quantiles).any():
                print(f"Warning: NaN/inf in model output, skipping batch")
                continue
            
            # Train with quantile loss (all quantiles)
            loss = quantile_loss(quantiles, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/inf loss, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Monitor QLIKE on q50 only (not used for training)
            q50 = quantiles[:, 1]
            qlike_val = qlike_loss(q50, y)
            if not (torch.isnan(qlike_val) or torch.isinf(qlike_val)):
                train_qlike += qlike_val.item()
            
            n_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_qlike = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for input_seq, y in val_loader:
                input_seq = input_seq.to(device)
                y = y.to(device).squeeze()
                
                # Check for NaN/inf in inputs
                if torch.isnan(input_seq).any() or torch.isinf(input_seq).any():
                    continue
                if torch.isnan(y).any() or torch.isinf(y).any():
                    continue
                
                quantiles = model(input_seq)
                
                # Check for NaN in model output
                if torch.isnan(quantiles).any() or torch.isinf(quantiles).any():
                    continue
                
                # Quantile loss for monitoring
                loss = quantile_loss(quantiles, y)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                
                # QLIKE on q50 for evaluation
                q50 = quantiles[:, 1]
                qlike_val = qlike_loss(q50, y)
                if not (torch.isnan(qlike_val) or torch.isinf(qlike_val)):
                    val_qlike += qlike_val.item()
                
                n_val_batches += 1
        
        # Safe division
        avg_train_loss = train_loss / max(n_batches, 1)
        avg_val_loss = val_loss / max(n_val_batches, 1)
        avg_val_qlike = val_qlike / max(n_val_batches, 1)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val QLIKE: {avg_val_qlike:.4f}")
    
    return model

