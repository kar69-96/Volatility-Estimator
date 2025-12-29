"""
Volatility Prediction Model using iTransformer Architecture.

iTransformer (Inverted Transformer) treats each variable/feature as a token,
which is more effective for multivariate time series forecasting.

Reference: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
           arXiv:2310.06625
"""

import math
from typing import List, Optional, Tuple

import numpy as np

# Lazy imports for torch
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
    return _torch


def _get_nn():
    torch = _get_torch()
    return torch.nn


class PositionalEncoding:
    """Sinusoidal positional encoding for transformers."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        torch = _get_torch()
        nn = _get_nn()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.pe = pe
        
    def forward(self, x):
        """Add positional encoding to input."""
        torch = _get_torch()
        pe = self.pe[:, :x.size(1), :].to(x.device)
        x = x + pe
        return self.dropout(x)


class Time2Vec:
    """
    Time2Vec learnable time encoding.
    
    Converts time index to a vector representation with both periodic
    and linear components.
    """
    
    def __init__(self, input_dim: int, embed_dim: int):
        torch = _get_torch()
        nn = _get_nn()
        
        self.embed_dim = embed_dim
        self.W = nn.Linear(input_dim, embed_dim)
        self.P = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Time index tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Time encoding of shape (batch, seq_len, embed_dim * 2)
        """
        torch = _get_torch()
        linear = self.W(x)
        periodic = torch.sin(self.P(x))
        return torch.cat([linear, periodic], dim=-1)


class iTransformerBlock:
    """
    iTransformer block with inverted attention.
    
    In iTransformer, attention is applied across features (variables)
    instead of across time steps.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        torch = _get_torch()
        nn = _get_nn()
        
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, n_features, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, n_features, d_model)
        """
        # Self-attention across features
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


def create_volatility_predictor(
    n_features: int,
    seq_length: int = 60,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
    d_ff: int = 256,
    dropout: float = 0.1,
    prediction_horizons: List[int] = None,
):
    """
    Create an iTransformer model for volatility prediction.
    
    The model uses the inverted transformer architecture where:
    - Each feature/variable is treated as a token
    - Time dimension is embedded into the token representation
    - Attention is applied across features, not time steps
    
    Args:
        n_features: Number of input features
        seq_length: Length of input sequences
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        prediction_horizons: List of prediction horizons (e.g., [1, 5, 10, 20])
        
    Returns:
        PyTorch model
    """
    torch = _get_torch()
    nn = _get_nn()
    
    if prediction_horizons is None:
        prediction_horizons = [1, 5, 10, 20]
    
    n_outputs = len(prediction_horizons)
    
    class VolatilityPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Embed each feature's time series
            # Input: (batch, seq_len, n_features)
            # After embedding: (batch, n_features, d_model)
            self.feature_embed = nn.Linear(seq_length, d_model)
            
            # Positional encoding for features
            self.pos_encoding = nn.Parameter(
                torch.randn(1, n_features, d_model) * 0.02
            )
            
            # Transformer blocks
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                )
                for _ in range(num_layers)
            ])
            
            self.norm = nn.LayerNorm(d_model)
            
            # Output projection
            # Aggregate across features and predict for each horizon
            self.output_proj = nn.Sequential(
                nn.Linear(n_features * d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, n_outputs),
            )
            
            self.prediction_horizons = prediction_horizons
            
        def forward(self, x):
            """
            Args:
                x: Input tensor of shape (batch, seq_len, n_features)
                
            Returns:
                Predictions of shape (batch, n_outputs)
            """
            batch_size = x.shape[0]
            
            # Transpose to (batch, n_features, seq_len)
            x = x.transpose(1, 2)
            
            # Embed each feature's time series
            # (batch, n_features, d_model)
            x = self.feature_embed(x)
            
            # Add positional encoding
            x = x + self.pos_encoding
            
            # Apply transformer blocks
            for block in self.blocks:
                x = block(x)
            
            x = self.norm(x)
            
            # Flatten and project to outputs
            x = x.reshape(batch_size, -1)
            outputs = self.output_proj(x)
            
            return outputs
        
        def predict(self, x, horizon_idx: int = 0):
            """
            Get prediction for a specific horizon.
            
            Args:
                x: Input tensor
                horizon_idx: Index of the prediction horizon
                
            Returns:
                Prediction for the specified horizon
            """
            outputs = self.forward(x)
            return outputs[:, horizon_idx]
    
    return VolatilityPredictor()


def create_simple_transformer_predictor(
    n_features: int,
    seq_length: int = 60,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    n_outputs: int = 1,
):
    """
    Create a simpler transformer model for volatility prediction.
    
    Uses standard transformer architecture (attention across time).
    
    Args:
        n_features: Number of input features
        seq_length: Length of input sequences
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        n_outputs: Number of output predictions
        
    Returns:
        PyTorch model
    """
    torch = _get_torch()
    nn = _get_nn()
    
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Input projection
            self.input_proj = nn.Linear(n_features, d_model)
            
            # Positional encoding
            self.pos_encoding = nn.Parameter(
                torch.randn(1, seq_length, d_model) * 0.02
            )
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            
            # Output projection (use last token's representation)
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_outputs),
            )
            
        def forward(self, x):
            """
            Args:
                x: Input tensor of shape (batch, seq_len, n_features)
                
            Returns:
                Predictions of shape (batch, n_outputs)
            """
            # Project input
            x = self.input_proj(x)
            
            # Add positional encoding
            x = x + self.pos_encoding[:, :x.size(1), :]
            
            # Apply transformer
            x = self.transformer(x)
            
            # Use last token for prediction
            x = x[:, -1, :]
            
            return self.output_proj(x)
    
    return SimpleTransformer()


class VolatilityPredictorWrapper:
    """
    High-level wrapper for volatility prediction model.
    
    Handles model creation, training, and inference.
    """
    
    def __init__(
        self,
        n_features: int,
        seq_length: int = 60,
        model_type: str = 'itransformer',
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        prediction_horizons: List[int] = None,
        device: str = 'auto',
    ):
        """
        Initialize volatility predictor.
        
        Args:
            n_features: Number of input features
            seq_length: Input sequence length
            model_type: 'itransformer' or 'transformer'
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of layers
            dropout: Dropout rate
            prediction_horizons: Prediction horizons
            device: 'auto', 'cuda', 'mps', or 'cpu'
        """
        from src.models.base_model import get_device
        
        self.n_features = n_features
        self.seq_length = seq_length
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 20]
        self.device = get_device(device)
        
        # Create model
        if model_type == 'itransformer':
            self.model = create_volatility_predictor(
                n_features=n_features,
                seq_length=seq_length,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                prediction_horizons=self.prediction_horizons,
            )
        else:
            self.model = create_simple_transformer_predictor(
                n_features=n_features,
                seq_length=seq_length,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                n_outputs=len(self.prediction_horizons),
            )
        
        self.model = self.model.to(self.device)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input array of shape (n_samples, seq_length, n_features)
            
        Returns:
            Predictions of shape (n_samples, n_horizons)
        """
        torch = _get_torch()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch = _get_torch()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_features': self.n_features,
            'seq_length': self.seq_length,
            'prediction_horizons': self.prediction_horizons,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        torch = _get_torch()
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

