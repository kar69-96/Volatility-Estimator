"""
Fed Rate Event Prediction Model.

Uses LSTM or simple transformer architecture to predict Fed rate changes
based on market conditions and historical patterns.
"""

from typing import List, Optional, Tuple

import numpy as np

# Lazy imports
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_nn():
    torch = _get_torch()
    return torch.nn


def create_fed_rate_lstm(
    n_features: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    n_classes: int = 3,  # Increase, Decrease, No Change
    predict_magnitude: bool = True,
):
    """
    Create LSTM model for Fed rate prediction.
    
    Multi-task learning: Classification (direction) + Regression (magnitude)
    
    Args:
        n_features: Number of input features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        n_classes: Number of classification classes
        predict_magnitude: Whether to also predict magnitude
        
    Returns:
        PyTorch model
    """
    torch = _get_torch()
    nn = _get_nn()
    
    class FedRateLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.predict_magnitude = predict_magnitude
            
            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
            
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            
            # Attention layer for sequence aggregation
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_dim // 2, 1),
            )
            
            # Classification head (direction: increase/decrease/no change)
            self.classifier = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
            
            # Regression head (magnitude in basis points)
            if predict_magnitude:
                self.regressor = nn.Sequential(
                    nn.Linear(lstm_output_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
            
        def forward(self, x, return_attention: bool = False):
            """
            Args:
                x: Input tensor of shape (batch, seq_len, n_features)
                return_attention: Whether to return attention weights
                
            Returns:
                Tuple of (class_logits, magnitude) or just class_logits
            """
            # LSTM forward pass
            lstm_out, _ = self.lstm(x)
            # lstm_out shape: (batch, seq_len, hidden_dim * (2 if bidirectional else 1))
            
            # Attention-weighted aggregation
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
            
            # Classification
            class_logits = self.classifier(context)
            
            outputs = {'class_logits': class_logits}
            
            # Regression
            if self.predict_magnitude:
                magnitude = self.regressor(context)
                outputs['magnitude'] = magnitude.squeeze(-1)
            
            if return_attention:
                outputs['attention'] = attn_weights.squeeze(-1)
            
            return outputs
        
        def predict_class(self, x):
            """Get predicted class (0: decrease, 1: no change, 2: increase)."""
            torch = _get_torch()
            outputs = self.forward(x)
            return torch.argmax(outputs['class_logits'], dim=1)
        
        def predict_proba(self, x):
            """Get class probabilities."""
            torch = _get_torch()
            outputs = self.forward(x)
            return torch.softmax(outputs['class_logits'], dim=1)
    
    return FedRateLSTM()


def create_fed_rate_transformer(
    n_features: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    seq_length: int = 60,
    n_classes: int = 3,
    predict_magnitude: bool = True,
):
    """
    Create transformer model for Fed rate prediction.
    
    Args:
        n_features: Number of input features
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        seq_length: Input sequence length
        n_classes: Number of classification classes
        predict_magnitude: Whether to predict magnitude
        
    Returns:
        PyTorch model
    """
    torch = _get_torch()
    nn = _get_nn()
    
    class FedRateTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.predict_magnitude = predict_magnitude
            
            # Input projection
            self.input_proj = nn.Linear(n_features, d_model)
            
            # Positional encoding
            self.pos_encoding = nn.Parameter(
                torch.randn(1, seq_length, d_model) * 0.02
            )
            
            # CLS token for classification
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            
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
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
            
            # Regression head
            if predict_magnitude:
                self.regressor = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )
        
        def forward(self, x):
            """
            Args:
                x: Input tensor of shape (batch, seq_len, n_features)
                
            Returns:
                Dictionary with 'class_logits' and optionally 'magnitude'
            """
            batch_size = x.shape[0]
            
            # Project input
            x = self.input_proj(x)
            
            # Add positional encoding
            x = x + self.pos_encoding[:, :x.size(1), :]
            
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Transformer forward
            x = self.transformer(x)
            
            # Use CLS token for prediction
            cls_output = x[:, 0]
            
            outputs = {
                'class_logits': self.classifier(cls_output)
            }
            
            if self.predict_magnitude:
                outputs['magnitude'] = self.regressor(cls_output).squeeze(-1)
            
            return outputs
    
    return FedRateTransformer()


class FedRatePredictorWrapper:
    """
    High-level wrapper for Fed rate prediction model.
    """
    
    def __init__(
        self,
        n_features: int,
        model_type: str = 'lstm',
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
        predict_magnitude: bool = True,
        device: str = 'auto',
    ):
        """
        Initialize Fed rate predictor.
        
        Args:
            n_features: Number of input features
            model_type: 'lstm' or 'transformer'
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout rate
            n_classes: Number of classes (3 for increase/decrease/no change)
            predict_magnitude: Whether to predict magnitude
            device: Device preference
        """
        from src.models.base_model import get_device
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.predict_magnitude = predict_magnitude
        self.device = get_device(device)
        
        # Class labels
        self.class_labels = ['Decrease', 'No Change', 'Increase']
        
        # Create model
        if model_type == 'lstm':
            self.model = create_fed_rate_lstm(
                n_features=n_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                n_classes=n_classes,
                predict_magnitude=predict_magnitude,
            )
        else:
            self.model = create_fed_rate_transformer(
                n_features=n_features,
                d_model=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                n_classes=n_classes,
                predict_magnitude=predict_magnitude,
            )
        
        self.model = self.model.to(self.device)
    
    def predict(self, X: np.ndarray) -> dict:
        """
        Make predictions.
        
        Args:
            X: Input array of shape (n_samples, seq_len, n_features)
            
        Returns:
            Dictionary with predictions
        """
        torch = _get_torch()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            outputs = self.model(X_tensor)
            
            probs = torch.softmax(outputs['class_logits'], dim=1)
            predicted_class = torch.argmax(probs, dim=1)
            
            result = {
                'predicted_class': predicted_class.cpu().numpy(),
                'class_probabilities': probs.cpu().numpy(),
                'class_labels': [self.class_labels[c] for c in predicted_class.cpu().numpy()],
            }
            
            if 'magnitude' in outputs:
                result['predicted_magnitude_bps'] = outputs['magnitude'].cpu().numpy()
            
            return result
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        return self.predict(X)['class_probabilities']
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch = _get_torch()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'predict_magnitude': self.predict_magnitude,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        torch = _get_torch()
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def load_fed_rate_data(
    start_date: str = '2010-01-01',
    end_date: str = None,
    cache_dir: str = './data/cache',
):
    """
    Load and preprocess Federal Reserve rate data.
    
    Uses yfinance to fetch Treasury Bill rates as proxy for Fed rates.
    
    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format, defaults to today)
        cache_dir: Directory for caching data
        
    Returns:
        DataFrame with columns: date, rate, rate_change, rate_change_pct, direction
        Returns None if data cannot be loaded
    """
    import pandas as pd
    import yfinance as yf
    from datetime import datetime
    from pathlib import Path
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    cache_path = Path(cache_dir) / 'fed_rates.parquet'
    
    # Try to load from cache
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            # Ensure date column is in the right format
            if 'date' in df.columns:
                # Convert to date objects for comparison
                df['date'] = pd.to_datetime(df['date']).dt.date
                start_dt = pd.to_datetime(start_date).date()
                end_dt = pd.to_datetime(end_date).date()
                df_filtered = df[
                    (df['date'] >= start_dt) & (df['date'] <= end_dt)
                ]
                if len(df_filtered) > 0:
                    return df_filtered
        except Exception as e:
            # If cache load fails, continue to fetch from API
            import warnings
            warnings.warn(f"Cache load failed: {e}, fetching from API")
            pass
    
    # Fetch 13-week Treasury Bill rate as proxy
    try:
        ticker = yf.Ticker('^IRX')
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
        
        df = pd.DataFrame({
            'date': data.index.date,
            'rate': data['Close'].values,
        })
        
        # Calculate rate changes
        df['rate_change'] = df['rate'].diff()
        df['rate_change_pct'] = df['rate'].pct_change() * 100
        
        # Classify changes
        df['direction'] = 0  # No change
        df.loc[df['rate_change'] > 0.05, 'direction'] = 2  # Increase
        df.loc[df['rate_change'] < -0.05, 'direction'] = 0  # Decrease
        df.loc[
            (df['rate_change'] >= -0.05) & (df['rate_change'] <= 0.05),
            'direction'
        ] = 1  # No change
        
        # Save to cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        except:
            pass
        
        return df
        
    except Exception as e:
        print(f"Warning: Could not load Fed rate data: {e}")
        return None

