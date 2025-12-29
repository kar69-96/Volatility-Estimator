"""
Neural GARCH Model.

Implements a neural network-based GARCH model for conditional variance estimation.
The model learns the GARCH dynamics using a small MLP, allowing for more flexible
nonlinear relationships than traditional GARCH.

Architecture:
    σ²_t = f(r_{t-1}, r_{t-2}, ..., σ²_{t-1}, σ²_{t-2}, ...)
    where f is a neural network
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


def create_neural_garch(
    p: int = 1,
    q: int = 1,
    hidden_layers: List[int] = None,
    activation: str = 'relu',
    dropout: float = 0.1,
    use_skip_connection: bool = True,
):
    """
    Create Neural GARCH model.
    
    The model takes as input:
    - Past squared returns (ε²_{t-1}, ..., ε²_{t-p})
    - Past conditional variances (σ²_{t-1}, ..., σ²_{t-q})
    
    And outputs:
    - Current conditional variance (σ²_t)
    
    Args:
        p: Number of lagged squared return terms (like ARCH order)
        q: Number of lagged variance terms (like GARCH order)
        hidden_layers: List of hidden layer sizes (default: [32, 16])
        activation: Activation function ('relu', 'gelu', 'tanh')
        dropout: Dropout rate
        use_skip_connection: Add skip connection from input to output
        
    Returns:
        PyTorch model
    """
    torch = _get_torch()
    nn = _get_nn()
    
    if hidden_layers is None:
        hidden_layers = [32, 16]
    
    input_dim = p + q  # p squared returns + q past variances
    
    # Select activation function
    if activation == 'relu':
        act_fn = nn.ReLU
    elif activation == 'gelu':
        act_fn = nn.GELU
    elif activation == 'tanh':
        act_fn = nn.Tanh
    else:
        act_fn = nn.ReLU
    
    class NeuralGARCH(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.p = p
            self.q = q
            self.use_skip_connection = use_skip_connection
            
            # Build MLP layers
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_layers:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
            
            self.mlp = nn.Sequential(*layers)
            
            # Output layer
            self.output = nn.Linear(prev_dim, 1)
            
            # Skip connection (linear GARCH component)
            if use_skip_connection:
                self.skip = nn.Linear(input_dim, 1, bias=False)
            
            # Learnable lower bound for variance (ensure positivity)
            self.log_floor = nn.Parameter(torch.tensor(-10.0))
            
        def forward(self, x):
            """
            Args:
                x: Input tensor of shape (batch, p + q)
                   First p values are squared returns
                   Last q values are past variances
                   
            Returns:
                Conditional variance of shape (batch, 1)
            """
            # MLP output
            h = self.mlp(x)
            output = self.output(h)
            
            # Add skip connection (allows model to learn linear GARCH)
            if self.use_skip_connection:
                output = output + self.skip(x)
            
            # Ensure positive variance with softplus
            floor = torch.exp(self.log_floor)
            variance = torch.nn.functional.softplus(output) + floor
            
            return variance
        
        def predict_sequence(self, returns: 'torch.Tensor') -> 'torch.Tensor':
            """
            Predict conditional variance for a sequence of returns.
            
            Args:
                returns: Tensor of returns of shape (seq_len,) or (batch, seq_len)
                
            Returns:
                Conditional variances of shape (seq_len - max(p, q),) or (batch, ...)
            """
            torch = _get_torch()
            
            if returns.dim() == 1:
                returns = returns.unsqueeze(0)
            
            batch_size, seq_len = returns.shape
            device = returns.device
            
            # Initialize variances with sample variance
            init_var = returns[:, :max(self.p, self.q)].var(dim=1, keepdim=True)
            
            # Squared returns
            squared_returns = returns ** 2
            
            variances = []
            
            # Need at least max(p, q) observations to start
            start_idx = max(self.p, self.q)
            
            # Initialize past variances
            past_variances = init_var.expand(batch_size, self.q)
            
            for t in range(start_idx, seq_len):
                # Get past squared returns
                past_squared = squared_returns[:, t - self.p:t]
                
                # Combine inputs
                inputs = torch.cat([past_squared, past_variances], dim=1)
                
                # Predict variance
                var_t = self(inputs)
                variances.append(var_t)
                
                # Update past variances
                if self.q > 1:
                    past_variances = torch.cat([
                        past_variances[:, 1:],
                        var_t
                    ], dim=1)
                else:
                    past_variances = var_t
            
            if len(variances) == 0:
                return torch.zeros(batch_size, 0, device=device)
            
            return torch.cat(variances, dim=1).squeeze(0)
    
    return NeuralGARCH()


def create_neural_egarch(
    p: int = 1,
    q: int = 1,
    hidden_layers: List[int] = None,
    dropout: float = 0.1,
):
    """
    Create Neural EGARCH (Exponential GARCH) model.
    
    Models log-variance instead of variance for:
    - Natural positivity constraint
    - Asymmetric effects (leverage effect)
    
    Args:
        p: Number of lagged innovation terms
        q: Number of lagged log-variance terms
        hidden_layers: Hidden layer sizes
        dropout: Dropout rate
        
    Returns:
        PyTorch model
    """
    torch = _get_torch()
    nn = _get_nn()
    
    if hidden_layers is None:
        hidden_layers = [32, 16]
    
    # Input: standardized innovations, |innovations|, past log-variances
    input_dim = p + p + q  # innovations + |innovations| + log-variances
    
    class NeuralEGARCH(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.p = p
            self.q = q
            
            # Build MLP
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_layers:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
            
            self.mlp = nn.Sequential(*layers)
            self.output = nn.Linear(prev_dim, 1)
            
            # Intercept term
            self.omega = nn.Parameter(torch.tensor(0.0))
            
        def forward(self, x):
            """
            Args:
                x: Input tensor (standardized innovations, |innovations|, log-variances)
                
            Returns:
                Log-variance
            """
            h = self.mlp(x)
            log_var = self.omega + self.output(h)
            return log_var
        
        def get_variance(self, x):
            """Get variance (exponential of log-variance)."""
            log_var = self.forward(x)
            return torch.exp(log_var)
    
    return NeuralEGARCH()


class NeuralGARCHWrapper:
    """
    High-level wrapper for Neural GARCH model.
    
    Provides convenient interface for training and prediction.
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        hidden_layers: List[int] = None,
        model_type: str = 'garch',  # 'garch' or 'egarch'
        device: str = 'auto',
    ):
        """
        Initialize Neural GARCH model.
        
        Args:
            p: ARCH order (lagged squared returns)
            q: GARCH order (lagged variances)
            hidden_layers: Hidden layer sizes
            model_type: 'garch' or 'egarch'
            device: Device preference
        """
        from src.models.base_model import get_device
        
        self.p = p
        self.q = q
        self.model_type = model_type
        self.device = get_device(device)
        
        if model_type == 'egarch':
            self.model = create_neural_egarch(
                p=p, q=q, hidden_layers=hidden_layers
            )
        else:
            self.model = create_neural_garch(
                p=p, q=q, hidden_layers=hidden_layers
            )
        
        self.model = self.model.to(self.device)
        self.fitted = False
    
    def fit(
        self,
        returns: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict:
        """
        Fit the model using maximum likelihood estimation.
        
        Args:
            returns: Array of returns
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            verbose: Print training progress
            
        Returns:
            Dictionary with training history
        """
        torch = _get_torch()
        
        returns_tensor = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        history = {'loss': []}
        
        self.model.train()
        
        for epoch in range(epochs):
            # Predict variances for the sequence
            variances = self.model.predict_sequence(returns_tensor)
            
            # Get corresponding squared returns for the target
            start_idx = max(self.p, self.q)
            actual_squared = returns_tensor[start_idx:] ** 2
            
            if len(variances) != len(actual_squared):
                min_len = min(len(variances), len(actual_squared))
                variances = variances[:min_len]
                actual_squared = actual_squared[:min_len]
            
            # Gaussian log-likelihood loss
            # L = -0.5 * sum(log(σ²) + ε²/σ²)
            loss = 0.5 * torch.mean(
                torch.log(variances + 1e-8) + actual_squared / (variances + 1e-8)
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            history['loss'].append(loss.item())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
        
        self.fitted = True
        return history
    
    def predict(self, returns: np.ndarray) -> np.ndarray:
        """
        Predict conditional variances.
        
        Args:
            returns: Array of returns
            
        Returns:
            Array of conditional variances
        """
        torch = _get_torch()
        
        self.model.eval()
        with torch.no_grad():
            returns_tensor = torch.tensor(
                returns, dtype=torch.float32, device=self.device
            )
            variances = self.model.predict_sequence(returns_tensor)
            return variances.cpu().numpy()
    
    def predict_volatility(self, returns: np.ndarray, annualize: bool = True) -> np.ndarray:
        """
        Predict conditional volatility (standard deviation).
        
        Args:
            returns: Array of returns
            annualize: Whether to annualize (multiply by sqrt(252))
            
        Returns:
            Array of conditional volatilities
        """
        variances = self.predict(returns)
        volatility = np.sqrt(variances)
        
        if annualize:
            volatility = volatility * np.sqrt(252) * 100  # Annualized percentage
        
        return volatility
    
    def forecast(self, returns: np.ndarray, horizon: int = 1, annualize: bool = True) -> np.ndarray:
        """
        Forecast future volatility for N days ahead.
        
        Uses the model's dynamics to project volatility forward by:
        1. Using the last known returns and volatilities
        2. Iteratively predicting future variance using the model
        3. Assuming future returns follow the historical mean (or zero)
        
        Args:
            returns: Historical returns array
            horizon: Number of days ahead to forecast
            annualize: Whether to annualize (multiply by sqrt(252))
            
        Returns:
            Array of forecasted volatilities (length = horizon)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        torch = _get_torch()
        
        self.model.eval()
        with torch.no_grad():
            # Get historical returns as tensor
            returns_tensor = torch.tensor(
                returns, dtype=torch.float32, device=self.device
            )
            
            # Get the last p+q returns and variances for initialization
            max_lag = max(self.p, self.q)
            if len(returns) < max_lag:
                raise ValueError(f"Need at least {max_lag} historical returns for forecasting")
            
            # Get historical variances
            hist_variances = self.predict(returns)
            if len(hist_variances) == 0:
                # Fallback: use sample variance
                hist_var = returns[-max_lag:].var()
            else:
                hist_var = hist_variances[-1]
            
            # Get last returns for ARCH component
            last_returns = returns[-self.p:] if len(returns) >= self.p else returns
            last_squared_returns = last_returns ** 2
            
            # Get last variances for GARCH component
            if len(hist_variances) >= self.q:
                last_variances = hist_variances[-self.q:]
            else:
                # Pad with last variance
                last_variances = np.full(self.q, hist_var)
            
            # Convert to tensors
            last_squared_tensor = torch.tensor(
                last_squared_returns, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            last_var_tensor = torch.tensor(
                last_variances, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            
            # Mean return for future periods (could be zero or historical mean)
            mean_return = float(returns.mean()) if len(returns) > 0 else 0.0
            
            # Forecast variances forward
            forecast_variances = []
            current_squared = last_squared_tensor.clone()  # Shape: (1, p)
            current_var = last_var_tensor.clone()  # Shape: (1, q)
            
            for _ in range(horizon):
                # Prepare input: [past squared returns, past variances]
                # Ensure both tensors are 2D: (batch, features)
                inputs = torch.cat([current_squared, current_var], dim=1)
                
                # Predict next variance
                next_var = self.model(inputs)  # Shape: (1, 1)
                var_value = next_var.item() if next_var.numel() == 1 else next_var.squeeze().item()
                # Ensure variance is positive and not NaN
                if np.isnan(var_value) or var_value <= 0:
                    # Use last variance as fallback
                    var_value = float(hist_var) if not np.isnan(hist_var) else 0.0001
                forecast_variances.append(var_value)
                
                # Update for next iteration:
                # Shift squared returns (add mean return squared for future)
                future_squared = torch.tensor(
                    [[mean_return ** 2]], dtype=torch.float32, device=self.device
                )  # Shape: (1, 1)
                
                if self.p > 1:
                    # Shift: remove first column, add new one at end
                    current_squared = torch.cat([current_squared[:, 1:], future_squared], dim=1)
                else:
                    current_squared = future_squared  # Shape: (1, 1)
                
                # Shift variances - ensure next_var is properly shaped
                next_var_reshaped = next_var if next_var.dim() == 2 else next_var.unsqueeze(0)
                if next_var_reshaped.shape[1] != 1:
                    next_var_reshaped = next_var_reshaped.unsqueeze(1) if next_var_reshaped.dim() == 1 else next_var_reshaped
                
                if self.q > 1:
                    # Shift: remove first column, add new one at end
                    current_var = torch.cat([current_var[:, 1:], next_var_reshaped], dim=1)
                else:
                    current_var = next_var_reshaped  # Shape: (1, 1)
            
            # Convert to numpy and compute volatility
            forecast_var = np.array(forecast_variances)
            forecast_vol = np.sqrt(forecast_var)
            
            if annualize:
                forecast_vol = forecast_vol * np.sqrt(252) * 100  # Annualized percentage
            
            return forecast_vol
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch = _get_torch()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'p': self.p,
            'q': self.q,
            'model_type': self.model_type,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        torch = _get_torch()
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fitted = True


def prepare_garch_data(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for Neural GARCH training.
    
    Creates input-output pairs for supervised learning.
    
    Args:
        returns: Array of returns
        p: Number of lagged squared returns
        q: Number of lagged variances
        
    Returns:
        Tuple of (X, y) arrays
    """
    n = len(returns)
    squared_returns = returns ** 2
    
    # Initialize variances with rolling sample variance
    window = max(p, q, 10)
    variances = np.zeros(n)
    for i in range(window, n):
        variances[i] = squared_returns[i - window:i].mean()
    
    X = []
    y = []
    
    start_idx = max(p, q)
    
    for t in range(start_idx, n):
        # Past squared returns
        past_squared = squared_returns[t - p:t]
        
        # Past variances
        past_var = variances[t - q:t]
        
        # Input features
        features = np.concatenate([past_squared, past_var])
        
        # Target is current squared return (proxy for true variance)
        target = squared_returns[t]
        
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y)

