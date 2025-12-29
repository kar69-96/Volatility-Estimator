"""
Feature Extractor Module.

Main interface for extracting and preprocessing features for deep learning models.
Combines technical indicators, volatility features, and market regime indicators.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.features.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_bollinger_width,
    calculate_atr,
    calculate_atr_percent,
    calculate_sma,
    calculate_ema,
    calculate_momentum,
    calculate_rate_of_change,
    calculate_stochastic_oscillator,
    calculate_williams_r,
    calculate_adx,
)
from src.features.return_features import calculate_log_returns
from src.features.volatility_features import (
    calculate_realized_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
)


class FeatureExtractor:
    """
    Feature extraction pipeline for volatility prediction models.
    
    Extracts and normalizes features from OHLC data including:
    - Price-based features (returns, momentum)
    - Volatility features (realized vol, range-based estimators)
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Market regime indicators
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        include_returns: bool = True,
        include_volatility: bool = True,
        include_technicals: bool = True,
        include_regime: bool = True,
        volatility_windows: List[int] = None,
        rsi_periods: List[int] = None,
    ):
        """
        Initialize feature extractor.
        
        Args:
            scaler_type: 'standard' (z-score) or 'minmax' (0-1)
            include_returns: Include return-based features
            include_volatility: Include volatility features
            include_technicals: Include technical indicators
            include_regime: Include regime indicators
            volatility_windows: Windows for volatility calculation
            rsi_periods: Periods for RSI calculation
        """
        self.scaler_type = scaler_type
        self.include_returns = include_returns
        self.include_volatility = include_volatility
        self.include_technicals = include_technicals
        self.include_regime = include_regime
        
        self.volatility_windows = volatility_windows or [5, 10, 20, 60]
        self.rsi_periods = rsi_periods or [14, 28]
        
        self.scaler = None
        self.feature_names: List[str] = []
        self.fitted = False
        
    def extract_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> pd.DataFrame:
        """
        Extract all features from OHLC data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=df.index)
        
        # Store date for reference
        if 'date' in df.columns:
            features['date'] = df['date']
        
        # Price data
        close = df['close']
        high = df['high']
        low = df['low']
        open_prices = df['open']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(0, index=df.index)
        
        # 1. Return-based features
        if self.include_returns:
            features = self._add_return_features(features, close)
        
        # 2. Volatility features
        if self.include_volatility:
            features = self._add_volatility_features(
                features, open_prices, high, low, close
            )
        
        # 3. Technical indicators
        if self.include_technicals:
            features = self._add_technical_features(
                features, open_prices, high, low, close, volume
            )
        
        # 4. Regime indicators
        if self.include_regime:
            features = self._add_regime_features(features, close)
        
        # Store feature names (excluding date)
        self.feature_names = [c for c in features.columns if c != 'date']
        
        # Scale features
        if fit_scaler or self.scaler is None:
            self._fit_scaler(features[self.feature_names])
            self.fitted = True
        
        features[self.feature_names] = self._transform_features(
            features[self.feature_names]
        )
        
        return features
    
    def _add_return_features(
        self,
        features: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """Add return-based features."""
        # Log returns
        log_ret = calculate_log_returns(close)
        features['log_return'] = log_ret
        
        # Squared returns (proxy for variance)
        features['squared_return'] = log_ret ** 2
        
        # Absolute returns
        features['abs_return'] = log_ret.abs()
        
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            features[f'log_return_lag_{lag}'] = log_ret.shift(lag)
        
        # Cumulative returns
        for window in [5, 10, 20]:
            features[f'cum_return_{window}d'] = log_ret.rolling(window).sum()
        
        return features
    
    def _add_volatility_features(
        self,
        features: pd.DataFrame,
        open_prices: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """Add volatility-based features."""
        log_ret = calculate_log_returns(close)
        
        for window in self.volatility_windows:
            # Realized volatility
            features[f'realized_vol_{window}d'] = calculate_realized_volatility(
                log_ret, window=window, annualize=False
            )
            
            # Parkinson volatility
            features[f'parkinson_vol_{window}d'] = calculate_parkinson_volatility(
                high, low, window=window, annualize=False
            )
            
            # Garman-Klass volatility
            features[f'gk_vol_{window}d'] = calculate_garman_klass_volatility(
                open_prices, high, low, close, window=window, annualize=False
            )
        
        # ATR percentage
        features['atr_pct_14'] = calculate_atr_percent(high, low, close, period=14)
        features['atr_pct_21'] = calculate_atr_percent(high, low, close, period=21)
        
        # Volatility ratios
        if 'realized_vol_5d' in features.columns and 'realized_vol_20d' in features.columns:
            features['vol_ratio_5_20'] = (
                features['realized_vol_5d'] / 
                features['realized_vol_20d'].replace(0, np.nan)
            )
        
        if 'realized_vol_10d' in features.columns and 'realized_vol_60d' in features.columns:
            features['vol_ratio_10_60'] = (
                features['realized_vol_10d'] / 
                features['realized_vol_60d'].replace(0, np.nan)
            )
        
        return features
    
    def _add_technical_features(
        self,
        features: pd.DataFrame,
        open_prices: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Add technical indicator features."""
        # RSI
        for period in self.rsi_periods:
            features[f'rsi_{period}'] = calculate_rsi(close, period=period)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_hist'] = histogram
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(close)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = calculate_bollinger_width(close)
        
        # Bollinger Band position (where price is within the bands)
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)
        
        # Stochastic
        stoch_k, stoch_d = calculate_stochastic_oscillator(high, low, close)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        # Williams %R
        features['williams_r'] = calculate_williams_r(high, low, close)
        
        # ADX
        features['adx'] = calculate_adx(high, low, close)
        
        # Moving average ratios
        sma_20 = calculate_sma(close, 20)
        sma_50 = calculate_sma(close, 50)
        sma_200 = calculate_sma(close, 200)
        
        features['price_sma20_ratio'] = close / sma_20
        features['price_sma50_ratio'] = close / sma_50
        features['sma20_sma50_ratio'] = sma_20 / sma_50
        
        # Momentum
        features['momentum_10'] = calculate_momentum(close, period=10)
        features['roc_10'] = calculate_rate_of_change(close, period=10)
        
        return features
    
    def _add_regime_features(
        self,
        features: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """Add market regime indicator features."""
        log_ret = calculate_log_returns(close)
        
        # Trend indicator (based on moving averages)
        sma_20 = calculate_sma(close, 20)
        sma_50 = calculate_sma(close, 50)
        
        features['trend_indicator'] = (sma_20 > sma_50).astype(float)
        
        # Volatility regime (high/medium/low based on percentiles)
        vol_20d = log_ret.rolling(20).std() * np.sqrt(252) * 100
        vol_percentile = vol_20d.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        features['vol_regime'] = vol_percentile
        
        # Momentum regime
        momentum = calculate_momentum(close, 20)
        mom_percentile = momentum.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        features['momentum_regime'] = mom_percentile
        
        # Day of week (cyclical encoding)
        if 'date' in features.columns:
            dates = pd.to_datetime(features['date'])
            day_of_week = dates.dt.dayofweek
            features['day_sin'] = np.sin(2 * np.pi * day_of_week / 5)
            features['day_cos'] = np.cos(2 * np.pi * day_of_week / 5)
        
        return features
    
    def _fit_scaler(self, features: pd.DataFrame) -> None:
        """Fit the scaler on feature data."""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        # Handle NaN values for fitting
        clean_features = features.fillna(features.median())
        self.scaler.fit(clean_features)
    
    def _transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if self.scaler is None:
            return features
        
        # Store original NaN locations
        nan_mask = features.isna()
        
        # Fill NaN for transformation
        filled_features = features.fillna(features.median())
        
        # Transform
        scaled = self.scaler.transform(filled_features)
        result = pd.DataFrame(scaled, index=features.index, columns=features.columns)
        
        # Restore NaN values
        result[nan_mask] = np.nan
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def get_n_features(self) -> int:
        """Get number of features."""
        return len(self.feature_names)


def extract_all_features(
    df: pd.DataFrame,
    scaler_type: str = 'standard',
    fit_scaler: bool = True
) -> Tuple[pd.DataFrame, FeatureExtractor]:
    """
    Convenience function to extract all features from OHLC data.
    
    Args:
        df: DataFrame with OHLC columns
        scaler_type: 'standard' or 'minmax'
        fit_scaler: Whether to fit the scaler
        
    Returns:
        Tuple of (features DataFrame, FeatureExtractor instance)
    """
    extractor = FeatureExtractor(scaler_type=scaler_type)
    features = extractor.extract_features(df, fit_scaler=fit_scaler)
    
    return features, extractor


def create_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    seq_length: int = 60,
    prediction_horizon: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series modeling.
    
    Args:
        features: DataFrame of feature values
        target: Series of target values (e.g., future volatility)
        seq_length: Length of input sequences
        prediction_horizon: How far ahead to predict
        
    Returns:
        Tuple of (X sequences, y targets)
    """
    # Remove date column if present
    feature_cols = [c for c in features.columns if c != 'date']
    feature_values = features[feature_cols].values
    target_values = target.values
    
    X, y = [], []
    
    for i in range(len(features) - seq_length - prediction_horizon + 1):
        # Input sequence
        seq = feature_values[i:i + seq_length]
        
        # Target (future value at prediction_horizon)
        target_val = target_values[i + seq_length + prediction_horizon - 1]
        
        # Skip if any NaN in sequence or target
        if not np.any(np.isnan(seq)) and not np.isnan(target_val):
            X.append(seq)
            y.append(target_val)
    
    return np.array(X), np.array(y)


def create_multi_horizon_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    seq_length: int = 60,
    horizons: List[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences with multiple prediction horizons.
    
    Args:
        features: DataFrame of feature values
        target: Series of target values
        seq_length: Length of input sequences
        horizons: List of prediction horizons (e.g., [1, 5, 10, 20])
        
    Returns:
        Tuple of (X sequences, y targets with shape [n_samples, n_horizons])
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]
    
    max_horizon = max(horizons)
    feature_cols = [c for c in features.columns if c != 'date']
    feature_values = features[feature_cols].values
    target_values = target.values
    
    X, y = [], []
    
    for i in range(len(features) - seq_length - max_horizon + 1):
        # Input sequence
        seq = feature_values[i:i + seq_length]
        
        # Multiple targets at different horizons
        targets = []
        valid = True
        for h in horizons:
            idx = i + seq_length + h - 1
            if idx < len(target_values):
                t = target_values[idx]
                if np.isnan(t):
                    valid = False
                    break
                targets.append(t)
            else:
                valid = False
                break
        
        # Skip if any NaN
        if valid and not np.any(np.isnan(seq)):
            X.append(seq)
            y.append(targets)
    
    return np.array(X), np.array(y)

