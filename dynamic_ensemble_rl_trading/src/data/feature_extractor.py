"""
Technical feature extraction module.

This module calculates 15 technical indicators and normalizes them
using rolling windows to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

try:
    import pandas_ta as ta
except ImportError:
    try:
        import talib
        TALIB_AVAILABLE = True
        PANDAS_TA_AVAILABLE = False
    except ImportError:
        TALIB_AVAILABLE = False
        PANDAS_TA_AVAILABLE = False
        logging.warning(
            "Neither pandas_ta nor talib available. "
            "Technical indicators will be calculated manually."
        )
else:
    PANDAS_TA_AVAILABLE = True
    TALIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class TechnicalFeatureExtractor:
    """
    Extracts technical indicators from OHLCV data.
    
    Calculates 15 key technical indicators including moving averages,
    momentum indicators, volatility measures, and trend indicators.
    All features are normalized using rolling windows to avoid look-ahead bias.
    """
    
    def __init__(
        self,
        normalization_window: int = 30,
        ohlcv_columns: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the TechnicalFeatureExtractor.
        
        Parameters
        ----------
        normalization_window : int, default=30
            Window size for rolling normalization to prevent look-ahead bias.
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names. If None, uses default names.
        """
        self.normalization_window = normalization_window
        
        if ohlcv_columns is None:
            self.ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        else:
            self.ohlcv_columns = ohlcv_columns
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all technical features from OHLCV data.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with datetime index.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with technical features as columns.
        """
        logger.info("Extracting technical features")
        
        features_df = pd.DataFrame(index=df.index)
        
        # Calculate all indicators
        features_df = self._calculate_sma_features(df, features_df)
        features_df = self._calculate_ema_features(df, features_df)
        features_df = self._calculate_rsi(df, features_df)
        features_df = self._calculate_macd(df, features_df)
        features_df = self._calculate_atr(df, features_df)
        features_df = self._calculate_bollinger_bands(df, features_df)
        features_df = self._calculate_additional_indicators(df, features_df)
        
        # Normalize features using rolling window
        features_df = self._normalize_features(features_df)
        
        logger.info(f"Extracted {len(features_df.columns)} technical features")
        
        return features_df
    
    def _calculate_sma_features(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Simple Moving Average features."""
        close = df[self.ohlcv_columns['close']]
        
        # SMA with different periods
        features_df['sma_10'] = close.rolling(window=10).mean()
        features_df['sma_20'] = close.rolling(window=20).mean()
        features_df['sma_50'] = close.rolling(window=50).mean()
        
        return features_df
    
    def _calculate_ema_features(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Exponential Moving Average features."""
        close = df[self.ohlcv_columns['close']]
        
        # EMA with different periods
        features_df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        features_df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        
        return features_df
    
    def _calculate_rsi(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        close = df[self.ohlcv_columns['close']]
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / (loss + 1e-10)
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        return features_df
    
    def _calculate_macd(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        close = df[self.ohlcv_columns['close']]
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        features_df['macd'] = ema_12 - ema_26
        features_df['macd_signal'] = features_df['macd'].ewm(
            span=9, adjust=False
        ).mean()
        features_df['macd_histogram'] = (
            features_df['macd'] - features_df['macd_signal']
        )
        
        return features_df
    
    def _calculate_atr(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Average True Range (volatility measure)."""
        high = df[self.ohlcv_columns['high']]
        low = df[self.ohlcv_columns['low']]
        close = df[self.ohlcv_columns['close']]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features_df['atr'] = tr.rolling(window=14).mean()
        
        return features_df
    
    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        close = df[self.ohlcv_columns['close']]
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        
        features_df['bb_upper'] = sma_20 + (std_20 * 2)
        features_df['bb_middle'] = sma_20
        features_df['bb_lower'] = sma_20 - (std_20 * 2)
        features_df['bb_width'] = (
            (features_df['bb_upper'] - features_df['bb_lower']) / sma_20
        )
        features_df['bb_position'] = (
            (close - features_df['bb_lower']) /
            (features_df['bb_upper'] - features_df['bb_lower'] + 1e-10)
        )
        
        return features_df
    
    def _calculate_additional_indicators(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate additional technical indicators."""
        close = df[self.ohlcv_columns['close']]
        volume = df[self.ohlcv_columns['volume']]
        high = df[self.ohlcv_columns['high']]
        low = df[self.ohlcv_columns['low']]
        
        # Price change
        features_df['price_change'] = close.pct_change()
        
        # Volume moving average
        features_df['volume_sma'] = volume.rolling(window=20).mean()
        features_df['volume_ratio'] = volume / (features_df['volume_sma'] + 1e-10)
        
        # High-Low range
        features_df['hl_range'] = (high - low) / close
        
        return features_df
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using rolling window to prevent look-ahead bias.
        
        Uses rolling mean and std for normalization, ensuring that at each
        time step, only past data is used for normalization.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with raw feature values.
        
        Returns
        -------
        pd.DataFrame
            Normalized features.
        """
        normalized_df = pd.DataFrame(index=features_df.index)
        
        for col in features_df.columns:
            rolling_mean = features_df[col].rolling(
                window=self.normalization_window,
                min_periods=1
            ).mean()
            rolling_std = features_df[col].rolling(
                window=self.normalization_window,
                min_periods=1
            ).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1e-10)
            
            # Z-score normalization
            normalized_df[f'{col}_norm'] = (
                (features_df[col] - rolling_mean) / rolling_std
            )
        
        # Replace infinite values with NaN, then forward fill
        normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan)
        normalized_df = normalized_df.ffill().fillna(0)
        
        return normalized_df
    
    def get_feature_names(self, features_df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Get list of all feature names.
        
        Parameters
        ----------
        features_df : pd.DataFrame, optional
            If provided, returns actual column names from the DataFrame.
            Otherwise, returns expected feature names.
        
        Returns
        -------
        list of str
            List of normalized feature names.
        """
        if features_df is not None:
            return list(features_df.columns)
        
        # Return expected feature names if DataFrame not provided
        base_features = [
            'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'rsi',
            'macd', 'macd_signal', 'macd_histogram',
            'atr',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'price_change', 'volume_sma', 'volume_ratio', 'hl_range'
        ]
        return [f'{feat}_norm' for feat in base_features]

