"""
Ground truth generation for market regime classification.

This module generates labels for market regimes (Bull, Bear, Sideways)
based on the normalized slope of SMA-50 as specified in the paper.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RegimeGroundTruth:
    """
    Generates ground truth labels for market regime classification.
    
    Uses the normalized slope of SMA-50 to classify market regimes:
    - Bull: m_t > delta_bull (0.05%)
    - Bear: m_t < delta_bear (-0.05%)
    - Sideways: otherwise
    
    Labels: 0 (Bear), 1 (Sideways), 2 (Bull)
    """
    
    def __init__(
        self,
        sma_window: int = 50,
        bull_threshold: float = 0.0005,  # 0.05%
        bear_threshold: float = -0.0005   # -0.05%
    ):
        """
        Initialize the RegimeGroundTruth generator.
        
        Parameters
        ----------
        sma_window : int, default=50
            Window size for Simple Moving Average.
        bull_threshold : float, default=0.0005
            Threshold for Bull market (0.05%).
        bear_threshold : float, default=-0.0005
            Threshold for Bear market (-0.05%).
        """
        self.sma_window = sma_window
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
    
    def calculate_sma_slope(
        self,
        price_data: pd.Series
    ) -> pd.Series:
        """
        Calculate normalized slope of SMA.
        
        The normalized slope m_t is calculated as:
        m_t = (SMA_n(t) - SMA_n(t-1)) / SMA_n(t-1) * 100
        
        Parameters
        ----------
        price_data : pd.Series
            Price series (typically close prices).
        
        Returns
        -------
        pd.Series
            Normalized slope values m_t.
        """
        # Calculate SMA
        sma = price_data.rolling(window=self.sma_window).mean()
        
        # Calculate slope: difference between consecutive SMA values
        sma_diff = sma.diff()
        
        # Normalize by previous SMA value
        sma_prev = sma.shift(1)
        
        # Avoid division by zero
        sma_prev = sma_prev.replace(0, np.nan)
        
        # Normalized slope: m_t = (SMA(t) - SMA(t-1)) / SMA(t-1) * 100
        normalized_slope = (sma_diff / sma_prev) * 100
        
        return normalized_slope
    
    def generate_labels(
        self,
        price_data: pd.Series
    ) -> pd.Series:
        """
        Generate regime labels based on SMA slope.
        
        Parameters
        ----------
        price_data : pd.Series
            Price series (typically close prices) with datetime index.
        
        Returns
        -------
        pd.Series
            Regime labels: 0 (Bear), 1 (Sideways), 2 (Bull).
        """
        logger.info("Generating regime labels from price data")
        
        # Calculate normalized slope
        slope = self.calculate_sma_slope(price_data)
        
        # Initialize labels with Sideways (1)
        labels = pd.Series(1, index=price_data.index, dtype=int)
        
        # Bull market: m_t > delta_bull
        bull_mask = slope > self.bull_threshold
        labels[bull_mask] = 2
        
        # Bear market: m_t < delta_bear
        bear_mask = slope < self.bear_threshold
        labels[bear_mask] = 0
        
        # Handle NaN values (insufficient data for SMA calculation)
        labels = labels.fillna(1)  # Default to Sideways
        
        # Log label distribution
        label_counts = labels.value_counts().sort_index()
        logger.info(f"Regime label distribution: {dict(label_counts)}")
        logger.info(
            f"Bear (0): {label_counts.get(0, 0)}, "
            f"Sideways (1): {label_counts.get(1, 0)}, "
            f"Bull (2): {label_counts.get(2, 0)}"
        )
        
        return labels
    
    def get_regime_name(self, label: int) -> str:
        """
        Get regime name from label.
        
        Parameters
        ----------
        label : int
            Regime label (0, 1, or 2).
        
        Returns
        -------
        str
            Regime name: 'Bear', 'Sideways', or 'Bull'.
        """
        regime_map = {
            0: 'Bear',
            1: 'Sideways',
            2: 'Bull'
        }
        return regime_map.get(label, 'Unknown')

