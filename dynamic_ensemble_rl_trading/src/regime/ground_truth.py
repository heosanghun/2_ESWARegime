"""
Ground truth generation for market regime classification.

This module generates labels for market regimes (Bull, Bear, Sideways).

Two methods are supported:
- 'sma'             : legacy SMA-50 normalized slope (lagging indicator).
- 'trend_scanning'  : forward-looking Trend Scanning (López de Prado, 2018)
                      — used to address Reviewer #3's "lagging label" concern.

The active method is controlled by the `method` constructor argument
(default 'sma' for backwards compatibility).
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RegimeGroundTruth:
    """
    Generates ground truth labels for market regime classification.

    Parameters
    ----------
    sma_window : int
        Window size for SMA (used when method='sma').
    bull_threshold / bear_threshold : float
        Slope thresholds for the SMA method.
    method : {'sma', 'trend_scanning', 'causal_trend_scanning'}
        Label generation method.
    trend_horizon_min, trend_horizon_max : int
        Horizon range scanned when method='trend_scanning'.
    trend_t_threshold : float
        |t-value| threshold for Bull/Bear labels.

    Labels: 0 (Bear), 1 (Sideways), 2 (Bull)
    """

    def __init__(
        self,
        sma_window: int = 50,
        bull_threshold: float = 0.0005,  # 0.05%
        bear_threshold: float = -0.0005,   # -0.05%
        method: str = "sma",
        trend_horizon_min: int = 5,
        trend_horizon_max: int = 20,
        trend_t_threshold: float = 1.5,
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
        self.method = str(method).lower()
        self.trend_horizon_min = int(trend_horizon_min)
        self.trend_horizon_max = int(trend_horizon_max)
        self.trend_t_threshold = float(trend_t_threshold)
        if self.method not in ("sma", "trend_scanning", "causal_trend_scanning"):
            raise ValueError(
                f"Unknown regime labeling method: {self.method}. "
                "Choose 'sma', 'trend_scanning', or 'causal_trend_scanning'."
            )
    
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
        logger.info(
            "Generating regime labels (method=%s)", self.method
        )

        if self.method in ("trend_scanning", "causal_trend_scanning"):
            from .trend_scanning import TrendScanningLabeler

            labeler = TrendScanningLabeler(
                horizon_min=self.trend_horizon_min,
                horizon_max=self.trend_horizon_max,
                t_threshold=self.trend_t_threshold,
            )
            direction = "backward" if self.method == "causal_trend_scanning" else "forward"
            labels = labeler.generate_labels(price_data, direction=direction)
            label_counts = labels.value_counts().sort_index()
            logger.info(
                "%s label distribution: %s",
                "CausalTrendScanning" if direction == "backward" else "TrendScanning",
                dict(label_counts),
            )
            return labels

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

