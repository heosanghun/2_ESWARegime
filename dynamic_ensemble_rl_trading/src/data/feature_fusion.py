"""
Multimodal feature fusion module.

This module combines visual, technical, and sentiment features into
a unified state vector S_t as specified in the paper.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from .feature_extractor import TechnicalFeatureExtractor
from .candlestick_generator import CandlestickGenerator
from .news_sentiment import NewsSentimentExtractor

logger = logging.getLogger(__name__)


class FeatureFusion:
    """
    Combines multimodal features into unified state vector.
    
    Implements the Multimodal Feature Fusion Layer that combines:
    - F_visual: Visual features from candlestick charts (CNN)
    - F_tech: Technical features from 15 indicators
    - F_senti: Sentiment features from news analysis
    
    Final state vector: S_t = concatenate(F_visual, F_tech, F_senti)
    """
    
    def __init__(
        self,
        technical_extractor: TechnicalFeatureExtractor,
        visual_extractor: CandlestickGenerator,
        sentiment_extractor: NewsSentimentExtractor
    ):
        """
        Initialize the FeatureFusion module.
        
        Parameters
        ----------
        technical_extractor : TechnicalFeatureExtractor
            Extractor for technical features.
        visual_extractor : CandlestickGenerator
            Extractor for visual features.
        sentiment_extractor : NewsSentimentExtractor
            Extractor for sentiment features.
        """
        self.technical_extractor = technical_extractor
        self.visual_extractor = visual_extractor
        self.sentiment_extractor = sentiment_extractor
    
    def create_unified_state(
        self,
        ohlcv_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        ohlcv_columns: Optional[dict] = None
    ) -> np.ndarray:
        """
        Create unified state vector S_t for a specific timestamp.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data with datetime index.
        timestamp : pd.Timestamp
            Target timestamp.
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names.
        
        Returns
        -------
        np.ndarray
            Unified state vector S_t.
        """
        # Extract technical features F_tech
        tech_features_df = self.technical_extractor.extract_features(ohlcv_data)
        
        if timestamp not in tech_features_df.index:
            # Find closest timestamp
            closest_idx = tech_features_df.index.get_indexer(
                [timestamp], method='nearest'
            )[0]
            if closest_idx == -1:
                raise ValueError(f"No technical features available for {timestamp}")
            tech_features = tech_features_df.iloc[closest_idx].values
        else:
            tech_features = tech_features_df.loc[timestamp].values
        
        # Extract visual features F_visual
        try:
            visual_features = self.visual_extractor.process_timestamp(
                ohlcv_data, timestamp, ohlcv_columns
            )
        except Exception as e:
            logger.warning(
                f"Error extracting visual features for {timestamp}: {e}. "
                "Using zero vector."
            )
            # Fallback: use zero vector with expected dimension
            visual_features = np.zeros(512)  # ResNet-18 feature dimension
        
        # Extract sentiment features F_senti
        try:
            sentiment_df = self.sentiment_extractor.get_sentiment_features(
                pd.DatetimeIndex([timestamp])
            )
            sentiment_features = sentiment_df.iloc[0].values
        except Exception as e:
            logger.warning(
                f"Error extracting sentiment features for {timestamp}: {e}. "
                "Using zero vector."
            )
            # Fallback: use zero vector
            sentiment_features = np.zeros(8)  # Expected sentiment feature count
        
        # Concatenate all features: S_t = [F_visual, F_tech, F_senti]
        unified_state = np.concatenate([
            visual_features,
            tech_features,
            sentiment_features
        ])
        
        # Validate dimensions
        if np.any(np.isnan(unified_state)) or np.any(np.isinf(unified_state)):
            logger.warning(
                f"Found NaN or Inf in unified state for {timestamp}. "
                "Replacing with zeros."
            )
            unified_state = np.nan_to_num(unified_state, nan=0.0, posinf=0.0, neginf=0.0)
        
        return unified_state
    
    def batch_create_unified_states(
        self,
        ohlcv_data: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        ohlcv_columns: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Create unified state vectors for multiple timestamps.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data with datetime index.
        timestamps : pd.DatetimeIndex
            Target timestamps.
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with unified state vectors, indexed by timestamp.
        """
        logger.info(f"Creating unified states for {len(timestamps)} timestamps")
        
        # Extract all features in batch
        tech_features_df = self.technical_extractor.extract_features(ohlcv_data)
        
        visual_features_df = self.visual_extractor.batch_process(
            ohlcv_data, timestamps, ohlcv_columns
        )
        
        sentiment_features_df = self.sentiment_extractor.get_sentiment_features(
            timestamps
        )
        
        # Align indices
        common_timestamps = (
            tech_features_df.index.intersection(timestamps)
            .intersection(visual_features_df.index)
            .intersection(sentiment_features_df.index)
        )
        
        if len(common_timestamps) == 0:
            raise ValueError("No common timestamps found across all feature types")
        
        # Combine features
        unified_states = []
        valid_timestamps = []
        
        for timestamp in common_timestamps:
            try:
                tech_features = tech_features_df.loc[timestamp].values
                visual_features = visual_features_df.loc[timestamp].values
                sentiment_features = sentiment_features_df.loc[timestamp].values
                
                unified_state = np.concatenate([
                    visual_features,
                    tech_features,
                    sentiment_features
                ])
                
                # Clean any invalid values
                unified_state = np.nan_to_num(
                    unified_state, nan=0.0, posinf=0.0, neginf=0.0
                )
                
                unified_states.append(unified_state)
                valid_timestamps.append(timestamp)
            except Exception as e:
                logger.warning(
                    f"Error creating unified state for {timestamp}: {e}"
                )
                continue
        
        if not unified_states:
            raise ValueError("No valid unified states created")
        
        # Create DataFrame
        state_dim = len(unified_states[0])
        unified_df = pd.DataFrame(
            unified_states,
            index=valid_timestamps,
            columns=[f'state_feat_{i}' for i in range(state_dim)]
        )
        
        logger.info(
            f"Created {len(unified_df)} unified state vectors "
            f"with dimension {state_dim}"
        )
        
        return unified_df
    
    def get_state_dimension(self, sample_features_df: Optional[pd.DataFrame] = None) -> int:
        """
        Get the dimension of the unified state vector.
        
        Parameters
        ----------
        sample_features_df : pd.DataFrame, optional
            Sample technical features DataFrame to get actual dimension.
            If None, uses estimated dimensions.
        
        Returns
        -------
        int
            Dimension of S_t.
        """
        # Visual features: ResNet-18 output (512)
        visual_dim = 512
        
        # Technical features: number of normalized indicators
        if sample_features_df is not None:
            tech_dim = len(sample_features_df.columns)
        else:
            tech_dim = len(self.technical_extractor.get_feature_names())
        
        # Sentiment features: 8 features
        sentiment_dim = 8
        
        return visual_dim + tech_dim + sentiment_dim

