"""
Example usage script demonstrating how to use the trading system.

This script shows a simplified example of how to use the main components
of the Dynamic Ensemble RL Trading System.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from src.data.data_processor import MarketDataHandler
from src.data.feature_extractor import TechnicalFeatureExtractor
from src.data.candlestick_generator import CandlestickGenerator
from src.data.news_sentiment import NewsSentimentExtractor
from src.data.feature_fusion import FeatureFusion
from src.regime.ground_truth import RegimeGroundTruth
from src.regime.regime_classifier import RegimeClassifier
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger(__name__, level="INFO")


def example_data_processing():
    """Example: Data processing and feature extraction."""
    logger.info("Example: Data Processing")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data handler
    data_handler = MarketDataHandler(
        data_path=config['data']['ohlcv_path']
    )
    
    # Load data
    ohlcv_data = data_handler.load_data(
        start_date='2021-10-12',
        end_date='2021-12-31'
    )
    
    logger.info(f"Loaded {len(ohlcv_data)} data points")
    
    # Extract technical features
    tech_extractor = TechnicalFeatureExtractor()
    tech_features = tech_extractor.extract_features(ohlcv_data)
    
    logger.info(f"Extracted {len(tech_features.columns)} technical features")
    
    return ohlcv_data, tech_features


def example_regime_classification():
    """Example: Regime classification."""
    logger.info("Example: Regime Classification")
    
    # Create sample price data
    dates = pd.date_range('2021-01-01', periods=200, freq='H')
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    price_series = pd.Series(prices, index=dates)
    
    # Generate ground truth labels
    gt = RegimeGroundTruth(
        sma_window=50,
        bull_threshold=0.0005,
        bear_threshold=-0.0005
    )
    
    labels = gt.generate_labels(price_series)
    
    logger.info(f"Generated {len(labels)} regime labels")
    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
    
    return labels


def example_feature_fusion():
    """Example: Multimodal feature fusion."""
    logger.info("Example: Feature Fusion")
    
    # This is a simplified example
    # In practice, you would load actual data
    
    # Initialize extractors
    tech_extractor = TechnicalFeatureExtractor()
    visual_extractor = CandlestickGenerator(use_resnet=False)  # Simplified
    sentiment_extractor = NewsSentimentExtractor(
        news_path='data/cryptonews_2021-10-12_2023-12-19.csv'
    )
    
    # Initialize fusion
    fusion = FeatureFusion(
        technical_extractor=tech_extractor,
        visual_extractor=visual_extractor,
        sentiment_extractor=sentiment_extractor
    )
    
    state_dim = fusion.get_state_dimension()
    logger.info(f"Unified state vector dimension: {state_dim}")
    
    return fusion


def main():
    """Main example function."""
    logger.info("=" * 60)
    logger.info("Dynamic Ensemble RL Trading System - Example Usage")
    logger.info("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    try:
        # Example 1: Data processing
        ohlcv_data, tech_features = example_data_processing()
        logger.info("Data processing example completed")
        
        # Example 2: Regime classification
        labels = example_regime_classification()
        logger.info("Regime classification example completed")
        
        # Example 3: Feature fusion
        fusion = example_feature_fusion()
        logger.info("Feature fusion example completed")
        
        logger.info("=" * 60)
        logger.info("All examples completed successfully")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.warning(f"Data file not found: {e}")
        logger.info("Please ensure data files are in the correct location")
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise


if __name__ == '__main__':
    main()

