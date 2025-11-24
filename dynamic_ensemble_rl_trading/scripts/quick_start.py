"""
Quick start script for testing the system with minimal setup.

This script provides a simplified way to test the system components
without requiring full data setup.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger(__name__, level="INFO")


def create_sample_data(num_points=1000):
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2021-01-01', periods=num_points, freq='H')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(num_points) * 0.01
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(num_points) * 0.001),
        'high': prices * (1 + abs(np.random.randn(num_points)) * 0.002),
        'low': prices * (1 - abs(np.random.randn(num_points)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, num_points)
    }, index=dates)
    
    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


def test_basic_components():
    """Test basic system components."""
    logger.info("Testing basic components...")
    
    set_seed(42)
    
    # Create sample data
    ohlcv_data = create_sample_data(500)
    logger.info(f"Created sample OHLCV data: {len(ohlcv_data)} points")
    
    # Test technical feature extraction
    from src.data.feature_extractor import TechnicalFeatureExtractor
    tech_extractor = TechnicalFeatureExtractor()
    tech_features = tech_extractor.extract_features(ohlcv_data)
    logger.info(f"Extracted {len(tech_features.columns)} technical features")
    
    # Test regime ground truth
    from src.regime.ground_truth import RegimeGroundTruth
    gt = RegimeGroundTruth()
    labels = gt.generate_labels(ohlcv_data['close'])
    logger.info(f"Generated {len(labels)} regime labels")
    logger.info(f"Regime distribution: {labels.value_counts().to_dict()}")
    
    # Test reward calculator
    from src.env.rewards import RegimeRewardCalculator
    reward_calc = RegimeRewardCalculator()
    
    bull_reward = reward_calc.calculate_bull_reward(10000, 10500)
    bear_reward = reward_calc.calculate_bear_reward(10000, 9800, 10)
    sideways_reward = reward_calc.calculate_sideways_reward(10000, 10050, 5)
    
    logger.info(f"Bull reward: {bull_reward:.4f}")
    logger.info(f"Bear reward: {bear_reward:.4f}")
    logger.info(f"Sideways reward: {sideways_reward:.4f}")
    
    # Test dynamic weighting
    from src.ensemble.weighting import DynamicWeightCalculator
    weight_calc = DynamicWeightCalculator()
    weight_calc.initialize_agents(5)
    
    # Add some returns
    for i in range(30):
        for agent_idx in range(5):
            return_val = np.random.randn() * 0.01
            weight_calc.update_returns(agent_idx, return_val)
    
    weights = weight_calc.calculate_weights()
    logger.info(f"Calculated weights: {weights}")
    logger.info(f"Weights sum: {np.sum(weights):.6f}")
    
    logger.info("Basic component tests completed successfully")


def main():
    """Main quick start function."""
    logger.info("=" * 60)
    logger.info("Quick Start - Component Testing")
    logger.info("=" * 60)
    
    try:
        test_basic_components()
        logger.info("=" * 60)
        logger.info("All quick start tests passed")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error in quick start: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

