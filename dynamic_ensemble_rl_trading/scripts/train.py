"""
Training script for regime classifier and PPO agents.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import logging
from src.data.data_processor import MarketDataHandler
from src.data.feature_extractor import TechnicalFeatureExtractor
from src.data.candlestick_generator import CandlestickGenerator
from src.data.news_sentiment import NewsSentimentExtractor
from src.data.feature_fusion import FeatureFusion
from src.regime.ground_truth import RegimeGroundTruth
from src.regime.regime_classifier import RegimeClassifier
from src.env.trading_env import MultiRegimeTradingEnv
from src.agents.agent_manager import HierarchicalAgentManager
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger(__name__, level="INFO")


def load_config(config_path: str):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_regime_classifier(config):
    """Train the regime classifier."""
    logger.info("Training regime classifier...")
    
    # Load data
    data_handler = MarketDataHandler(config['data']['ohlcv_path'])
    ohlcv_data = data_handler.load_data(
        start_date=config['training']['train_start_date'],
        end_date=config['training']['train_end_date']
    )
    
    # Generate ground truth labels
    ground_truth = RegimeGroundTruth(
        sma_window=config['regime']['sma_window'],
        bull_threshold=config['regime']['bull_threshold'],
        bear_threshold=config['regime']['bear_threshold']
    )
    
    labels = ground_truth.generate_labels(ohlcv_data[data_handler.get_ohlcv_columns()['close']])
    
    # Extract features
    tech_extractor = TechnicalFeatureExtractor()
    visual_extractor = CandlestickGenerator()
    sentiment_extractor = NewsSentimentExtractor(config['data']['news_path'])
    sentiment_extractor.load_news_data(
        start_date=config['training']['train_start_date'],
        end_date=config['training']['train_end_date']
    )
    
    feature_fusion = FeatureFusion(tech_extractor, visual_extractor, sentiment_extractor)
    state_data = feature_fusion.batch_create_unified_states(ohlcv_data, ohlcv_data.index)
    
    # Align data
    common_indices = state_data.index.intersection(labels.index)
    X = state_data.loc[common_indices].values
    y = labels.loc[common_indices].values
    
    # Train classifier
    classifier = RegimeClassifier(
        n_estimators=config['hyperparameters']['regime_classifier']['n_estimators'],
        max_depth=config['hyperparameters']['regime_classifier']['max_depth'],
        confidence_threshold=config['regime']['confidence_threshold']
    )
    
    # Split for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    classifier.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    # Save model
    model_path = Path(config['models']['regime_classifier']) / 'model.json'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(model_path))
    
    logger.info("Regime classifier training completed")


def train_ppo_agents(config):
    """Train PPO agents for all regimes."""
    logger.info("Training PPO agents...")
    
    set_seed(config.get('random_seed', 42))
    
    # Load data and create environments
    data_handler = MarketDataHandler(config['data']['ohlcv_path'])
    ohlcv_data = data_handler.load_data(
        start_date=config['training']['train_start_date'],
        end_date=config['training']['train_end_date']
    )
    
    # Create unified states
    tech_extractor = TechnicalFeatureExtractor()
    visual_extractor = CandlestickGenerator()
    sentiment_extractor = NewsSentimentExtractor(config['data']['news_path'])
    sentiment_extractor.load_news_data(
        start_date=config['training']['train_start_date'],
        end_date=config['training']['train_end_date']
    )
    
    feature_fusion = FeatureFusion(tech_extractor, visual_extractor, sentiment_extractor)
    state_data = feature_fusion.batch_create_unified_states(ohlcv_data, ohlcv_data.index)
    
    # Create environments
    bull_env = MultiRegimeTradingEnv(
        ohlcv_data=ohlcv_data,
        state_data=state_data,
        regime_type='Bull',
        initial_balance=config['training']['initial_capital'],
        transaction_fee=config['training']['transaction_fee'],
        slippage=config['training']['slippage']
    )
    
    bear_env = MultiRegimeTradingEnv(
        ohlcv_data=ohlcv_data,
        state_data=state_data,
        regime_type='Bear',
        initial_balance=config['training']['initial_capital'],
        transaction_fee=config['training']['transaction_fee'],
        slippage=config['training']['slippage']
    )
    
    sideways_env = MultiRegimeTradingEnv(
        ohlcv_data=ohlcv_data,
        state_data=state_data,
        regime_type='Sideways',
        initial_balance=config['training']['initial_capital'],
        transaction_fee=config['training']['transaction_fee'],
        slippage=config['training']['slippage']
    )
    
    # Initialize agent manager
    agent_manager = HierarchicalAgentManager(
        bull_env=bull_env,
        bear_env=bear_env,
        sideways_env=sideways_env,
        num_agents_per_pool=config['ensemble']['num_agents_per_pool']
    )
    
    # Train all pools
    total_timesteps = config['hyperparameters']['training']['total_timesteps']
    agent_manager.train_all_pools(total_timesteps=total_timesteps)
    
    # Save agents
    agents_path = Path(config['models']['ppo_agents'])
    agents_path.mkdir(parents=True, exist_ok=True)
    agent_manager.save_all_pools(str(agents_path))
    
    logger.info("PPO agents training completed")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train trading system components')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--component', type=str, choices=['regime', 'agents', 'all'],
                        default='all', help='Component to train')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.component in ['regime', 'all']:
        train_regime_classifier(config)
    
    if args.component in ['agents', 'all']:
        train_ppo_agents(config)


if __name__ == '__main__':
    main()

