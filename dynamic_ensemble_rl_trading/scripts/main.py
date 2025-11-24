"""
Main execution script for Dynamic Ensemble RL Trading System.

Implements Algorithm 1 from the paper:
- Initialize regime classifier and agent pools
- Main trading loop with regime classification and ensemble decision
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import Dict, Any
import sys

# Import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_processor import MarketDataHandler
from src.data.feature_extractor import TechnicalFeatureExtractor
from src.data.candlestick_generator import CandlestickGenerator
from src.data.news_sentiment import NewsSentimentExtractor
from src.data.feature_fusion import FeatureFusion
from src.regime.ground_truth import RegimeGroundTruth
from src.regime.regime_classifier import RegimeClassifier
from src.env.trading_env import MultiRegimeTradingEnv
from src.agents.agent_manager import HierarchicalAgentManager
from src.ensemble.weighting import DynamicWeightCalculator
from src.ensemble.ensemble_trader import EnsembleTrader
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger(__name__, level="INFO")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all system components.
    
    Returns
    -------
    dict
        Dictionary containing all initialized components.
    """
    logger.info("Initializing system components...")
    
    # Set random seed for reproducibility
    set_seed(config.get('random_seed', 42))
    
    # Load data
    data_handler = MarketDataHandler(
        data_path=config['data']['ohlcv_path']
    )
    ohlcv_data = data_handler.load_data(
        start_date=config['training']['test_start_date'],
        end_date=config['training']['test_end_date']
    )
    
    # Initialize feature extractors
    tech_extractor = TechnicalFeatureExtractor(
        normalization_window=config['features']['technical']['normalization_window']
    )
    
    visual_extractor = CandlestickGenerator(
        image_size=config['features']['visual']['image_size'],
        lookback_hours=config['features']['visual']['lookback_hours'],
        use_resnet=config['features']['visual']['use_resnet']
    )
    
    sentiment_extractor = NewsSentimentExtractor(
        news_path=config['data']['news_path'],
        aggregation_window_hours=config['features']['sentiment']['aggregation_window']
    )
    sentiment_extractor.load_news_data(
        start_date=config['training']['test_start_date'],
        end_date=config['training']['test_end_date']
    )
    
    # Create unified states
    feature_fusion = FeatureFusion(
        technical_extractor=tech_extractor,
        visual_extractor=visual_extractor,
        sentiment_extractor=sentiment_extractor
    )
    
    state_data = feature_fusion.batch_create_unified_states(
        ohlcv_data,
        ohlcv_data.index
    )
    
    # Initialize regime classifier
    regime_classifier = RegimeClassifier(
        n_estimators=config['hyperparameters']['regime_classifier']['n_estimators'],
        max_depth=config['hyperparameters']['regime_classifier']['max_depth'],
        confidence_threshold=config['regime']['confidence_threshold']
    )
    
    # Load trained regime classifier
    regime_model_path = Path(config['models']['regime_classifier']) / 'model.json'
    if regime_model_path.exists():
        regime_classifier.load_model(str(regime_model_path))
        logger.info("Loaded trained regime classifier")
    else:
        logger.warning("Regime classifier model not found. Training required.")
    
    # Initialize trading environments
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
    
    # Load trained agents
    agents_path = Path(config['models']['ppo_agents'])
    if agents_path.exists():
        agent_manager.load_all_pools(str(agents_path))
        logger.info("Loaded trained agent pools")
    else:
        logger.warning("Agent pools not found. Training required.")
    
    # Initialize ensemble components
    weight_calculator = DynamicWeightCalculator(
        performance_window=config['ensemble']['performance_window'],
        temperature=config['ensemble']['temperature']
    )
    
    ensemble_trader = EnsembleTrader(
        weight_calculator=weight_calculator,
        performance_window=config['ensemble']['performance_window'],
        temperature=config['ensemble']['temperature']
    )
    
    logger.info("System initialization completed")
    
    return {
        'ohlcv_data': ohlcv_data,
        'state_data': state_data,
        'regime_classifier': regime_classifier,
        'agent_manager': agent_manager,
        'ensemble_trader': ensemble_trader,
        'bull_env': bull_env,
        'bear_env': bear_env,
        'sideways_env': sideways_env,
        'config': config
    }


def main_trading_loop(system_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main trading loop implementing Algorithm 1.
    
    Parameters
    ----------
    system_components : dict
        Dictionary containing all system components.
    
    Returns
    -------
    dict
        Trading results and statistics.
    """
    logger.info("Starting main trading loop...")
    
    ohlcv_data = system_components['ohlcv_data']
    state_data = system_components['state_data']
    regime_classifier = system_components['regime_classifier']
    agent_manager = system_components['agent_manager']
    ensemble_trader = system_components['ensemble_trader']
    config = system_components['config']
    
    # Initialize portfolio
    initial_capital = config['training']['initial_capital']
    portfolio_value = initial_capital
    previous_regime = 'Sideways'  # Default initial regime
    
    # Initialize ensemble trader
    num_agents = config['ensemble']['num_agents_per_pool']
    ensemble_trader.initialize_agents(num_agents)
    
    # Trading history
    trading_history = []
    portfolio_values = [initial_capital]
    regime_history = []
    action_history = []
    
    # Main loop
    timestamps = state_data.index
    num_timesteps = len(timestamps)
    
    for t in range(num_timesteps - 1):
        timestamp = timestamps[t]
        
        # Step 1: Get state S_t
        state = state_data.loc[timestamp].values
        
        # Step 2: Predict regime R_t with confidence
        regime_result = regime_classifier.predict_with_confidence(
            state,
            previous_regime=(
                0 if previous_regime == 'Bear'
                else 1 if previous_regime == 'Sideways'
                else 2
            )
        )
        
        current_regime = regime_result['regime_name']
        regime_confidence = regime_result['confidence']
        
        # Step 3: Select active pool
        active_pool = agent_manager.get_pool(current_regime)
        
        # Step 4: Calculate ensemble weights and get action
        ensemble_result = ensemble_trader.get_ensemble_action(
            state,
            active_pool
        )
        
        action = ensemble_result['action']
        weights = ensemble_result['weights']
        
        # Step 5: Execute action using active environment
        # Use the environment corresponding to current regime
        if current_regime == 'Bull':
            active_env = system_components.get('bull_env')
        elif current_regime == 'Bear':
            active_env = system_components.get('bear_env')
        else:
            active_env = system_components.get('sideways_env')
        
        if active_env is not None:
            # Execute action in environment
            next_obs, reward, terminated, truncated, info = active_env.step(action)
            
            # Update portfolio value from environment
            portfolio_value = info.get('portfolio_value', portfolio_value)
            
            # Update agent performance for weighting
            if 'agent_index' in info:
                ensemble_trader.update_agent_performance(
                    info['agent_index'],
                    portfolio_value
                )
        else:
            # Fallback: simplified portfolio update
            # In production, always use environment
            price_change = 0.0
            if t > 0:
                prev_price = ohlcv_data.loc[timestamps[t-1], 'close']
                curr_price = ohlcv_data.loc[timestamp, 'close']
                price_change = (curr_price - prev_price) / prev_price
            
            portfolio_value = portfolio_value * (1 + price_change * 0.5)  # Simplified
        
        # Record history
        trading_history.append({
            'timestamp': timestamp,
            'regime': current_regime,
            'regime_confidence': regime_confidence,
            'action': action,
            'weights': weights,
            'portfolio_value': portfolio_value
        })
        
        portfolio_values.append(portfolio_value)
        regime_history.append(current_regime)
        action_history.append(action)
        
        # Update previous regime
        previous_regime = current_regime
        
        if (t + 1) % 100 == 0:
            logger.info(
                f"Step {t+1}/{num_timesteps-1}: "
                f"Regime={current_regime}, "
                f"Portfolio={portfolio_value:.2f}"
            )
    
    logger.info("Main trading loop completed")
    
    return {
        'trading_history': trading_history,
        'portfolio_values': portfolio_values,
        'regime_history': regime_history,
        'action_history': action_history,
        'final_portfolio_value': portfolio_value
    }


def main():
    """Main entry point."""
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    config = load_config(str(config_path))
    
    # Initialize system
    system_components = initialize_system(config)
    
    # Run main trading loop
    results = main_trading_loop(system_components)
    
    # Save results
    results_path = Path(config['results']['backtest']) / 'trading_results.pkl'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Final portfolio value: {results['final_portfolio_value']:.2f}")


if __name__ == '__main__':
    main()

