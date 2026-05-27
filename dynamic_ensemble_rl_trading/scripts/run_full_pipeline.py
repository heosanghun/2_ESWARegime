"""
Full pipeline script: data generation -> model training -> evaluation -> verification

Automates the full process for verifying paper performance metrics.
"""

import sys
from pathlib import Path
import yaml
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_synthetic_data import generate_ohlcv_data, generate_news_data
from scripts.train import train_regime_classifier, train_ppo_agents
from scripts.main import main as run_trading_system

# Configure logging
log_dir = Path('results')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_dir / 'full_pipeline.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """Load configuration."""
    config_dir = Path(config_path).parent
    
    # Load main config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load hyperparameters if exists
    hyperparams_path = config_dir / 'hyperparameters.yaml'
    if hyperparams_path.exists():
        with open(hyperparams_path, 'r') as f:
            hyperparams = yaml.safe_load(f)
        config['hyperparameters'] = hyperparams
    else:
        # Default hyperparameters
        config['hyperparameters'] = {
            'regime_classifier': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            },
            'training': {
                'total_timesteps': 100000  # Reduced for faster testing
            }
        }
    
    return config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run full pipeline: data generation -> training -> evaluation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data generation (use existing data)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use existing models)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode with reduced timesteps'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 100)
    logger.info("Starting full pipeline")
    logger.info("=" * 100)
    
    # Load config
    config = load_config(args.config)
    
    # Step 1: Generate data (if not skipped)
    if not args.skip_data:
        logger.info("\n" + "=" * 100)
        logger.info("Step 1: Data generation")
        logger.info("=" * 100)
        
        try:
            generate_ohlcv_data(
                start_date=config['training']['train_start_date'],
                end_date=config['training']['test_end_date'],
                output_path=config['data']['ohlcv_path']
            )
            
            generate_news_data(
                start_date=config['training']['train_start_date'],
                end_date=config['training']['test_end_date'],
                total_articles=31037,
                output_path=config['data']['news_path']
            )
            
            logger.info("Data generation complete!")
        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            return
    else:
        logger.info("Skipping data generation (using existing data)")
    
    # Step 2: Train models (if not skipped)
    if not args.skip_training:
        logger.info("\n" + "=" * 100)
        logger.info("Step 2: Model training")
        logger.info("=" * 100)
        
        # Adjust timesteps for quick test
        if args.quick_test:
            original_timesteps = config['hyperparameters']['training']['total_timesteps']
            config['hyperparameters']['training']['total_timesteps'] = 10000
            logger.info(f"Quick test mode: reduced timesteps from {original_timesteps} to 10000")
        
        try:
            # Train regime classifier
            logger.info("Training Regime Classifier...")
            train_regime_classifier(config)
            
            # Train PPO agents
            logger.info("Training PPO Agents...")
            train_ppo_agents(config)
            
            logger.info("Model training complete!")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
    else:
        logger.info("Skipping model training (using existing models)")
    
    # Step 3: Run trading system and evaluation
    logger.info("\n" + "=" * 100)
    logger.info("Step 3: Run trading system and evaluation")
    logger.info("=" * 100)
    
    try:
        # Run main trading system
        run_trading_system()
        
        logger.info("Trading system execution complete!")
    except Exception as e:
        logger.error(f"Trading system execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    logger.info("\n" + "=" * 100)
    logger.info("Full pipeline complete!")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
