"""
Evaluation script for backtesting and performance analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import pickle
import logging
from src.backtest.backtester import Backtester
from src.backtest.metrics import PerformanceMetrics
from src.visualization.plotter import TradingPlotter
from src.utils.logger import setup_logger

logger = setup_logger(__name__, level="INFO")


def load_config(config_path: str):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_system(config):
    """Evaluate the trading system."""
    logger.info("Evaluating trading system...")
    
    # Load trading results
    results_path = Path(config['results']['backtest']) / 'trading_results.pkl'
    
    if not results_path.exists():
        logger.error(f"Trading results not found: {results_path}")
        logger.info("Please run main.py first to generate trading results")
        return
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Load OHLCV data for backtesting
    from src.data.data_processor import MarketDataHandler
    data_handler = MarketDataHandler(config['data']['ohlcv_path'])
    ohlcv_data = data_handler.load_data(
        start_date=config['training']['test_start_date'],
        end_date=config['training']['test_end_date']
    )
    
    # Run backtest
    backtester = Backtester(
        initial_capital=config['training']['initial_capital'],
        transaction_fee=config['training']['transaction_fee'],
        slippage=config['training']['slippage']
    )
    
    backtest_results = backtester.run_backtest(
        results['trading_history'],
        ohlcv_data
    )
    
    # Calculate Buy & Hold benchmark
    benchmark_results = backtester.calculate_buy_hold_benchmark(
        ohlcv_data,
        pd.to_datetime(config['training']['test_start_date']),
        pd.to_datetime(config['training']['test_end_date'])
    )
    
    # Compare metrics
    metrics_calc = PerformanceMetrics()
    comparison = metrics_calc.compare_with_benchmark(
        backtest_results['metrics'],
        benchmark_results['metrics']
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("PERFORMANCE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Cumulative Return: {backtest_results['metrics']['cumulative_return']*100:.2f}%")
    logger.info(f"CAGR: {backtest_results['metrics']['cagr']*100:.2f}%")
    logger.info(f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {backtest_results['metrics']['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate: {backtest_results['metrics']['win_rate']*100:.2f}%")
    logger.info(f"Profit Factor: {backtest_results['metrics']['profit_factor']:.2f}")
    
    # Create visualizations
    plotter = TradingPlotter()
    plots_dir = Path(config['results']['plots'])
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot portfolio vs benchmark
    timestamps = pd.DatetimeIndex([r['timestamp'] for r in results['trading_history']])
    plotter.plot_portfolio_vs_benchmark(
        backtest_results['portfolio_values'],
        benchmark_results['portfolio_values'],
        timestamps[:len(backtest_results['portfolio_values'])],
        save_path=str(plots_dir / 'portfolio_vs_benchmark.png')
    )
    
    # Plot regime detection
    regimes = [r['regime'] for r in results['trading_history']]
    confidences = [r['regime_confidence'] for r in results['trading_history']]
    plotter.plot_regime_detection(
        timestamps,
        regimes,
        np.array(confidences),
        save_path=str(plots_dir / 'regime_detection.png')
    )
    
    # Plot dynamic weights
    weights_history = [r['weights'] for r in results['trading_history']]
    plotter.plot_dynamic_weights(
        timestamps,
        weights_history,
        save_path=str(plots_dir / 'dynamic_weights.png')
    )
    
    # Plot performance metrics
    plotter.plot_performance_metrics(
        backtest_results['metrics'],
        benchmark_results['metrics'],
        save_path=str(plots_dir / 'performance_metrics.png')
    )
    
    logger.info(f"Plots saved to {plots_dir}")
    logger.info("Evaluation completed")


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Evaluate trading system')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    evaluate_system(config)

