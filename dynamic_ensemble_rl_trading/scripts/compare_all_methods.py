"""
Comprehensive comparison script for all methods.

This script evaluates all baseline methods, ablation models, and the proposed method,
then generates Table 2 format comparison results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import logging
from typing import Dict, List, Optional
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.backtester import Backtester
from src.backtest.metrics import PerformanceMetrics
from src.evaluation.comparison_table import ComparisonTable
from src.evaluation.comprehensive_metrics import ComprehensiveMetrics
from src.data.data_processor import MarketDataHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MethodEvaluator:
    """
    Evaluator for all trading methods.
    
    Evaluates:
    - Buy & Hold
    - Single PPO Agent
    - XGBoost Trader
    - CNN Trader
    - Simple Ensemble
    - Proposed Method (Dynamic Ensemble)
    - Ablation Models (Model 1-4)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize method evaluator.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.backtester = Backtester(
            initial_capital=self.config['training']['initial_capital'],
            transaction_fee=self.config['training']['transaction_fee'],
            slippage=self.config['training']['slippage']
        )
        
        self.metrics_calc = PerformanceMetrics()
        self.comprehensive_metrics = ComprehensiveMetrics()
        self.comparison_table = ComparisonTable()
        
        logger.info("Initialized Method Evaluator")
    
    def evaluate_buy_hold(
        self,
        ohlcv_data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict:
        """
        Evaluate Buy & Hold baseline.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data.
        start_date : pd.Timestamp
            Start date.
        end_date : pd.Timestamp
            End date.
        
        Returns
        -------
        dict
            Performance metrics.
        """
        logger.info("Evaluating Buy & Hold...")
        
        result = self.backtester.calculate_buy_hold_benchmark(
            ohlcv_data,
            start_date,
            end_date
        )
        
        metrics = result['metrics']
        logger.info(f"Buy & Hold - Sharpe: {metrics['sharpe_ratio']:.2f}")
        
        return metrics
    
    def evaluate_method(
        self,
        method_name: str,
        trading_history: List[Dict],
        ohlcv_data: pd.DataFrame
    ) -> Dict:
        """
        Evaluate a trading method.
        
        Parameters
        ----------
        method_name : str
            Name of the method.
        trading_history : list of dict
            Trading history with timestamps and actions.
        ohlcv_data : pd.DataFrame
            OHLCV data for backtesting.
        
        Returns
        -------
        dict
            Performance metrics.
        """
        logger.info(f"Evaluating {method_name}...")
        
        result = self.backtester.run_backtest(
            trading_history,
            ohlcv_data
        )
        
        metrics = result['metrics']
        logger.info(f"{method_name} - Sharpe: {metrics['sharpe_ratio']:.2f}")
        
        return metrics
    
    def run_all_evaluations(
        self,
        methods_results: Dict[str, List[Dict]],
        ohlcv_data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> ComparisonTable:
        """
        Run evaluations for all methods.
        
        Parameters
        ----------
        methods_results : dict
            Dictionary mapping method names to trading histories.
        ohlcv_data : pd.DataFrame
            OHLCV data.
        start_date : pd.Timestamp
            Start date.
        end_date : pd.Timestamp
            End date.
        
        Returns
        -------
        ComparisonTable
            Comparison table with all results.
        """
        # Evaluate Buy & Hold
        buy_hold_metrics = self.evaluate_buy_hold(ohlcv_data, start_date, end_date)
        self.comparison_table.add_result('Buy & Hold', buy_hold_metrics)
        
        # Evaluate all other methods
        for method_name, trading_history in methods_results.items():
            if len(trading_history) > 0:
                metrics = self.evaluate_method(method_name, trading_history, ohlcv_data)
                self.comparison_table.add_result(method_name, metrics)
            else:
                logger.warning(f"No trading history for {method_name}, skipping...")
        
        return self.comparison_table
    
    def save_results(
        self,
        output_dir: str,
        comparison_table: ComparisonTable
    ) -> None:
        """
        Save all results to files.
        
        Parameters
        ----------
        output_dir : str
            Output directory.
        comparison_table : ComparisonTable
            Comparison table to save.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        comparison_table.save_csv(str(output_path / 'comparison_table.csv'))
        comparison_table.save_markdown(str(output_path / 'comparison_table.md'))
        comparison_table.save_latex(str(output_path / 'comparison_table.tex'))
        
        # Print table
        comparison_table.print_table()
        
        logger.info(f"Results saved to {output_dir}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare all trading methods')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/comparison',
        help='Directory to save results'
    )
    parser.add_argument(
        '--methods-file',
        type=str,
        default=None,
        help='Path to pickle file containing method results'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MethodEvaluator(args.config)
    
    # Load OHLCV data
    data_handler = MarketDataHandler(evaluator.config['data']['ohlcv_path'])
    ohlcv_data = data_handler.load_data(
        start_date=evaluator.config['training']['test_start_date'],
        end_date=evaluator.config['training']['test_end_date']
    )
    
    start_date = pd.to_datetime(evaluator.config['training']['test_start_date'])
    end_date = pd.to_datetime(evaluator.config['training']['test_end_date'])
    
    # Load method results
    if args.methods_file and Path(args.methods_file).exists():
        logger.info(f"Loading method results from {args.methods_file}")
        with open(args.methods_file, 'rb') as f:
            methods_results = pickle.load(f)
    else:
        logger.warning("No methods file provided. Using empty results.")
        logger.warning("Please run individual method evaluations first.")
        methods_results = {}
    
    # Run evaluations
    comparison_table = evaluator.run_all_evaluations(
        methods_results,
        ohlcv_data,
        start_date,
        end_date
    )
    
    # Save results
    evaluator.save_results(args.results_dir, comparison_table)
    
    logger.info("Comparison completed!")


if __name__ == "__main__":
    main()
