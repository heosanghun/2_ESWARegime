"""
Complete evaluation script for all methods.

This script:
1. Loads or generates trading histories for all methods
2. Evaluates each method using backtesting
3. Generates Table 2 format comparison
4. Saves comprehensive results
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import logging
from typing import Dict, List, Optional, Any
import pickle
from datetime import datetime

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompleteEvaluator:
    """
    Complete evaluator for all trading methods.
    
    Handles:
    - Data loading
    - Method execution (or loading pre-computed results)
    - Backtesting
    - Performance comparison
    - Report generation
    """
    
    def __init__(self, config_path: str):
        """
        Initialize complete evaluator.
        
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
        
        # Results storage
        self.all_results = {}
        
        logger.info("Initialized Complete Evaluator")
    
    def load_data(self) -> pd.DataFrame:
        """Load OHLCV data."""
        logger.info("Loading OHLCV data...")
        data_handler = MarketDataHandler(self.config['data']['ohlcv_path'])
        ohlcv_data = data_handler.load_data(
            start_date=self.config['training']['test_start_date'],
            end_date=self.config['training']['test_end_date']
        )
        logger.info(f"Loaded {len(ohlcv_data)} data points")
        return ohlcv_data
    
    def evaluate_buy_hold(self, ohlcv_data: pd.DataFrame) -> Dict:
        """Evaluate Buy & Hold baseline."""
        logger.info("=" * 80)
        logger.info("Evaluating Buy & Hold")
        logger.info("=" * 80)
        
        start_date = pd.to_datetime(self.config['training']['test_start_date'])
        end_date = pd.to_datetime(self.config['training']['test_end_date'])
        
        result = self.backtester.calculate_buy_hold_benchmark(
            ohlcv_data,
            start_date,
            end_date
        )
        
        metrics = result['metrics']
        self.all_results['Buy & Hold'] = {
            'metrics': metrics,
            'portfolio_values': result['portfolio_values'],
            'returns': result['returns']
        }
        
        logger.info(f"Buy & Hold Results:")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Cumulative Return: {metrics['cumulative_return']*100:.2f}%")
        logger.info(f"  CAGR: {metrics['cagr']*100:.2f}%")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        
        return metrics
    
    def evaluate_method_from_history(
        self,
        method_name: str,
        trading_history: List[Dict],
        ohlcv_data: pd.DataFrame
    ) -> Dict:
        """
        Evaluate method from trading history.
        
        Parameters
        ----------
        method_name : str
            Name of the method.
        trading_history : list of dict
            Trading history with 'timestamp' and 'action' keys.
        ohlcv_data : pd.DataFrame
            OHLCV data.
        
        Returns
        -------
        dict
            Performance metrics.
        """
        logger.info("=" * 80)
        logger.info(f"Evaluating {method_name}")
        logger.info("=" * 80)
        
        if len(trading_history) == 0:
            logger.warning(f"No trading history for {method_name}, skipping...")
            return None
        
        result = self.backtester.run_backtest(
            trading_history,
            ohlcv_data
        )
        
        metrics = result['metrics']
        self.all_results[method_name] = {
            'metrics': metrics,
            'portfolio_values': result['portfolio_values'],
            'returns': result['returns'],
            'trades': result.get('trades', [])
        }
        
        logger.info(f"{method_name} Results:")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Cumulative Return: {metrics['cumulative_return']*100:.2f}%")
        logger.info(f"  CAGR: {metrics['cagr']*100:.2f}%")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        if 'profit_factor' in metrics:
            logger.info(f"  Profit Factor: {metrics['profit_factor']:.4f}")
        
        return metrics
    
    def load_precomputed_results(self, results_file: str) -> Dict[str, List[Dict]]:
        """
        Load pre-computed trading histories.
        
        Parameters
        ----------
        results_file : str
            Path to pickle file with trading histories.
        
        Returns
        -------
        dict
            Dictionary mapping method names to trading histories.
        """
        if not Path(results_file).exists():
            logger.warning(f"Results file not found: {results_file}")
            return {}
        
        logger.info(f"Loading pre-computed results from {results_file}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def generate_comparison_table(self) -> ComparisonTable:
        """Generate comparison table from all results."""
        logger.info("Generating comparison table...")
        
        for method_name, result_data in self.all_results.items():
            if 'metrics' in result_data:
                self.comparison_table.add_result(method_name, result_data['metrics'])
        
        return self.comparison_table
    
    def save_all_results(self, output_dir: str) -> None:
        """
        Save all results to files.
        
        Parameters
        ----------
        output_dir : str
            Output directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        comparison_table = self.generate_comparison_table()
        comparison_table.save_csv(str(output_path / 'comparison_table.csv'))
        comparison_table.save_markdown(str(output_path / 'comparison_table.md'))
        comparison_table.save_latex(str(output_path / 'comparison_table.tex'))
        
        # Save detailed results
        results_file = output_path / 'detailed_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(self.all_results, f)
        
        # Print table
        comparison_table.print_table()
        
        logger.info(f"All results saved to {output_dir}")
    
    def compare_with_paper(self) -> Dict:
        """
        Compare results with paper's reported performance.
        
        Returns
        -------
        dict
            Comparison results.
        """
        paper_metrics = {
            'Proposed Method': {
                'sharpe_ratio': 1.89,
                'cumulative_return': 0.893,
                'cagr': 0.342,
                'max_drawdown': -0.162,
                'win_rate': 0.678
            }
        }
        
        comparison = {}
        
        if 'Proposed Method' in self.all_results:
            our_metrics = self.all_results['Proposed Method']['metrics']
            paper_metrics_proposed = paper_metrics['Proposed Method']
            
            comparison['Proposed Method'] = {}
            for metric_name in paper_metrics_proposed:
                our_val = our_metrics.get(metric_name, 0)
                paper_val = paper_metrics_proposed[metric_name]
                diff = our_val - paper_val
                diff_pct = (diff / abs(paper_val)) * 100 if paper_val != 0 else 0
                
                comparison['Proposed Method'][metric_name] = {
                    'ours': our_val,
                    'paper': paper_val,
                    'difference': diff,
                    'difference_pct': diff_pct,
                    'within_tolerance': abs(diff_pct) < 5.0  # 5% tolerance
                }
        
        return comparison


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all trading methods')
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
        '--precomputed-results',
        type=str,
        default=None,
        help='Path to pickle file with pre-computed trading histories'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CompleteEvaluator(args.config)
    
    # Load data
    ohlcv_data = evaluator.load_data()
    
    # Evaluate Buy & Hold
    evaluator.evaluate_buy_hold(ohlcv_data)
    
    # Load pre-computed results if available
    if args.precomputed_results:
        methods_results = evaluator.load_precomputed_results(args.precomputed_results)
        
        # Evaluate all methods
        for method_name, trading_history in methods_results.items():
            evaluator.evaluate_method_from_history(
                method_name,
                trading_history,
                ohlcv_data
            )
    else:
        logger.warning("No pre-computed results provided.")
        logger.warning("Please run individual method evaluations first or provide --precomputed-results")
    
    # Compare with paper
    paper_comparison = evaluator.compare_with_paper()
    
    if paper_comparison:
        logger.info("=" * 80)
        logger.info("Comparison with Paper Results")
        logger.info("=" * 80)
        for method_name, metrics_comp in paper_comparison.items():
            logger.info(f"\n{method_name}:")
            for metric_name, comp_data in metrics_comp.items():
                status = "✓" if comp_data['within_tolerance'] else "✗"
                logger.info(
                    f"  {status} {metric_name}: "
                    f"Ours={comp_data['ours']:.4f}, "
                    f"Paper={comp_data['paper']:.4f}, "
                    f"Diff={comp_data['difference_pct']:.2f}%"
                )
    
    # Save all results
    evaluator.save_all_results(args.results_dir)
    
    logger.info("=" * 80)
    logger.info("Evaluation completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
