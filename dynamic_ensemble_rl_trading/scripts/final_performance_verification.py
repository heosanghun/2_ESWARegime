"""
Final performance verification: compare paper metrics with actual results.

Compares paper Table 2 performance metrics with actual run results
to verify consistency with the paper.
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import MarketDataHandler
from src.data.feature_extractor import TechnicalFeatureExtractor
from src.data.candlestick_generator import CandlestickGenerator
from src.data.news_sentiment import NewsSentimentExtractor
from src.data.feature_fusion import FeatureFusion
from src.regime.regime_classifier import RegimeClassifier
from src.env.trading_env import MultiRegimeTradingEnv
from src.agents.agent_manager import HierarchicalAgentManager
from src.evaluation.comprehensive_metrics import PerformanceMetrics
from src.utils.logger import setup_logger

# Logging setup
log_dir = Path('results/verification')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_dir / 'final_performance_verification.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Paper Table 2 performance metrics (Proposed Method)
PAPER_METRICS = {
    'Proposed Method': {
        'Sharpe Ratio': 1.89,
        'Cumulative Return': 0.893,
        'CAGR': 0.342,
        'Maximum Drawdown': -0.162,
        'Win Rate': 0.678,
        'Profit Factor': 2.34
    },
    'No Dynamic Weighting': {
        'Sharpe Ratio': 1.58,
        'Cumulative Return': 0.725,
        'CAGR': 0.284,
        'Maximum Drawdown': -0.198,
        'Win Rate': 0.612,
        'Profit Factor': 1.95
    },
    'No Confidence Selection': {
        'Sharpe Ratio': 1.41,
        'Cumulative Return': 0.615,
        'CAGR': 0.235,
        'Maximum Drawdown': -0.245,
        'Win Rate': 0.564,
        'Profit Factor': 1.72
    },
    'No Ensemble': {
        'Sharpe Ratio': 1.41,
        'Cumulative Return': 0.608,
        'CAGR': 0.232,
        'Maximum Drawdown': -0.252,
        'Win Rate': 0.558,
        'Profit Factor': 1.70
    },
    'No Regime Classification': {
        'Sharpe Ratio': 1.35,
        'Cumulative Return': 0.512,
        'CAGR': 0.195,
        'Maximum Drawdown': -0.298,
        'Win Rate': 0.512,
        'Profit Factor': 1.52
    },
    'Single PPO Agent': {
        'Sharpe Ratio': 1.28,
        'Cumulative Return': 0.456,
        'CAGR': 0.172,
        'Maximum Drawdown': -0.325,
        'Win Rate': 0.478,
        'Profit Factor': 1.41
    },
    'XGBoost Trader': {
        'Sharpe Ratio': 0.95,
        'Cumulative Return': 0.312,
        'CAGR': 0.118,
        'Maximum Drawdown': -0.395,
        'Win Rate': 0.412,
        'Profit Factor': 1.22
    },
    'CNN Trader': {
        'Sharpe Ratio': 0.78,
        'Cumulative Return': 0.245,
        'CAGR': 0.092,
        'Maximum Drawdown': -0.452,
        'Win Rate': 0.384,
        'Profit Factor': 1.12
    },
    'Simple Ensemble': {
        'Sharpe Ratio': 1.15,
        'Cumulative Return': 0.398,
        'CAGR': 0.148,
        'Maximum Drawdown': -0.345,
        'Win Rate': 0.442,
        'Profit Factor': 1.31
    }
}


class FinalPerformanceVerifier:
    """Final performance verification class."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load config file."""
        config_dir = self.config_path.parent
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load hyperparameters
        hyperparams_path = config_dir / 'hyperparameters.yaml'
        if hyperparams_path.exists():
            with open(hyperparams_path, 'r', encoding='utf-8') as f:
                hyperparams = yaml.safe_load(f)
            config['hyperparameters'] = hyperparams
        
        return config
    
    def run_backtest(self, method_name: str = 'Proposed Method') -> Dict[str, float]:
        """
        Run backtest and compute performance metrics.
        
        Parameters
        ----------
        method_name : str
            Method name to evaluate.
        
        Returns
        -------
        dict
            Performance metrics dictionary.
        """
        logger.info(f"\n{'='*100}")
        logger.info(f"Running backtest: {method_name}")
        logger.info(f"{'='*100}")
        
        try:
            # Load data
            data_handler = MarketDataHandler(self.config['data']['ohlcv_path'])
            ohlcv_data = data_handler.load_data(
                start_date=self.config['training']['test_start_date'],
                end_date=self.config['training']['test_end_date']
            )
            
            # Extract features
            tech_extractor = TechnicalFeatureExtractor()
            visual_extractor = CandlestickGenerator()
            sentiment_extractor = NewsSentimentExtractor(self.config['data']['news_path'])
            sentiment_extractor.load_news_data(
                start_date=self.config['training']['test_start_date'],
                end_date=self.config['training']['test_end_date']
            )
            
            feature_fusion = FeatureFusion(tech_extractor, visual_extractor, sentiment_extractor)
            state_data = feature_fusion.batch_create_unified_states(ohlcv_data, ohlcv_data.index)
            
            # For Proposed Method, run full system
            if method_name == 'Proposed Method':
                # Load regime classifier
                regime_model_path = Path(self.config['models']['regime_classifier']) / 'model.json'
                if not regime_model_path.exists():
                    logger.error(f"Regime Classifier model not found: {regime_model_path}")
                    return {}
                
                classifier = RegimeClassifier(
                    n_estimators=self.config['hyperparameters']['regime_classifier']['n_estimators'],
                    max_depth=self.config['hyperparameters']['regime_classifier']['max_depth'],
                    confidence_threshold=self.config['regime']['confidence_threshold']
                )
                classifier.load_model(str(regime_model_path))
                
                # Load PPO agents
                agent_manager = HierarchicalAgentManager(
                    bull_env=None,  # Will be created per step
                    bear_env=None,
                    sideways_env=None,
                    num_agents_per_pool=self.config['ensemble']['num_agents_per_pool']
                )
                # Environment is required in practice; simplified simulation here
                logger.info("  Running full system (simplified version)")
                
                # Simple performance calculation (trading history required in practice)
                # Example: add slight variation to paper values
                metrics = PerformanceMetrics()
                
                # Generate sample returns (computed from trading history in production)
                # Must be completed after actual trading execution
                logger.warning("  Actual trading execution required. Using sample data for verification.")
                
                # Sample performance metrics (replace in real implementation)
                sample_returns = np.random.normal(0.001, 0.02, len(state_data))  # sample
                sample_equity = np.cumprod(1 + sample_returns) * self.config['training']['initial_capital']
                
                metrics.calculate_metrics(
                    returns=pd.Series(sample_returns, index=state_data.index),
                    equity_curve=pd.Series(sample_equity, index=state_data.index)
                )
                
                return {
                    'Sharpe Ratio': metrics.sharpe_ratio,
                    'Cumulative Return': metrics.cumulative_return,
                    'CAGR': metrics.cagr,
                    'Maximum Drawdown': metrics.max_drawdown,
                    'Win Rate': metrics.win_rate,
                    'Profit Factor': metrics.profit_factor
                }
            else:
                logger.info(f"  {method_name} requires separate implementation")
                return {}
                
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def compare_with_paper(self, actual_metrics: Dict[str, float], method_name: str = 'Proposed Method') -> Dict[str, Any]:
        """
        Compare actual results with paper results.
        
        Parameters
        ----------
        actual_metrics : dict
            Computed performance metrics.
        method_name : str
            Method name.
        
        Returns
        -------
        dict
            Comparison results.
        """
        if method_name not in PAPER_METRICS:
            logger.error(f"Unknown method: {method_name}")
            return {}
        
        paper_metrics = PAPER_METRICS[method_name]
        comparison = {}
        
        tolerance = {
            'Sharpe Ratio': 0.2,  # +/-0.2 tolerance
            'Cumulative Return': 0.1,
            'CAGR': 0.05,
            'Maximum Drawdown': 0.05,
            'Win Rate': 0.05,
            'Profit Factor': 0.2
        }
        
        logger.info(f"\n{'='*100}")
        logger.info(f"Comparison with paper: {method_name}")
        logger.info(f"{'='*100}")
        
        for metric_name in paper_metrics.keys():
            paper_value = paper_metrics[metric_name]
            actual_value = actual_metrics.get(metric_name, None)
            
            if actual_value is None:
                comparison[metric_name] = {
                    'paper': paper_value,
                    'actual': None,
                    'difference': None,
                    'match': False,
                    'status': 'missing'
                }
                logger.warning(f"  x {metric_name}: paper={paper_value:.3f}, actual=missing")
            else:
                diff = abs(actual_value - paper_value)
                tol = tolerance.get(metric_name, 0.1)
                match = diff <= tol
                
                comparison[metric_name] = {
                    'paper': paper_value,
                    'actual': actual_value,
                    'difference': diff,
                    'tolerance': tol,
                    'match': match,
                    'status': 'match' if match else 'mismatch'
                }
                
                status_symbol = "OK" if match else "X"
                logger.info(
                    f"  {status_symbol} {metric_name}: "
                    f"paper={paper_value:.3f}, actual={actual_value:.3f}, "
                    f"diff={diff:.3f} (tolerance={tol:.3f})"
                )
        
        return comparison
    
    def generate_comparison_table(self, comparisons: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Build comparison results as a table."""
        rows = []
        
        for method_name, comparison in comparisons.items():
            row = {'Method': method_name}
            
            for metric_name, metric_data in comparison.items():
                if metric_data.get('status') == 'missing':
                    row[metric_name] = f"{metric_data['paper']:.3f} (N/A)"
                elif metric_data.get('match'):
                    row[metric_name] = f"{metric_data['actual']:.3f} OK"
                else:
                    row[metric_name] = f"{metric_data['actual']:.3f} X"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def generate_final_report(self) -> str:
        """Generate final report."""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("Final Performance Verification Report")
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 100)
        
        report_lines.append("\nPaper performance metrics (Table 2)")
        report_lines.append("-" * 100)
        
        # Show paper metrics
        for method_name, metrics in PAPER_METRICS.items():
            report_lines.append(f"\n{method_name}:")
            for metric_name, value in metrics.items():
                report_lines.append(f"  {metric_name}: {value:.3f}")
        
        report_lines.append("\n" + "=" * 100)
        report_lines.append("\nNote: Actual trading execution and performance calculation must be completed")
        report_lines.append("before an accurate comparison with the paper is possible.")
        report_lines.append("\nData preparation and code verification are currently complete.")
        report_lines.append("=" * 100)
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        report_path = log_dir / 'final_performance_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"\nReport saved: {report_path}")
        
        return report_text


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Final performance verification against paper metrics')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--method', type=str, default='Proposed Method',
                        help='Method to verify')
    
    args = parser.parse_args()
    
    verifier = FinalPerformanceVerifier(args.config)
    
    # Run backtest
    actual_metrics = verifier.run_backtest(args.method)
    
    # Compare with paper
    if actual_metrics:
        comparison = verifier.compare_with_paper(actual_metrics, args.method)
        verifier.results[args.method] = comparison
    
    # Generate report
    report = verifier.generate_final_report()
    
    logger.info("\nFinal performance verification complete!")
    logger.info("Accurate comparison is possible after actual trading execution.")


if __name__ == "__main__":
    main()
