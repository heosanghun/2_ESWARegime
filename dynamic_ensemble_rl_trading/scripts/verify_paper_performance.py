"""
Paper performance metrics verification script.

Verifies performance metrics reported in the paper.
Runs actual execution when real data and trained models exist;
otherwise generates a verification report from paper-reported metrics.
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

from src.evaluation.comparison_table import ComparisonTable
from src.evaluation.comprehensive_metrics import ComprehensiveMetrics

# Configure logging
log_dir = Path('results')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_dir / 'verification.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Performance metrics reported in the paper
PAPER_PERFORMANCE = {
    'Proposed Method': {
        'sharpe_ratio': 1.89,
        'cumulative_return': 0.893,  # 89.3%
        'cagr': 0.342,  # 34.2%
        'max_drawdown': -0.162,  # -16.2%
        'win_rate': 0.678,  # 67.8%
        'profit_factor': None  # not specified in paper
    },
    'Model 1 (No Dynamic Weighting)': {
        'sharpe_ratio': 1.58,
        'cumulative_return': None,
        'cagr': None,
        'max_drawdown': None,
        'win_rate': None,
        'profit_factor': None
    },
    'Model 2 (No Confidence Selection)': {
        'sharpe_ratio': 1.41,
        'cumulative_return': None,
        'cagr': None,
        'max_drawdown': None,
        'win_rate': None,
        'profit_factor': None
    },
    'Model 3 (No Ensemble)': {
        'sharpe_ratio': 1.41,
        'cumulative_return': None,
        'cagr': None,
        'max_drawdown': None,
        'win_rate': None,
        'profit_factor': None
    },
    'Model 4 (No Regime Classification)': {
        'sharpe_ratio': 1.35,
        'cumulative_return': None,
        'cagr': None,
        'max_drawdown': None,
        'win_rate': None,
        'profit_factor': None
    },
    'Buy & Hold': {
        'sharpe_ratio': None,  # not specified in paper
        'cumulative_return': None,
        'cagr': None,
        'max_drawdown': None,
        'win_rate': None,
        'profit_factor': None
    }
}

# Tolerance
TOLERANCE = {
    'sharpe_ratio': 0.1,
    'cumulative_return': 0.05,  # 5%
    'cagr': 0.02,  # 2%
    'max_drawdown': 0.02,  # 2%
    'win_rate': 0.03  # 3%
}


class PaperPerformanceVerifier:
    """
    Paper performance metrics verifier.
    
    Compares actual run results with paper-reported performance.
    """
    
    def __init__(self):
        """Initialize verifier."""
        self.comparison_table = ComparisonTable()
        self.verification_results = {}
        logger.info("Initialized Paper Performance Verifier")
    
    def verify_method(
        self,
        method_name: str,
        actual_metrics: Dict[str, float],
        paper_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Verify performance for a specific method.
        
        Parameters
        ----------
        method_name : str
            Method name.
        actual_metrics : dict
            Measured performance metrics.
        paper_metrics : dict, optional
            Performance metrics reported in the paper.
        
        Returns
        -------
        dict
            Verification results.
        """
        if paper_metrics is None:
            paper_metrics = PAPER_PERFORMANCE.get(method_name, {})
        
        verification = {
            'method': method_name,
            'actual': actual_metrics,
            'paper': paper_metrics,
            'comparison': {},
            'within_tolerance': {},
            'all_within_tolerance': True
        }
        
        for metric_name in actual_metrics.keys():
            if metric_name in paper_metrics and paper_metrics[metric_name] is not None:
                actual_val = actual_metrics[metric_name]
                paper_val = paper_metrics[metric_name]
                
                # Absolute difference
                diff = actual_val - paper_val
                
                # Relative difference (%)
                if paper_val != 0:
                    diff_pct = (diff / abs(paper_val)) * 100
                else:
                    diff_pct = 0.0
                
                # Check tolerance
                tolerance = TOLERANCE.get(metric_name, 0.1)
                if metric_name in ['cumulative_return', 'cagr', 'max_drawdown', 'win_rate']:
                    # Percentage metrics use relative difference
                    within_tol = abs(diff_pct) < (tolerance * 100)
                else:
                    # Absolute difference for others
                    within_tol = abs(diff) < tolerance
                
                verification['comparison'][metric_name] = {
                    'actual': actual_val,
                    'paper': paper_val,
                    'difference': diff,
                    'difference_pct': diff_pct,
                    'within_tolerance': within_tol
                }
                
                verification['within_tolerance'][metric_name] = within_tol
                
                if not within_tol:
                    verification['all_within_tolerance'] = False
        
        self.verification_results[method_name] = verification
        return verification
    
    def generate_verification_report(self) -> str:
        """Generate verification report."""
        report = []
        report.append("=" * 100)
        report.append("Paper Performance Metrics Verification Report")
        report.append("=" * 100)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Verification results per method
        for method_name, verification in self.verification_results.items():
            report.append("-" * 100)
            report.append(f"Method: {method_name}")
            report.append("-" * 100)
            
            if verification['all_within_tolerance']:
                report.append("All metrics are within tolerance.")
            else:
                report.append("Some metrics exceed tolerance.")
            
            report.append("")
            report.append("Performance comparison:")
            report.append("")
            
            for metric_name, comp in verification['comparison'].items():
                status = "OK" if comp['within_tolerance'] else "FAIL"
                report.append(
                    f"  {status} {metric_name.upper()}:"
                )
                report.append(
                    f"    actual: {comp['actual']:.4f}, "
                    f"paper: {comp['paper']:.4f}, "
                    f"diff: {comp['difference']:+.4f} ({comp['difference_pct']:+.2f}%)"
                )
            
            report.append("")
        
        # Overall summary
        report.append("=" * 100)
        report.append("Overall Summary")
        report.append("=" * 100)
        
        total_methods = len(self.verification_results)
        passed_methods = sum(1 for v in self.verification_results.values() if v['all_within_tolerance'])
        
        report.append(f"Total methods: {total_methods}")
        report.append(f"Methods within tolerance: {passed_methods}")
        report.append(f"Methods exceeding tolerance: {total_methods - passed_methods}")
        report.append("")
        
        # Tolerance criteria
        report.append("Tolerance criteria:")
        for metric_name, tolerance in TOLERANCE.items():
            if metric_name in ['cumulative_return', 'cagr', 'max_drawdown', 'win_rate']:
                report.append(f"  {metric_name}: +/-{tolerance*100}%")
            else:
                report.append(f"  {metric_name}: +/-{tolerance}")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_verification_report(self, output_path: str) -> None:
        """Save verification report."""
        report = self.generate_verification_report()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Verification report saved to {output_path}")
        print("\n" + report)


def load_actual_results(results_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load actual run results.
    
    Parameters
    ----------
    results_file : str
        Results file path (pickle or JSON).
    
    Returns
    -------
    dict
        Performance metrics by method.
    """
    results_path = Path(results_file)
    
    if not results_path.exists():
        logger.warning(f"Results file not found: {results_file}")
        return {}
    
    if results_path.suffix == '.pkl':
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract metrics from results
        actual_results = {}
        for method_name, result_data in results.items():
            if 'metrics' in result_data:
                actual_results[method_name] = result_data['metrics']
        
        return actual_results
    else:
        logger.error(f"Unsupported file format: {results_path.suffix}")
        return {}


def create_paper_based_comparison_table() -> ComparisonTable:
    """
    Create comparison table from paper-reported performance metrics.
    
    Returns
    -------
    ComparisonTable
        Comparison table based on paper performance metrics.
    """
    table = ComparisonTable()
    
    for method_name, metrics in PAPER_PERFORMANCE.items():
        # Remove None values
        filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
        if filtered_metrics:
            table.add_result(method_name, filtered_metrics)
    
    return table


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify paper performance metrics')
    parser.add_argument(
        '--actual-results',
        type=str,
        default=None,
        help='Path to actual results file (pickle)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/verification',
        help='Output directory for verification reports'
    )
    parser.add_argument(
        '--paper-only',
        action='store_true',
        help='Generate comparison table from paper metrics only'
    )
    
    args = parser.parse_args()
    
    verifier = PaperPerformanceVerifier()
    
    if args.paper_only or args.actual_results is None:
        # Create comparison table from paper metrics only
        logger.info("Generating comparison table from paper metrics...")
        table = create_paper_based_comparison_table()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        table.save_csv(str(output_dir / 'paper_performance_table.csv'))
        table.save_markdown(str(output_dir / 'paper_performance_table.md'))
        table.save_latex(str(output_dir / 'paper_performance_table.tex'))
        table.print_table()
        
        logger.info("Paper performance comparison table generated.")
        
        # Generate verification report (paper metrics only)
        report_path = output_dir / 'paper_performance_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("Paper Performance Metrics Summary\n")
            f.write("=" * 100 + "\n\n")
            f.write("This report summarizes performance metrics reported in the paper.\n")
            f.write("Use --actual-results to compare with actual run results.\n\n")
            
            for method_name, metrics in PAPER_PERFORMANCE.items():
                f.write(f"{method_name}:\n")
                for metric_name, value in metrics.items():
                    if value is not None:
                        if metric_name in ['cumulative_return', 'cagr', 'win_rate']:
                            f.write(f"  {metric_name}: {value*100:.2f}%\n")
                        elif metric_name == 'max_drawdown':
                            f.write(f"  {metric_name}: {value*100:.2f}%\n")
                        else:
                            f.write(f"  {metric_name}: {value:.4f}\n")
                f.write("\n")
        
        logger.info(f"Paper performance report saved to {report_path}")
    
    if args.actual_results:
        # Compare actual results with paper performance
        logger.info(f"Loading actual results from {args.actual_results}...")
        actual_results = load_actual_results(args.actual_results)
        
        if not actual_results:
            logger.warning("No actual results found. Generating paper-only report.")
            return
        
        # Verify each method
        for method_name, actual_metrics in actual_results.items():
            paper_metrics = PAPER_PERFORMANCE.get(method_name)
            verifier.verify_method(method_name, actual_metrics, paper_metrics)
        
        # Generate and save verification report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'verification_report.txt'
        verifier.save_verification_report(str(report_path))
        
        # Also add to comparison table
        for method_name, actual_metrics in actual_results.items():
            verifier.comparison_table.add_result(method_name, actual_metrics)
        
        verifier.comparison_table.save_csv(str(output_dir / 'actual_vs_paper_comparison.csv'))
        verifier.comparison_table.save_markdown(str(output_dir / 'actual_vs_paper_comparison.md'))
        verifier.comparison_table.print_table()
        
        logger.info("Verification completed!")


if __name__ == "__main__":
    main()
