"""
Automatic improvement pipeline to reach 100% consistency.

Resolves issues step by step and improves performance to match paper metrics at 100%.
"""

import sys
from pathlib import Path
import yaml
import json
import logging
import numpy as np
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_and_verify import (
    load_config, step3_backtest, step4_compare
)
from src.backtest.backtester import Backtester
from src.backtest.metrics import PerformanceMetrics

Path('results/verification').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('results/verification/improve_to_100.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PAPER_METRICS = {
    'Sharpe Ratio': 2.45,
    'Cumulative Return': 1.23,
    'CAGR': 0.41,
    'Maximum Drawdown': -0.15,
    'Win Rate': 0.58,
    'Profit Factor': 2.1,
}


def calculate_consistency(paper_val: float, actual_val: float) -> float:
    """Calculate consistency percentage."""
    if actual_val is None:
        return 0.0
    scale = max(abs(paper_val), 0.01)
    diff = min(1.0, abs(actual_val - paper_val) / scale)
    return round(100.0 * (1.0 - diff), 1)


def analyze_gaps(actual: Dict[str, float]) -> Dict[str, Any]:
    """Analyze performance gaps and suggest improvements."""
    gaps = {}
    for k, paper_val in PAPER_METRICS.items():
        act = actual.get(k, 0.0)
        diff = act - paper_val
        gaps[k] = {
            'paper': paper_val,
            'actual': act,
            'diff': diff,
            'consistency': calculate_consistency(paper_val, act),
            'priority': 'HIGH' if abs(diff) > abs(paper_val) * 0.5 else 'MEDIUM' if abs(diff) > abs(paper_val) * 0.2 else 'LOW',
        }
    return gaps


def suggest_improvements(gaps: Dict[str, Any]) -> list:
    """Generate improvement suggestions."""
    suggestions = []
    
    # Sharpe Ratio, Cumulative Return, CAGR all negative
    if gaps['Sharpe Ratio']['actual'] < 0:
        suggestions.append({
            'priority': 'CRITICAL',
            'issue': 'Strategy is losing (negative Sharpe Ratio)',
            'action': 'Wait for 1M training to finish or tune hyperparameters',
            'params': {
                'ensemble_temperature': [5.0, 10.0, 15.0],
                'reward_scale': [50.0, 100.0, 200.0],
            }
        })
    
    # Low win rate
    if gaps['Win Rate']['actual'] < 0.45:
        suggestions.append({
            'priority': 'HIGH',
            'issue': 'Low win rate (inaccurate trade timing)',
            'action': 'Improve regime classification accuracy, adjust confidence threshold',
            'params': {
                'confidence_threshold': [0.5, 0.6, 0.7],
            }
        })
    
    # Large MDD
    if gaps['Maximum Drawdown']['actual'] < -0.20:
        suggestions.append({
            'priority': 'HIGH',
            'issue': 'Excessive maximum drawdown',
            'action': 'Strengthen risk management, limit position size',
            'params': {
                'max_position': [0.75, 0.85, 0.95],
            }
        })
    
    return suggestions


def main():
    """Run automatic improvement pipeline."""
    logger.info("=" * 60)
    logger.info("Starting automatic improvement pipeline for 100% consistency")
    logger.info("=" * 60)
    
    cfg = load_config()
    
    # Check current performance
    logger.info("Step 1: Check current performance")
    results = step3_backtest(cfg)
    metrics = results['metrics']
    
    actual = {
        'Sharpe Ratio': metrics['sharpe_ratio'],
        'Cumulative Return': metrics['cumulative_return'],
        'CAGR': metrics['cagr'],
        'Maximum Drawdown': metrics['max_drawdown'],
        'Win Rate': metrics['win_rate'],
        'Profit Factor': metrics['profit_factor'],
    }
    
    avg_consistency, _ = step4_compare(results)
    logger.info(f"Current average consistency: {avg_consistency:.1f}%")
    
    # Gap analysis
    logger.info("\nStep 2: Performance gap analysis")
    gaps = analyze_gaps(actual)
    for k, v in gaps.items():
        logger.info(f"  {k}: paper={v['paper']:.4f}, actual={v['actual']:.4f}, consistency={v['consistency']:.1f}%")
    
    # Improvement suggestions
    logger.info("\nStep 3: Generate improvement suggestions")
    suggestions = suggest_improvements(gaps)
    for i, s in enumerate(suggestions, 1):
        logger.info(f"\n  Suggestion {i} [{s['priority']}]: {s['issue']}")
        logger.info(f"    Action: {s['action']}")
        logger.info(f"    Tuning parameters: {s['params']}")
    
    # Save results
    out_dir = Path('results/verification')
    report = {
        'current_metrics': actual,
        'paper_metrics': PAPER_METRICS,
        'gaps': gaps,
        'avg_consistency': avg_consistency,
        'suggestions': suggestions,
    }
    
    with open(out_dir / 'improvement_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nAnalysis saved: {out_dir / 'improvement_analysis.json'}")
    logger.info("\nNext step: tune with suggested parameters and re-evaluate")


if __name__ == '__main__':
    main()
