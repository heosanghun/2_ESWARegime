"""
100% 일치 달성을 위한 자동 개선 파이프라인.

단계별로 문제를 해결하고 성능을 개선하여 논문 성과지표와 100% 일치를 달성.
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
    """일치성 퍼센트 계산."""
    if actual_val is None:
        return 0.0
    scale = max(abs(paper_val), 0.01)
    diff = min(1.0, abs(actual_val - paper_val) / scale)
    return round(100.0 * (1.0 - diff), 1)


def analyze_gaps(actual: Dict[str, float]) -> Dict[str, Any]:
    """성능 차이 분석 및 개선 방향 제시."""
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
    """개선 제안 생성."""
    suggestions = []
    
    # Sharpe Ratio, Cumulative Return, CAGR가 모두 음수
    if gaps['Sharpe Ratio']['actual'] < 0:
        suggestions.append({
            'priority': 'CRITICAL',
            'issue': '전략이 손실 발생 (Sharpe Ratio 음수)',
            'action': '1M 학습 완료 대기 또는 하이퍼파라미터 튜닝',
            'params': {
                'ensemble_temperature': [5.0, 10.0, 15.0],
                'reward_scale': [50.0, 100.0, 200.0],
            }
        })
    
    # Win Rate 낮음
    if gaps['Win Rate']['actual'] < 0.45:
        suggestions.append({
            'priority': 'HIGH',
            'issue': 'Win Rate 낮음 (거래 타이밍 부정확)',
            'action': 'Regime 분류 정확도 향상, Confidence Threshold 조정',
            'params': {
                'confidence_threshold': [0.5, 0.6, 0.7],
            }
        })
    
    # MDD 큼
    if gaps['Maximum Drawdown']['actual'] < -0.20:
        suggestions.append({
            'priority': 'HIGH',
            'issue': 'Maximum Drawdown 과다',
            'action': '리스크 관리 강화, 포지션 크기 제한',
            'params': {
                'max_position': [0.75, 0.85, 0.95],
            }
        })
    
    return suggestions


def main():
    """자동 개선 파이프라인 실행."""
    logger.info("=" * 60)
    logger.info("100% 일치 달성을 위한 자동 개선 파이프라인 시작")
    logger.info("=" * 60)
    
    cfg = load_config()
    
    # 현재 성능 확인
    logger.info("Step 1: 현재 성능 확인")
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
    logger.info(f"현재 평균 일치성: {avg_consistency:.1f}%")
    
    # 차이 분석
    logger.info("\nStep 2: 성능 차이 분석")
    gaps = analyze_gaps(actual)
    for k, v in gaps.items():
        logger.info(f"  {k}: 논문={v['paper']:.4f}, 실제={v['actual']:.4f}, 일치성={v['consistency']:.1f}%")
    
    # 개선 제안
    logger.info("\nStep 3: 개선 제안 생성")
    suggestions = suggest_improvements(gaps)
    for i, s in enumerate(suggestions, 1):
        logger.info(f"\n  제안 {i} [{s['priority']}]: {s['issue']}")
        logger.info(f"    조치: {s['action']}")
        logger.info(f"    튜닝 파라미터: {s['params']}")
    
    # 결과 저장
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
    
    logger.info(f"\n분석 결과 저장: {out_dir / 'improvement_analysis.json'}")
    logger.info("\n다음 단계: 제안된 파라미터로 튜닝 후 재평가")


if __name__ == '__main__':
    main()
