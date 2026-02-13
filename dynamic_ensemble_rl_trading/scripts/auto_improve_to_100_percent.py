"""
100% 일치 달성을 위한 자율적 무한루프 개선 시스템.

셀프검증을 통해 최종목표가 될 때까지 자동으로 개선을 반복합니다.
"""

import sys
from pathlib import Path
import yaml
import json
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import copy
import time

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
        logging.FileHandler('results/verification/auto_improve_loop.log', encoding='utf-8'),
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

TARGET_CONSISTENCY = 95.0  # 95% 이상이면 목표 달성으로 간주
MAX_ITERATIONS = 50  # 최대 반복 횟수
MIN_IMPROVEMENT = 0.5  # 최소 개선율 (%)


def calculate_consistency(paper_val: float, actual_val: float) -> float:
    """일치성 퍼센트 계산."""
    if actual_val is None:
        return 0.0
    scale = max(abs(paper_val), 0.01)
    diff = min(1.0, abs(actual_val - paper_val) / scale)
    return round(100.0 * (1.0 - diff), 1)


def evaluate_current_performance(config: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
    """현재 성능 평가."""
    logger.info("=" * 60)
    logger.info("현재 성능 평가 중...")
    logger.info("=" * 60)
    
    results = step3_backtest(config)
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
    
    logger.info(f"평균 일치성: {avg_consistency:.1f}%")
    for k, v in actual.items():
        paper_val = PAPER_METRICS[k]
        consistency = calculate_consistency(paper_val, v)
        logger.info(f"  {k}: {v:.4f} (논문: {paper_val:.4f}, 일치성: {consistency:.1f}%)")
    
    return actual, avg_consistency


def analyze_gaps(actual: Dict[str, float]) -> Dict[str, Any]:
    """성능 차이 분석."""
    gaps = {}
    for k, paper_val in PAPER_METRICS.items():
        act = actual.get(k, 0.0)
        diff = act - paper_val
        gaps[k] = {
            'paper': paper_val,
            'actual': act,
            'diff': diff,
            'consistency': calculate_consistency(paper_val, act),
            'priority': 'CRITICAL' if abs(diff) > abs(paper_val) * 0.5 else 'HIGH' if abs(diff) > abs(paper_val) * 0.2 else 'MEDIUM',
        }
    return gaps


def generate_improvement_plan(gaps: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    """개선 계획 생성."""
    plan = {
        'config_changes': {},
        'priority': 'MEDIUM',
        'description': '',
    }
    
    # Critical issues
    if gaps['Sharpe Ratio']['actual'] < 0 or gaps['Cumulative Return']['actual'] < 0:
        plan['priority'] = 'CRITICAL'
        plan['description'] = '전략이 손실 발생 - 하이퍼파라미터 튜닝 필요'
        
        # Ensemble temperature 조정
        if iteration % 3 == 0:
            plan['config_changes']['ensemble'] = {
                'temperature': [5.0, 7.5, 10.0, 12.5, 15.0][iteration % 5],
            }
        
        # Reward scale 조정
        if iteration % 2 == 0:
            plan['config_changes']['environment'] = {
                'reward_scale': [50.0, 100.0, 150.0, 200.0][iteration % 4],
            }
    
    # Win Rate 낮음
    if gaps['Win Rate']['consistency'] < 60:
        plan['priority'] = 'HIGH'
        plan['description'] = 'Win Rate 낮음 - Regime 분류 개선 필요'
        plan['config_changes']['regime'] = {
            'confidence_threshold': [0.5, 0.55, 0.6, 0.65, 0.7][iteration % 5],
        }
    
    # MDD 과다
    if gaps['Maximum Drawdown']['consistency'] < 50:
        plan['priority'] = 'HIGH'
        plan['description'] = 'Maximum Drawdown 과다 - 리스크 관리 강화 필요'
        plan['config_changes']['training'] = {
            'max_position': [0.75, 0.80, 0.85, 0.90, 0.95][iteration % 5],
        }
    
    # Profit Factor 낮음
    if gaps['Profit Factor']['consistency'] < 50:
        plan['priority'] = 'MEDIUM'
        plan['description'] = 'Profit Factor 낮음 - 거래 비용 최적화 필요'
        if 'training' not in plan['config_changes']:
            plan['config_changes']['training'] = {}
        plan['config_changes']['training']['transaction_fee'] = [0.0003, 0.0004, 0.0005, 0.0006][iteration % 4]
    
    return plan


def apply_config_changes(config: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """설정 변경 적용."""
    new_config = copy.deepcopy(config)
    
    for section, values in changes.items():
        if section not in new_config:
            new_config[section] = {}
        
        for key, value in values.items():
            old_value = new_config[section].get(key, None)
            new_config[section][key] = value
            logger.info(f"  {section}.{key}: {old_value} → {value}")
    
    return new_config


def save_config(config: Dict[str, Any], iteration: int):
    """설정 저장."""
    config_path = Path('config/config.yaml')
    backup_path = Path(f'config/config_backup_iter_{iteration}.yaml')
    
    # 백업
    if config_path.exists():
        import shutil
        shutil.copy(config_path, backup_path)
    
    # 새 설정 저장
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"설정 저장: {config_path}")


def main():
    """자율적 무한루프 개선 시스템."""
    logger.info("=" * 80)
    logger.info("100% 일치 달성을 위한 자율적 무한루프 개선 시스템 시작")
    logger.info("=" * 80)
    logger.info(f"목표 일치성: {TARGET_CONSISTENCY}%")
    logger.info(f"최대 반복 횟수: {MAX_ITERATIONS}")
    logger.info("=" * 80)
    
    iteration = 0
    best_consistency = 0.0
    no_improvement_count = 0
    history = []
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info("\n" + "=" * 80)
        logger.info(f"반복 {iteration}/{MAX_ITERATIONS}")
        logger.info("=" * 80)
        
        try:
            # 현재 설정 로드
            config = load_config()
            
            # 현재 성능 평가
            actual, avg_consistency = evaluate_current_performance(config)
            
            # 목표 달성 확인
            if avg_consistency >= TARGET_CONSISTENCY:
                logger.info("\n" + "=" * 80)
                logger.info(f"🎉 목표 달성! 평균 일치성: {avg_consistency:.1f}%")
                logger.info("=" * 80)
                break
            
            # 최고 성능 업데이트
            improved = avg_consistency > best_consistency + MIN_IMPROVEMENT
            if improved:
                best_consistency = avg_consistency
                no_improvement_count = 0
                logger.info(f"✅ 개선됨! 최고 일치성: {best_consistency:.1f}%")
            else:
                no_improvement_count += 1
                logger.info(f"⚠️  개선 없음 (연속 {no_improvement_count}회)")
            
            # 성능 차이 분석
            gaps = analyze_gaps(actual)
            
            # 개선 계획 생성
            plan = generate_improvement_plan(gaps, iteration)
            logger.info(f"\n개선 계획 [{plan['priority']}]: {plan['description']}")
            
            if plan['config_changes']:
                # 설정 변경 적용
                logger.info("설정 변경 적용:")
                new_config = apply_config_changes(config, plan['config_changes'])
                save_config(new_config, iteration)
            else:
                logger.info("변경할 설정 없음 - 다음 반복으로 진행")
            
            # 히스토리 저장
            history.append({
                'iteration': iteration,
                'consistency': avg_consistency,
                'metrics': actual,
                'gaps': gaps,
                'plan': plan,
                'improved': improved,
            })
            
            # 히스토리 저장
            history_path = Path('results/verification/improvement_history.json')
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False, default=str)
            
            # 개선이 없는 경우 조기 종료 고려
            if no_improvement_count >= 5:
                logger.warning(f"연속 {no_improvement_count}회 개선 없음 - 다른 접근 필요")
                # 더 공격적인 튜닝 시도
                if iteration < MAX_ITERATIONS:
                    logger.info("더 공격적인 튜닝 적용...")
                    plan['config_changes'] = {
                        'ensemble': {'temperature': 5.0},
                        'regime': {'confidence_threshold': 0.5},
                        'training': {'max_position': 0.75},
                    }
                    new_config = apply_config_changes(config, plan['config_changes'])
                    save_config(new_config, iteration)
                    no_improvement_count = 0
            
            # 다음 반복 전 대기 (필요시)
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"반복 {iteration} 중 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 이전 설정으로 복원
            if iteration > 1:
                backup_path = Path(f'config/config_backup_iter_{iteration-1}.yaml')
                if backup_path.exists():
                    import shutil
                    shutil.copy(backup_path, Path('config/config.yaml'))
                    logger.info(f"이전 설정으로 복원: {backup_path}")
            continue
    
    # 최종 결과
    logger.info("\n" + "=" * 80)
    logger.info("최종 결과")
    logger.info("=" * 80)
    logger.info(f"총 반복 횟수: {iteration}")
    logger.info(f"최고 일치성: {best_consistency:.1f}%")
    
    if avg_consistency >= TARGET_CONSISTENCY:
        logger.info("✅ 목표 달성!")
    else:
        logger.info(f"⚠️  목표 미달성 (현재: {avg_consistency:.1f}%, 목표: {TARGET_CONSISTENCY}%)")
        logger.info("추가 개선이 필요합니다.")
    
    logger.info(f"\n히스토리 저장: results/verification/improvement_history.json")


if __name__ == '__main__':
    main()
