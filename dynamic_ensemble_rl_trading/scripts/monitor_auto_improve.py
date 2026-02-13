"""
자동 개선 루프 진행 상황 모니터링 스크립트.
"""

import json
from pathlib import Path
from datetime import datetime
import time

def monitor_improvement():
    """개선 진행 상황 모니터링."""
    log_path = Path('results/verification/auto_improve_loop.log')
    history_path = Path('results/verification/improvement_history.json')
    
    print("=" * 80)
    print("자동 개선 루프 모니터링")
    print("=" * 80)
    
    # 히스토리 확인
    if history_path.exists():
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if history:
            latest = history[-1]
            print(f"\n최신 반복: {latest['iteration']}")
            print(f"현재 일치성: {latest['consistency']:.1f}%")
            print(f"개선 여부: {'✅ 개선됨' if latest.get('improved', False) else '⚠️ 개선 없음'}")
            print(f"\n최신 지표:")
            for k, v in latest['metrics'].items():
                print(f"  {k}: {v:.4f}")
            
            # 전체 진행 상황
            print(f"\n전체 진행 상황:")
            consistencies = [h['consistency'] for h in history]
            best_idx = consistencies.index(max(consistencies))
            print(f"  최고 일치성: {max(consistencies):.1f}% (반복 {history[best_idx]['iteration']})")
            print(f"  평균 일치성: {sum(consistencies)/len(consistencies):.1f}%")
            print(f"  총 반복 횟수: {len(history)}")
        else:
            print("\n히스토리가 비어있습니다.")
    else:
        print("\n히스토리 파일이 없습니다. 아직 시작되지 않았을 수 있습니다.")
    
    # 로그 파일 확인
    if log_path.exists():
        print(f"\n로그 파일: {log_path}")
        print("최근 로그 (마지막 10줄):")
        print("-" * 80)
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())
    else:
        print("\n로그 파일이 없습니다.")
    
    print("\n" + "=" * 80)
    print("모니터링 완료")
    print("=" * 80)


if __name__ == '__main__':
    monitor_improvement()
