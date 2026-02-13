"""
논문 성능 지표 검증 스크립트.

이 스크립트는 논문에서 보고한 성능 지표를 검증합니다.
실제 데이터와 학습된 모델이 있으면 실제 실행하고,
없으면 논문의 보고된 성능 지표를 기반으로 검증 리포트를 생성합니다.
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


# 논문에서 보고한 성능 지표
PAPER_PERFORMANCE = {
    'Proposed Method': {
        'sharpe_ratio': 1.89,
        'cumulative_return': 0.893,  # 89.3%
        'cagr': 0.342,  # 34.2%
        'max_drawdown': -0.162,  # -16.2%
        'win_rate': 0.678,  # 67.8%
        'profit_factor': None  # 논문에서 명시되지 않음
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
        'sharpe_ratio': None,  # 논문에서 명시되지 않음
        'cumulative_return': None,
        'cagr': None,
        'max_drawdown': None,
        'win_rate': None,
        'profit_factor': None
    }
}

# 허용 오차
TOLERANCE = {
    'sharpe_ratio': 0.1,
    'cumulative_return': 0.05,  # 5%
    'cagr': 0.02,  # 2%
    'max_drawdown': 0.02,  # 2%
    'win_rate': 0.03  # 3%
}


class PaperPerformanceVerifier:
    """
    논문 성능 지표 검증기.
    
    실제 실행 결과와 논문의 보고된 성능을 비교합니다.
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
        특정 방법론의 성능을 검증.
        
        Parameters
        ----------
        method_name : str
            방법론 이름.
        actual_metrics : dict
            실제 측정된 성능 지표.
        paper_metrics : dict, optional
            논문에서 보고한 성능 지표.
        
        Returns
        -------
        dict
            검증 결과.
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
                
                # 절대 차이
                diff = actual_val - paper_val
                
                # 상대 차이 (%)
                if paper_val != 0:
                    diff_pct = (diff / abs(paper_val)) * 100
                else:
                    diff_pct = 0.0
                
                # 허용 오차 확인
                tolerance = TOLERANCE.get(metric_name, 0.1)
                if metric_name in ['cumulative_return', 'cagr', 'max_drawdown', 'win_rate']:
                    # 백분율 지표는 상대 차이로 확인
                    within_tol = abs(diff_pct) < (tolerance * 100)
                else:
                    # 절대 차이로 확인
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
        """검증 리포트 생성."""
        report = []
        report.append("=" * 100)
        report.append("논문 성능 지표 검증 리포트")
        report.append("=" * 100)
        report.append(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 각 방법론별 검증 결과
        for method_name, verification in self.verification_results.items():
            report.append("-" * 100)
            report.append(f"방법론: {method_name}")
            report.append("-" * 100)
            
            if verification['all_within_tolerance']:
                report.append("✅ 모든 지표가 허용 오차 내에 있습니다.")
            else:
                report.append("⚠️ 일부 지표가 허용 오차를 벗어났습니다.")
            
            report.append("")
            report.append("성능 지표 비교:")
            report.append("")
            
            for metric_name, comp in verification['comparison'].items():
                status = "✅" if comp['within_tolerance'] else "❌"
                report.append(
                    f"  {status} {metric_name.upper()}:"
                )
                report.append(
                    f"    실제: {comp['actual']:.4f}, "
                    f"논문: {comp['paper']:.4f}, "
                    f"차이: {comp['difference']:+.4f} ({comp['difference_pct']:+.2f}%)"
                )
            
            report.append("")
        
        # 종합 요약
        report.append("=" * 100)
        report.append("종합 요약")
        report.append("=" * 100)
        
        total_methods = len(self.verification_results)
        passed_methods = sum(1 for v in self.verification_results.values() if v['all_within_tolerance'])
        
        report.append(f"총 방법론 수: {total_methods}")
        report.append(f"허용 오차 내 방법론: {passed_methods}")
        report.append(f"허용 오차 초과 방법론: {total_methods - passed_methods}")
        report.append("")
        
        # 허용 오차 기준
        report.append("허용 오차 기준:")
        for metric_name, tolerance in TOLERANCE.items():
            if metric_name in ['cumulative_return', 'cagr', 'max_drawdown', 'win_rate']:
                report.append(f"  {metric_name}: ±{tolerance*100}%")
            else:
                report.append(f"  {metric_name}: ±{tolerance}")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_verification_report(self, output_path: str) -> None:
        """검증 리포트 저장."""
        report = self.generate_verification_report()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Verification report saved to {output_path}")
        print("\n" + report)


def load_actual_results(results_file: str) -> Dict[str, Dict[str, float]]:
    """
    실제 실행 결과 로드.
    
    Parameters
    ----------
    results_file : str
        결과 파일 경로 (pickle 또는 JSON).
    
    Returns
    -------
    dict
        방법론별 성능 지표.
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
    논문의 보고된 성능 지표로 비교 테이블 생성.
    
    Returns
    -------
    ComparisonTable
        논문 성능 지표 기반 비교 테이블.
    """
    table = ComparisonTable()
    
    for method_name, metrics in PAPER_PERFORMANCE.items():
        # None 값 제거
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
        # 논문 성능 지표만으로 비교 테이블 생성
        logger.info("Generating comparison table from paper metrics...")
        table = create_paper_based_comparison_table()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        table.save_csv(str(output_dir / 'paper_performance_table.csv'))
        table.save_markdown(str(output_dir / 'paper_performance_table.md'))
        table.save_latex(str(output_dir / 'paper_performance_table.tex'))
        table.print_table()
        
        logger.info("Paper performance comparison table generated.")
        
        # 검증 리포트 생성 (논문 성능만)
        report_path = output_dir / 'paper_performance_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("논문 성능 지표 요약\n")
            f.write("=" * 100 + "\n\n")
            f.write("이 리포트는 논문에서 보고한 성능 지표를 요약합니다.\n")
            f.write("실제 실행 결과와 비교하려면 --actual-results 옵션을 사용하세요.\n\n")
            
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
        # 실제 결과와 논문 성능 비교
        logger.info(f"Loading actual results from {args.actual_results}...")
        actual_results = load_actual_results(args.actual_results)
        
        if not actual_results:
            logger.warning("No actual results found. Generating paper-only report.")
            return
        
        # 각 방법론 검증
        for method_name, actual_metrics in actual_results.items():
            paper_metrics = PAPER_PERFORMANCE.get(method_name)
            verifier.verify_method(method_name, actual_metrics, paper_metrics)
        
        # 검증 리포트 생성 및 저장
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'verification_report.txt'
        verifier.save_verification_report(str(report_path))
        
        # 비교 테이블에도 추가
        for method_name, actual_metrics in actual_results.items():
            verifier.comparison_table.add_result(method_name, actual_metrics)
        
        verifier.comparison_table.save_csv(str(output_dir / 'actual_vs_paper_comparison.csv'))
        verifier.comparison_table.save_markdown(str(output_dir / 'actual_vs_paper_comparison.md'))
        verifier.comparison_table.print_table()
        
        logger.info("Verification completed!")


if __name__ == "__main__":
    main()
