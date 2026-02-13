"""
전체 파이프라인 실행 및 논문 성능 검증.

순서:
1. 전체 시스템 백테스트 (main.py 로직 인라인)
2. 성능 지표 계산 (Sharpe, CumRet, CAGR, MDD, Win Rate, Profit Factor)
3. 논문 Table 2와 비교 검증
4. 추가 단계: ablation/baseline 검토 및 리포트 생성
"""

import sys
from pathlib import Path
import yaml
import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
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
from src.ensemble.weighting import DynamicWeightCalculator
from src.ensemble.ensemble_trader import EnsembleTrader
from src.backtest.backtester import Backtester
from src.backtest.metrics import PerformanceMetrics
from src.utils.seed import set_seed

Path('results/verification').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/verification/full_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 논문 Table 2 (Proposed Method 목표)
PAPER_TABLE2 = {
    'Proposed Method': {'Sharpe Ratio': 2.45, 'Cumulative Return': 1.23, 'CAGR': 0.41,
                       'Maximum Drawdown': -0.15, 'Win Rate': 0.58, 'Profit Factor': 2.1},
    'No Dynamic Weighting': {'Sharpe Ratio': 2.12, 'Cumulative Return': 1.08, 'CAGR': 0.36,
                             'Maximum Drawdown': -0.18, 'Win Rate': 0.55, 'Profit Factor': 1.9},
    'No Confidence Selection': {'Sharpe Ratio': 1.98, 'Cumulative Return': 0.95, 'CAGR': 0.32,
                                'Maximum Drawdown': -0.22, 'Win Rate': 0.52, 'Profit Factor': 1.7},
    'No Ensemble': {'Sharpe Ratio': 1.65, 'Cumulative Return': 0.78, 'CAGR': 0.26,
                    'Maximum Drawdown': -0.28, 'Win Rate': 0.48, 'Profit Factor': 1.5},
    'No Regime Classification': {'Sharpe Ratio': 1.42, 'Cumulative Return': 0.65, 'CAGR': 0.22,
                                 'Maximum Drawdown': -0.32, 'Win Rate': 0.45, 'Profit Factor': 1.3},
    'Single PPO Agent': {'Sharpe Ratio': 1.28, 'Cumulative Return': 0.58, 'CAGR': 0.19,
                         'Maximum Drawdown': -0.35, 'Win Rate': 0.42, 'Profit Factor': 1.2},
    'XGBoost Trader': {'Sharpe Ratio': 0.95, 'Cumulative Return': 0.42, 'CAGR': 0.14,
                       'Maximum Drawdown': -0.42, 'Win Rate': 0.38, 'Profit Factor': 1.1},
    'CNN Trader': {'Sharpe Ratio': 0.78, 'Cumulative Return': 0.35, 'CAGR': 0.12,
                   'Maximum Drawdown': -0.48, 'Win Rate': 0.35, 'Profit Factor': 1.05},
    'Simple Ensemble': {'Sharpe Ratio': 1.15, 'Cumulative Return': 0.52, 'CAGR': 0.17,
                        'Maximum Drawdown': -0.38, 'Win Rate': 0.40, 'Profit Factor': 1.15},
}


def load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    hp_path = config_path.parent / 'hyperparameters.yaml'
    if hp_path.exists():
        with open(hp_path, 'r', encoding='utf-8') as f:
            config['hyperparameters'] = yaml.safe_load(f)
    return config


def run_backtest_and_get_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    """백테스트 실행 후 성능 지표 반환."""
    set_seed(config.get('random_seed', 42))

    logger.info("Loading data and building states...")
    data_handler = MarketDataHandler(config['data']['ohlcv_path'])
    ohlcv_data = data_handler.load_data(
        start_date=config['training']['test_start_date'],
        end_date=config['training']['test_end_date']
    )
    tech_extractor = TechnicalFeatureExtractor()
    visual_extractor = CandlestickGenerator()
    sentiment_extractor = NewsSentimentExtractor(config['data']['news_path'])
    sentiment_extractor.load_news_data(
        start_date=config['training']['test_start_date'],
        end_date=config['training']['test_end_date']
    )
    feature_fusion = FeatureFusion(tech_extractor, visual_extractor, sentiment_extractor)
    state_data = feature_fusion.batch_create_unified_states(ohlcv_data, ohlcv_data.index)

    regime_classifier = RegimeClassifier(
        n_estimators=config['hyperparameters']['regime_classifier']['n_estimators'],
        max_depth=config['hyperparameters']['regime_classifier']['max_depth'],
        confidence_threshold=config['regime']['confidence_threshold']
    )
    regime_path = Path(config['models']['regime_classifier']) / 'model.json'
    if not regime_path.exists():
        raise FileNotFoundError(f"Regime classifier not found: {regime_path}")
    regime_classifier.load_model(str(regime_path))

    bull_env = MultiRegimeTradingEnv(
        ohlcv_data=ohlcv_data, state_data=state_data, regime_type='Bull',
        initial_balance=config['training']['initial_capital'],
        transaction_fee=config['training']['transaction_fee'],
        slippage=config['training']['slippage']
    )
    bear_env = MultiRegimeTradingEnv(
        ohlcv_data=ohlcv_data, state_data=state_data, regime_type='Bear',
        initial_balance=config['training']['initial_capital'],
        transaction_fee=config['training']['transaction_fee'],
        slippage=config['training']['slippage']
    )
    sideways_env = MultiRegimeTradingEnv(
        ohlcv_data=ohlcv_data, state_data=state_data, regime_type='Sideways',
        initial_balance=config['training']['initial_capital'],
        transaction_fee=config['training']['transaction_fee'],
        slippage=config['training']['slippage']
    )

    agent_manager = HierarchicalAgentManager(
        bull_env=bull_env, bear_env=bear_env, sideways_env=sideways_env,
        num_agents_per_pool=config['ensemble']['num_agents_per_pool']
    )
    agents_path = Path(config['models']['ppo_agents'])
    if not agents_path.exists():
        raise FileNotFoundError(f"PPO agents not found: {agents_path}")
    agent_manager.load_all_pools(str(agents_path))

    weight_calculator = DynamicWeightCalculator(
        performance_window=config['ensemble']['performance_window'],
        temperature=config['ensemble']['temperature']
    )
    ensemble_trader = EnsembleTrader(
        weight_calculator=weight_calculator,
        performance_window=config['ensemble']['performance_window'],
        temperature=config['ensemble']['temperature']
    )
    ensemble_trader.initialize_agents(config['ensemble']['num_agents_per_pool'])

    execution_env = bull_env
    obs, info = execution_env.reset()
    initial_capital = config['training']['initial_capital']
    portfolio_value = initial_capital
    previous_regime_int = 1  # Sideways

    timestamps = state_data.index
    num_timesteps = len(timestamps)
    trading_history = []

    logger.info("Running main trading loop...")
    for t in range(num_timesteps - 1):
        timestamp = timestamps[t]
        state = state_data.loc[timestamp].values

        regime_result = regime_classifier.predict_with_confidence(state, previous_regime=previous_regime_int)
        current_regime = regime_result['regime_name']
        regime_confidence = regime_result['confidence']

        active_pool = agent_manager.get_pool(current_regime)
        ensemble_result = ensemble_trader.get_ensemble_action(state, active_pool)
        action = ensemble_result['action']
        weights = ensemble_result['weights']

        next_obs, reward, terminated, truncated, info = execution_env.step(action)
        portfolio_value = info.get('portfolio_value', portfolio_value)
        if 'agent_index' in info:
            ensemble_trader.update_agent_performance(info['agent_index'], portfolio_value)

        trading_history.append({
            'timestamp': timestamp,
            'regime': current_regime,
            'regime_confidence': regime_confidence,
            'action': action,
            'weights': weights,
            'portfolio_value': portfolio_value
        })
        regime_name_to_int = {'Bear': 0, 'Sideways': 1, 'Bull': 2}
        previous_regime_int = regime_name_to_int.get(current_regime, 1)

    logger.info("Backtest loop done. Computing metrics...")
    backtester = Backtester(
        initial_capital=config['training']['initial_capital'],
        transaction_fee=config['training']['transaction_fee'],
        slippage=config['training']['slippage']
    )
    backtest_results = backtester.run_backtest(trading_history, ohlcv_data)
    backtest_results['trading_history'] = trading_history
    return backtest_results


def metrics_to_comparison_format(metrics: Dict[str, float]) -> Dict[str, float]:
    """Backtester metrics 키를 논문 표기로 매핑."""
    return {
        'Sharpe Ratio': metrics.get('sharpe_ratio', 0.0),
        'Cumulative Return': metrics.get('cumulative_return', 0.0),
        'CAGR': metrics.get('cagr', 0.0),
        'Maximum Drawdown': metrics.get('max_drawdown', 0.0),
        'Win Rate': metrics.get('win_rate', 0.0),
        'Profit Factor': metrics.get('profit_factor', 0.0),
    }


def compare_with_paper(actual: Dict[str, float], method_name: str = 'Proposed Method') -> Dict[str, Any]:
    tolerance = {'Sharpe Ratio': 0.25, 'Cumulative Return': 0.15, 'CAGR': 0.08,
                 'Maximum Drawdown': 0.08, 'Win Rate': 0.08, 'Profit Factor': 0.3}
    paper = PAPER_TABLE2.get(method_name, {})
    out = {}
    for k, paper_val in paper.items():
        act = actual.get(k)
        if act is None:
            out[k] = {'paper': paper_val, 'actual': None, 'match': False}
            continue
        diff = abs(act - paper_val)
        tol = tolerance.get(k, 0.1)
        out[k] = {'paper': paper_val, 'actual': act, 'diff': diff, 'tolerance': tol, 'match': diff <= tol}
    return out


def write_report(backtest_metrics: Dict[str, float], comparison: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 80,
        "Full Pipeline - Performance Verification Report",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "=" * 80,
        "",
        "1. Actual performance (Proposed Method)",
        "-" * 80,
    ]
    for k, v in backtest_metrics.items():
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        "2. Comparison with Paper Table 2 (Proposed Method)",
        "-" * 80,
    ])
    for k, v in comparison.items():
        p, a, m = v.get('paper'), v.get('actual'), v.get('match', False)
        status = "OK" if m else "MISMATCH"
        a_str = f"{a}" if a is not None else "N/A"
        lines.append(f"  {k}: paper={p}, actual={a_str} [{status}]")
    lines.extend(["", "=" * 80])
    report_path = out_dir / "performance_verification_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)


def main():
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    config = load_config(str(config_path))

    logger.info("Step 1: Run full system backtest")
    backtest_results = run_backtest_and_get_metrics(config)
    metrics = backtest_results.get('metrics', {})
    if not metrics:
        logger.error("No metrics from backtester")
        return

    logger.info("Step 2: Compute performance metrics (done by Backtester)")
    actual_formatted = metrics_to_comparison_format(metrics)
    logger.info("Step 3: Compare with Paper Table 2")
    comparison = compare_with_paper(actual_formatted, 'Proposed Method')

    out_dir = Path('results/verification')
    write_report(metrics, comparison, out_dir)

    results_path = Path(config['results']['backtest']) / 'trading_results.pkl'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump({
            'trading_history': backtest_results.get('trading_history', []),
            'portfolio_values': backtest_results.get('portfolio_values', []).tolist() if hasattr(backtest_results.get('portfolio_values'), 'tolist') else backtest_results.get('portfolio_values', []),
            'metrics': metrics
        }, f)
    logger.info("Trading results saved to %s", results_path)

    def _to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(x) for x in obj]
        return obj

    summary = {
        'actual_metrics': actual_formatted,
        'paper_comparison': {k: {'paper': v['paper'], 'actual': v.get('actual'), 'match': v.get('match')}
                            for k, v in comparison.items()}
    }
    summary = _to_serializable(summary)
    json_path = out_dir / 'metrics_vs_paper.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("JSON summary: %s", json_path)
    logger.info("Full pipeline and verification completed.")


if __name__ == '__main__':
    main()
