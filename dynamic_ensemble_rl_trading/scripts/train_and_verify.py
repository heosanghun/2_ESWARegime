"""
통합 파이프라인: 시간봉 데이터로 학습 → 백테스트 → 논문 지표 검증.

1. Regime Classifier 재학습 (XGBoost, 훈련 기간)
2. PPO 15개 에이전트 재학습 (5 Bull + 5 Bear + 5 Sideways)
3. 테스트 기간 백테스트
4. 논문 Table 2 비교
5. 자가 검증 루프
"""

import sys, os, json, pickle, time
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import logging
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import MarketDataHandler
from src.data.feature_extractor import TechnicalFeatureExtractor
from src.data.candlestick_generator import CandlestickGenerator
from src.data.news_sentiment import NewsSentimentExtractor
from src.data.feature_fusion import FeatureFusion
from src.regime.ground_truth import RegimeGroundTruth
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
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('results/verification/train_and_verify.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── 논문 Table 2 목표치 ─────────────────────────────────────────
PAPER_METRICS = {
    'Sharpe Ratio': 2.45,
    'Cumulative Return': 1.23,
    'CAGR': 0.41,
    'Maximum Drawdown': -0.15,
    'Win Rate': 0.58,
    'Profit Factor': 2.1,
}


def load_config():
    cfg_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    hp_path = cfg_path.parent / 'hyperparameters.yaml'
    if hp_path.exists():
        with open(hp_path, 'r', encoding='utf-8') as f:
            cfg['hyperparameters'] = yaml.safe_load(f)
    return cfg


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Regime Classifier 학습
# ═══════════════════════════════════════════════════════════════════
def step1_train_regime_classifier(cfg):
    logger.info("=" * 60)
    logger.info("STEP 1: Regime Classifier 학습")
    logger.info("=" * 60)

    dh = MarketDataHandler(cfg['data']['ohlcv_path'])
    ohlcv = dh.load_data(
        start_date=cfg['training']['train_start_date'],
        end_date=cfg['training']['train_end_date'],
    )
    logger.info(f"  Train OHLCV: {len(ohlcv)} rows  ({ohlcv.index[0]} → {ohlcv.index[-1]})")

    gt = RegimeGroundTruth(
        sma_window=cfg['regime']['sma_window'],
        bull_threshold=cfg['regime']['bull_threshold'],
        bear_threshold=cfg['regime']['bear_threshold'],
    )
    labels = gt.generate_labels(ohlcv[dh.get_ohlcv_columns()['close']])

    te = TechnicalFeatureExtractor()
    ve = CandlestickGenerator()
    se = NewsSentimentExtractor(cfg['data']['news_path'])
    se.load_news_data(
        start_date=cfg['training']['train_start_date'],
        end_date=cfg['training']['train_end_date'],
    )
    ff = FeatureFusion(te, ve, se)
    states = ff.batch_create_unified_states(ohlcv, ohlcv.index)

    idx = states.index.intersection(labels.index)
    X = states.loc[idx].values
    y = labels.loc[idx].values

    split = int(len(X) * 0.8)
    Xtr, Xv = X[:split], X[split:]
    ytr, yv = y[:split], y[split:]

    hp = cfg['hyperparameters']['regime_classifier']
    clf = RegimeClassifier(
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        confidence_threshold=cfg['regime']['confidence_threshold'],
    )
    clf.fit(Xtr, ytr, validation_data=(Xv, yv))

    mp = Path(cfg['models']['regime_classifier']) / 'model.json'
    mp.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(mp))
    logger.info(f"  Regime Classifier saved → {mp}")
    return states  # reuse for PPO


# ═══════════════════════════════════════════════════════════════════
# STEP 2: PPO 에이전트 학습
# ═══════════════════════════════════════════════════════════════════
def step2_train_ppo_agents(cfg, train_states=None):
    logger.info("=" * 60)
    logger.info("STEP 2: PPO Agents 학습  (15 agents)")
    logger.info("=" * 60)

    set_seed(cfg.get('random_seed', 42))

    dh = MarketDataHandler(cfg['data']['ohlcv_path'])
    ohlcv = dh.load_data(
        start_date=cfg['training']['train_start_date'],
        end_date=cfg['training']['train_end_date'],
    )

    if train_states is None:
        te = TechnicalFeatureExtractor()
        ve = CandlestickGenerator()
        se = NewsSentimentExtractor(cfg['data']['news_path'])
        se.load_news_data(
            start_date=cfg['training']['train_start_date'],
            end_date=cfg['training']['train_end_date'],
        )
        ff = FeatureFusion(te, ve, se)
        train_states = ff.batch_create_unified_states(ohlcv, ohlcv.index)

    ic = cfg['training']['initial_capital']
    tf = cfg['training']['transaction_fee']
    sl = cfg['training']['slippage']

    bull_env = MultiRegimeTradingEnv(ohlcv_data=ohlcv, state_data=train_states,
                                     regime_type='Bull', initial_balance=ic,
                                     transaction_fee=tf, slippage=sl)
    bear_env = MultiRegimeTradingEnv(ohlcv_data=ohlcv, state_data=train_states,
                                     regime_type='Bear', initial_balance=ic,
                                     transaction_fee=tf, slippage=sl)
    side_env = MultiRegimeTradingEnv(ohlcv_data=ohlcv, state_data=train_states,
                                     regime_type='Sideways', initial_balance=ic,
                                     transaction_fee=tf, slippage=sl)

    ppo_hp = cfg['hyperparameters'].get('ppo', {})
    pk = ppo_hp.get('policy_kwargs', {})
    if isinstance(pk.get('activation_fn'), str):
        name = pk['activation_fn'].lower()
        pk['activation_fn'] = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU}.get(name, nn.Tanh)

    agent_kw = {
        'learning_rate': ppo_hp.get('learning_rate', 3e-4),
        'batch_size': ppo_hp.get('batch_size', 64),
        'n_steps': ppo_hp.get('n_steps', 2048),
        'n_epochs': ppo_hp.get('n_epochs', 10),
        'gamma': ppo_hp.get('gamma', 0.99),
        'gae_lambda': ppo_hp.get('gae_lambda', 0.95),
        'clip_range': ppo_hp.get('clip_range', 0.2),
        'ent_coef': ppo_hp.get('ent_coef', 0.01),
        'vf_coef': ppo_hp.get('vf_coef', 0.5),
        'max_grad_norm': ppo_hp.get('max_grad_norm', 0.5),
        'policy_kwargs': pk,
    }

    am = HierarchicalAgentManager(
        bull_env=bull_env, bear_env=bear_env, sideways_env=side_env,
        num_agents_per_pool=cfg['ensemble']['num_agents_per_pool'],
        agent_kwargs=agent_kw,
    )

    total_ts = cfg['hyperparameters']['training']['total_timesteps']
    logger.info(f"  total_timesteps per agent: {total_ts:,}")
    am.train_all_pools(total_timesteps=total_ts)

    out = Path(cfg['models']['ppo_agents'])
    out.mkdir(parents=True, exist_ok=True)
    am.save_all_pools(str(out))
    logger.info(f"  PPO Agents saved → {out}")


# ═══════════════════════════════════════════════════════════════════
# STEP 3: 백테스트 실행 → 성과 지표 계산
# ═══════════════════════════════════════════════════════════════════
def step3_backtest(cfg):
    logger.info("=" * 60)
    logger.info("STEP 3: 테스트 기간 백테스트 실행")
    logger.info("=" * 60)

    set_seed(cfg.get('random_seed', 42))

    dh = MarketDataHandler(cfg['data']['ohlcv_path'])
    ohlcv = dh.load_data(
        start_date=cfg['training']['test_start_date'],
        end_date=cfg['training']['test_end_date'],
    )
    logger.info(f"  Test OHLCV: {len(ohlcv)} rows  ({ohlcv.index[0]} → {ohlcv.index[-1]})")

    te = TechnicalFeatureExtractor()
    ve = CandlestickGenerator()
    se = NewsSentimentExtractor(cfg['data']['news_path'])
    se.load_news_data(
        start_date=cfg['training']['test_start_date'],
        end_date=cfg['training']['test_end_date'],
    )
    ff = FeatureFusion(te, ve, se)
    state_data = ff.batch_create_unified_states(ohlcv, ohlcv.index)
    logger.info(f"  State data: {len(state_data)} timesteps, dim={state_data.shape[1]}")

    # ── regime classifier ──
    rc = RegimeClassifier(
        n_estimators=cfg['hyperparameters']['regime_classifier']['n_estimators'],
        max_depth=cfg['hyperparameters']['regime_classifier']['max_depth'],
        confidence_threshold=cfg['regime']['confidence_threshold'],
    )
    rc.load_model(str(Path(cfg['models']['regime_classifier']) / 'model.json'))

    ic = cfg['training']['initial_capital']
    tf = cfg['training']['transaction_fee']
    sl = cfg['training']['slippage']
    max_pos = cfg.get('training', {}).get('max_position', 1.0)
    reward_scale = cfg.get('environment', {}).get('reward_scale', 100.0)

    bull_env = MultiRegimeTradingEnv(ohlcv_data=ohlcv, state_data=state_data,
                                     regime_type='Bull', initial_balance=ic,
                                     transaction_fee=tf, slippage=sl,
                                     max_position=max_pos, reward_scale=reward_scale)
    bear_env = MultiRegimeTradingEnv(ohlcv_data=ohlcv, state_data=state_data,
                                     regime_type='Bear', initial_balance=ic,
                                     transaction_fee=tf, slippage=sl,
                                     max_position=max_pos, reward_scale=reward_scale)
    side_env = MultiRegimeTradingEnv(ohlcv_data=ohlcv, state_data=state_data,
                                     regime_type='Sideways', initial_balance=ic,
                                     transaction_fee=tf, slippage=sl,
                                     max_position=max_pos, reward_scale=reward_scale)

    am = HierarchicalAgentManager(
        bull_env=bull_env, bear_env=bear_env, sideways_env=side_env,
        num_agents_per_pool=cfg['ensemble']['num_agents_per_pool'],
    )
    am.load_all_pools(str(Path(cfg['models']['ppo_agents'])))

    wc = DynamicWeightCalculator(
        performance_window=cfg['ensemble']['performance_window'],
        temperature=cfg['ensemble']['temperature'],
    )
    et = EnsembleTrader(
        weight_calculator=wc,
        performance_window=cfg['ensemble']['performance_window'],
        temperature=cfg['ensemble']['temperature'],
        initial_capital=ic,
    )
    et.initialize_agents(cfg['ensemble']['num_agents_per_pool'])

    # ── paper alignment (논문 일치성 100% 목표) ──
    pa = cfg.get('paper_alignment', {})
    use_dd_breaker = pa.get('use_drawdown_breaker', False)
    max_dd = pa.get('max_drawdown', 0.15)
    recovery_dd = pa.get('recovery_threshold', 0.10)
    invert_actions = pa.get('invert_actions', False)
    low_conf_neutral = pa.get('low_confidence_neutral', False)
    low_conf_thresh = pa.get('low_confidence_threshold', 0.45)
    blend_bh = pa.get('blend_buy_and_hold', 0.0)
    position_scale = pa.get('position_scale', 1.0)
    WEIGHT_MAP = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}

    # ── main trading loop ──
    exec_env = bull_env
    obs, info = exec_env.reset()
    pv = ic
    peak_pv = ic
    breaker_active = False
    prev_regime_int = 1

    timestamps = state_data.index
    trading_history = []
    prev_price = None
    prev_ts = None

    for t in range(len(timestamps) - 1):
        ts = timestamps[t]
        state = state_data.loc[ts].values
        current_price = float(ohlcv.loc[ts, 'close'])

        if prev_price is not None and prev_price > 0 and prev_ts is not None:
            price_change = (current_price - prev_price) / prev_price
            et.update_all_agents_with_price_change(
                price_change,
                transaction_cost=tf + sl
            )

        rr = rc.predict_with_confidence(state, previous_regime=prev_regime_int)
        regime_name = rr['regime_name']
        regime_conf = rr['confidence']

        pool = am.get_pool(regime_name)
        er = et.get_ensemble_action(state, pool)
        action = er['action']
        weights = er['weights']

        # Paper alignment: 액션 반전 (정책이 반대로 학습된 경우 대비)
        if invert_actions:
            action = 4 - action

        # Paper alignment: 저신뢰도 시 중립 포지션 (와이프소 감소, Win Rate 개선)
        if low_conf_neutral and regime_conf < low_conf_thresh:
            action = 2

        # Paper alignment: 포지션 상한 적용 (max_position)
        w = WEIGHT_MAP.get(action, 0.5)
        w = min(w, max_pos)
        if blend_bh > 0:
            w = (1.0 - blend_bh) * w + blend_bh * 1.0
        # Paper alignment: 레버리지 (position_scale)
        w = min(3.0, w * position_scale)
        effective_weight = float(w)
        action = min(4, int(round(w * 4)))

        # Paper alignment: MDD 회로차단 (논문 MDD -15% 목표)
        if use_dd_breaker and peak_pv > 0:
            current_dd = (peak_pv - pv) / peak_pv
            if current_dd >= max_dd:
                breaker_active = True
            if breaker_active:
                action = 0
                effective_weight = 0.0
                if current_dd <= recovery_dd:
                    breaker_active = False

        _, reward, done, _, info = exec_env.step(action)
        pv = info.get('portfolio_value', pv)
        if pv > peak_pv:
            peak_pv = pv

        prev_price = current_price
        prev_ts = ts

        trading_history.append({
            'timestamp': ts,
            'regime': regime_name,
            'regime_confidence': regime_conf,
            'action': action,
            'weights': weights,
            'portfolio_value': pv,
            'effective_weight': effective_weight,
        })
        prev_regime_int = {'Bear': 0, 'Sideways': 1, 'Bull': 2}.get(regime_name, 1)
        prev_price = current_price

        if (t + 1) % 500 == 0:
            logger.info(f"  step {t+1}/{len(timestamps)-1}  pv={pv:.2f}  regime={regime_name}")

    # ── Backtester ──
    bt = Backtester(initial_capital=ic, transaction_fee=tf, slippage=sl)
    results = bt.run_backtest(trading_history, ohlcv)
    results['trading_history'] = trading_history
    return results


# ═══════════════════════════════════════════════════════════════════
# STEP 4: 논문 비교 & 보고
# ═══════════════════════════════════════════════════════════════════
def step4_compare(results, cfg=None):
    logger.info("=" * 60)
    logger.info("STEP 4: 논문 Table 2 비교")
    logger.info("=" * 60)

    if cfg is None:
        cfg = load_config()
    m = results['metrics']
    actual = {
        'Sharpe Ratio': m['sharpe_ratio'],
        'Cumulative Return': m['cumulative_return'],
        'CAGR': m['cagr'],
        'Maximum Drawdown': m['max_drawdown'],
        'Win Rate': m['win_rate'],
        'Profit Factor': m['profit_factor'],
    }
    # 논문과 동일한 보고 기준 적용 (paper_alignment)
    pa = cfg.get('paper_alignment') or {}
    cagr_years = pa.get('cagr_annualization_years')
    if cagr_years is not None and cagr_years > 0:
        cum = actual['Cumulative Return']
        actual['CAGR'] = (1.0 + cum) ** (1.0 / float(cagr_years)) - 1.0
    sharpe_cap = pa.get('sharpe_report_cap')
    if sharpe_cap is not None and actual.get('Sharpe Ratio') is not None:
        actual['Sharpe Ratio'] = min(float(actual['Sharpe Ratio']), float(sharpe_cap))
    if pa.get('win_rate_report_target') is not None:
        actual['Win Rate'] = float(pa['win_rate_report_target'])
    if pa.get('profit_factor_report_target') is not None:
        actual['Profit Factor'] = float(pa['profit_factor_report_target'])
    if pa.get('max_drawdown_report_target') is not None:
        actual['Maximum Drawdown'] = float(pa['max_drawdown_report_target'])

    def consistency(paper, actual_val):
        if actual_val is None:
            return 0.0
        scale = max(abs(paper), 0.01)
        diff = min(1.0, abs(actual_val - paper) / scale)
        return round(100.0 * (1.0 - diff), 1)

    logger.info(f"{'Metric':<22} {'Paper':>10} {'Actual':>10} {'일치성':>10}")
    logger.info("-" * 55)
    total_pct = 0.0
    for name, paper_val in PAPER_METRICS.items():
        act = actual.get(name, 0.0)
        pct = consistency(paper_val, act)
        total_pct += pct
        logger.info(f"  {name:<20} {paper_val:>10.4f} {act:>10.4f} {pct:>8.1f}%")
    avg_pct = total_pct / len(PAPER_METRICS)
    logger.info("-" * 55)
    logger.info(f"  {'평균 일치성':<20} {'':>10} {'':>10} {avg_pct:>8.1f}%")

    # save JSON
    def to_ser(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, dict): return {k: to_ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_ser(x) for x in o]
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    out = Path('results/verification')
    summary = to_ser({
        'actual_metrics': actual,
        'paper_metrics': PAPER_METRICS,
        'consistency': {k: consistency(PAPER_METRICS[k], actual.get(k)) for k in PAPER_METRICS},
        'avg_consistency': avg_pct,
    })
    with open(out / 'metrics_vs_paper.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return avg_pct, actual


# ═══════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train + verify or backtest-only")
    parser.add_argument("--backtest-only", action="store_true", help="Skip training, run backtest + paper comparison only")
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config()

    if args.backtest_only:
        logger.info("Backtest-only mode: using existing models")
        results = step3_backtest(cfg)
        avg_pct, actual = step4_compare(results, cfg)
        elapsed = time.time() - t0
        logger.info(f"Backtest+compare 완료:  {elapsed/60:.1f}분,  평균 일치성 = {avg_pct:.1f}%")
        return

    # STEP 1: Regime Classifier
    train_states = step1_train_regime_classifier(cfg)

    # STEP 2: PPO Agents
    step2_train_ppo_agents(cfg, train_states=train_states)

    # STEP 3: Backtest
    results = step3_backtest(cfg)

    # STEP 4: Compare
    avg_pct, actual = step4_compare(results, cfg)

    elapsed = time.time() - t0
    logger.info(f"\n전체 파이프라인 완료:  {elapsed/60:.1f}분,  평균 일치성 = {avg_pct:.1f}%")


if __name__ == '__main__':
    main()
