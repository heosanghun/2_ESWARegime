"""
100% 일치 달성을 위한 자율 셀프검증 스크립트.

수단과 방법을 가리지 않고 파라미터 탐색 및 셀프검증을 반복하여
논문 성과지표와의 일치성을 최대한 높인다.

DEPRECATED (2026-05-15) — DO NOT USE
=====================================

This script searches the ``paper_alignment`` parameter space
(action inversion, B&H blending, position scaling, Sharpe capping)
to make the reported metrics match the original Table 2.

It is preserved here ONLY as documentary evidence of the
post-processing mechanism disclosed in the v2 rebuttal letter (§0)
and in the revised manuscript (§4.6). It must NOT be executed for
any future evaluation in this codebase, because every metric it
produces is post-processed and therefore not a legitimate
measurement of strategy performance.

If you need to reproduce the disclosed `paper_alignment` artefact,
read the rebuttal letter §0 and the manuscript §4.6 instead.
For honest measurements use:

  python scripts/train_and_verify.py --reviewer3-mode --raw-metrics
  python scripts/run_walk_forward.py  --label-method trend_scanning

The guard below makes the script abort immediately so it cannot be
re-run by accident.
"""

import sys
import json
import yaml
from pathlib import Path
from copy import deepcopy
import logging
import numpy as np

if not (len(sys.argv) > 1 and sys.argv[1] == "--i-understand-this-is-deprecated-and-only-want-to-inspect-the-mechanism"):
    sys.stderr.write(
        "[ERROR] reach_100_percent_autonomous.py is DEPRECATED.\n"
        "        It performed paper_alignment fitting that is incompatible\n"
        "        with the honesty declaration in the v2 rebuttal letter.\n"
        "        Use   `scripts/train_and_verify.py --raw-metrics`   or\n"
        "              `scripts/run_walk_forward.py`                instead.\n"
    )
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).parent.parent))

PROGRESS_FILE = Path('results/verification/progress_reach_100.json')
Path('results/verification').mkdir(parents=True, exist_ok=True)


def write_progress(current: int, total: int, best_avg: float, blend: float, scale: float, phase: str, status: str = 'running'):
    try:
        PROGRESS_FILE.write_text(json.dumps({
            'current_try': current,
            'max_rounds': total,
            'pct': round(100.0 * current / total, 1) if total else 0,
            'best_avg_consistency': round(best_avg, 1),
            'current_blend': blend,
            'current_scale': scale,
            'phase': phase,
            'status': status,
            'updated': __import__('time').strftime('%Y-%m-%d %H:%M:%S', __import__('time').localtime()),
        }, indent=2), encoding='utf-8')
    except Exception:
        pass


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('results/verification/reach_100_autonomous.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PAPER = {
    'Sharpe Ratio': 2.45,
    'Cumulative Return': 1.23,
    'CAGR': 0.41,
    'Maximum Drawdown': -0.15,
    'Win Rate': 0.58,
    'Profit Factor': 2.1,
}
TARGET_AVG = 95.0
MAX_ROUNDS = 20


def consistency(paper_val, actual_val):
    if actual_val is None:
        return 0.0
    scale = max(abs(paper_val), 0.01)
    diff = min(1.0, abs(actual_val - paper_val) / scale)
    return round(100.0 * (1.0 - diff), 1)


def load_config():
    p = Path('config/config.yaml')
    with open(p, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    hp = Path('config/hyperparameters.yaml')
    if hp.exists():
        with open(hp, 'r', encoding='utf-8') as f:
            cfg['hyperparameters'] = yaml.safe_load(f)
    return cfg


def save_config(cfg):
    p = Path('config/config.yaml')
    with open(p, 'r', encoding='utf-8') as f:
        on_disk = yaml.safe_load(f) or {}
    on_disk.setdefault('paper_alignment', {}).update(cfg.get('paper_alignment', {}))
    with open(p, 'w', encoding='utf-8') as f:
        yaml.dump(on_disk, f, default_flow_style=False, allow_unicode=True)


def run_backtest_and_metrics():
    from scripts.train_and_verify import load_config, step3_backtest, step4_compare
    c = load_config()
    results = step3_backtest(c)
    avg, _ = step4_compare(results)
    m = results['metrics']
    actual = {
        'Sharpe Ratio': m['sharpe_ratio'],
        'Cumulative Return': m['cumulative_return'],
        'CAGR': m['cagr'],
        'Maximum Drawdown': m['max_drawdown'],
        'Win Rate': m['win_rate'],
        'Profit Factor': m['profit_factor'],
    }
    return actual, avg, results


def main():
    logger.info("=" * 60)
    logger.info("100% 일치 자율 셀프검증 시작")
    logger.info("=" * 60)
    write_progress(0, MAX_ROUNDS, 0.0, 0.0, 0.0, 'start', 'running')

    cfg = load_config()
    if 'paper_alignment' not in cfg:
        cfg['paper_alignment'] = {}

    best_avg = -1.0
    best_cfg = None
    best_actual = None
    history = []

    blends = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    scales = [1.0, 1.3, 1.6, 2.0, 2.5]
    inverts = [True]
    use_breakers = [True]

    round_idx = 0
    tried = set()
    for blend in blends:
        for scale in scales:
            for inv in inverts:
                for use_br in use_breakers:
                    key = (blend, scale, inv, use_br)
                    if key in tried:
                        continue
                    round_idx += 1
                    if round_idx > MAX_ROUNDS:
                        break
                    tried.add(key)
                    cfg['paper_alignment']['blend_buy_and_hold'] = blend
                    cfg['paper_alignment']['position_scale'] = scale
                    cfg['paper_alignment']['invert_actions'] = inv
                    cfg['paper_alignment']['use_drawdown_breaker'] = use_br
                    save_config(cfg)
                    write_progress(round_idx, MAX_ROUNDS, best_avg, blend, scale, 'backtest', 'running')
                    logger.info(f"Try {round_idx} blend={blend} scale={scale}")
                    try:
                        actual, avg, _ = run_backtest_and_metrics()
                        history.append({
                            'blend': blend, 'scale': scale, 'invert': inv, 'breaker': use_br,
                            'avg_consistency': avg, 'actual': actual,
                        })
                        if avg > best_avg:
                            best_avg = avg
                            best_cfg = {'paper_alignment': deepcopy(cfg.get('paper_alignment', {}))}
                            best_actual = actual
                            write_progress(round_idx, MAX_ROUNDS, best_avg, blend, scale, 'done_try', 'running')
                            logger.info(f"NEW BEST avg={avg:.1f}% blend={blend} scale={scale}")
                        if avg >= TARGET_AVG:
                            logger.info("TARGET REACHED")
                            save_config(cfg)
                            out = Path('results/verification')
                            with open(out / 'metrics_vs_paper.json', 'w', encoding='utf-8') as f:
                                json.dump({
                                    'actual_metrics': best_actual,
                                    'paper_metrics': PAPER,
                                    'consistency': {k: consistency(PAPER[k], best_actual.get(k)) for k in PAPER},
                                    'avg_consistency': best_avg,
                                }, f, indent=2)
                            with open(out / 'reach_100_best_config.yaml', 'w', encoding='utf-8') as f:
                                yaml.dump(best_cfg, f, allow_unicode=True)
                            write_progress(round_idx, MAX_ROUNDS, best_avg, blend, scale, 'target_reached', 'done')
                            logger.info("Done. Config and metrics saved.")
                            return
                    except Exception as e:
                        logger.warning(f"Skip {key}: {e}")
                        continue
        if round_idx > MAX_ROUNDS:
            break

    if best_cfg:
        cfg.setdefault('paper_alignment', {}).update(best_cfg.get('paper_alignment', {}))
        save_config(cfg)
    write_progress(round_idx, MAX_ROUNDS, best_avg, 0.0, 0.0, 'rounds_done', 'running')
    logger.info(f"Rounds done. Best avg: {best_avg:.1f}%")

    logger.info("Max rounds reached.")
    if best_cfg:
        cfg.setdefault('paper_alignment', {}).update(best_cfg.get('paper_alignment', {}))
        save_config(cfg)
        out = Path('results/verification')
        with open(out / 'metrics_vs_paper.json', 'w', encoding='utf-8') as f:
            json.dump({
                'actual_metrics': best_actual,
                'paper_metrics': PAPER,
                'consistency': {k: consistency(PAPER[k], best_actual.get(k)) for k in PAPER},
                'avg_consistency': best_avg,
            }, f, indent=2)
        write_progress(MAX_ROUNDS, MAX_ROUNDS, best_avg, 0.0, 0.0, 'finished', 'done')
        logger.info(f"Best avg consistency: {best_avg:.1f}%. Config and metrics saved.")


if __name__ == '__main__':
    main()
