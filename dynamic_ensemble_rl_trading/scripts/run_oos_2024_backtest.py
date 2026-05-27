"""OOS 2024 forward-test backtest (no retraining).

Loads the existing walk_forward_reward_v2 fold_5 models (training cutoff
2023-08-19) and runs them on the 2024-03-01 → 2024-08-31 window, which is
strictly out-of-sample (≥ 6 months after training data ends).

Configurations evaluated (deterministic, PPOAgent.predict patched):
  - Buy & Hold (computed directly from data/raw/btcusdt_1h_oos2024.csv)
  - ATR 1.8% sideways filter screen (best honest defensive configuration)
  - S1 soft routing + EMA12             (primary honest configuration)

Output:
  results/audit/oos_2024_forward/atr_metrics.json
  results/audit/oos_2024_forward/s1_metrics.json
  results/audit/oos_2024_forward/buy_and_hold.json
  results/audit/oos_2024_forward/advanced_metrics.json
  results/audit/oos_2024_forward/equity_curves.csv
  results/audit/oos_2024_forward/summary.md
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

logger = logging.getLogger("oos2024_bt")

BARS_PER_YEAR = 24 * 365
SQRT_BARS_PER_YEAR = float(np.sqrt(BARS_PER_YEAR))

OOS_OHLCV = "data/raw/btcusdt_1h_oos2024.csv"
OOS_NEWS = "data/cryptonews_finbert_2024-03-01_2024-08-31.csv"
WINDOW_START = "2024-03-01"
WINDOW_END = "2024-08-31"
SOURCE_FOLD = 5  # most recent training cutoff = 2023-08-19


RERUN_CONFIGS = [
    {
        "id": "atr18_screen",
        "label": "ATR 1.8% sideways filter (best honest)",
        "confidence_threshold": 0.35,
        "prob_ema_span": 12,
        "routing_mode": "soft",
        "label_method": "trend_scanning",
        "atr_threshold": 0.018,
    },
    {
        "id": "s1_soft_ema12",
        "label": "S1 soft routing + EMA12 (primary honest)",
        "confidence_threshold": 0.35,
        "prob_ema_span": 12,
        "routing_mode": "soft",
        "label_method": "trend_scanning",
        "atr_threshold": None,
    },
]


# ---------------------------------------------------------------------------
# Risk metrics (shared with bear-window script, kept local for portability)
# ---------------------------------------------------------------------------

def compute_advanced_metrics(
    returns: np.ndarray,
    pv_series: Optional[np.ndarray] = None,
    *,
    target_return: float = 0.0,
    annualization_factor: int = BARS_PER_YEAR,
) -> Dict:
    af = annualization_factor
    sqrt_af = float(np.sqrt(af))
    n = int(len(returns))
    if n == 0:
        return {"n_bars": 0}
    rets = np.asarray(returns, dtype=float)
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=0))
    sharpe = (mean_r / std_r) * sqrt_af if std_r > 0 else float("nan")
    excess = rets - target_return
    downside = np.where(excess < 0, excess, 0.0)
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    sortino = (mean_r / downside_std) * sqrt_af if downside_std > 0 else float("nan")
    cum_ret = float(np.prod(1.0 + rets) - 1.0)
    years = n / af
    cagr = float((1.0 + cum_ret) ** (1.0 / years) - 1.0) if years > 0 and 1.0 + cum_ret > 0 else float("nan")
    if pv_series is None:
        pv = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    else:
        pv = np.asarray(pv_series, dtype=float)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / np.where(peak > 0, peak, 1.0)
    mdd = float(dd.min())
    calmar = (cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan")
    pain_idx = float(np.mean(np.abs(dd)))
    ulcer_idx = float(np.sqrt(np.mean(dd ** 2)))
    p5 = float(np.percentile(rets, 5))
    tail = rets[rets <= p5]
    cvar = float(np.mean(tail)) if len(tail) > 0 else float("nan")
    return {
        "n_bars": n,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Cumulative Return": cum_ret,
        "CAGR": cagr,
        "Maximum Drawdown": mdd,
        "Calmar Ratio": calmar,
        "VaR 95% (per-bar)": p5,
        "CVaR 95% (per-bar)": cvar,
        "Pain Index": pain_idx,
        "Ulcer Index": ulcer_idx,
    }


# ---------------------------------------------------------------------------
# Monkey-patch deterministic PPO
# ---------------------------------------------------------------------------

def patch_deterministic() -> None:
    from src.agents.ppo_agent import PPOAgent

    original = PPOAgent.predict

    def deterministic_predict(self, state, deterministic: bool = True):  # type: ignore[override]
        return original(self, state, deterministic=True)

    PPOAgent.predict = deterministic_predict  # type: ignore[assignment]
    logger.info("[monkey-patch] PPOAgent.predict forced to deterministic=True")


# ---------------------------------------------------------------------------
# Configuration override for OOS 2024
# ---------------------------------------------------------------------------

def build_oos_cfg(base_cfg: dict, cfg_spec: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    # Re-route data sources
    cfg["data"]["ohlcv_path"] = OOS_OHLCV
    cfg["data"]["news_path"] = OOS_NEWS
    cfg.setdefault("features", {}).setdefault("sentiment", {})["rescored_csv"] = OOS_NEWS

    # Training/test window — training window is unused (no retraining) but
    # train_end_date is used by feature_fusion warmup; set it just before
    # the OOS test start to keep the SubjectiveEMA initialisation sensible.
    cfg["training"]["train_start_date"] = WINDOW_START
    cfg["training"]["train_end_date"] = WINDOW_START
    cfg["training"]["test_start_date"] = WINDOW_START
    cfg["training"]["test_end_date"] = WINDOW_END

    # Apply config spec
    regime_cfg = cfg.setdefault("regime", {})
    regime_cfg["confidence_threshold"] = float(cfg_spec["confidence_threshold"])
    regime_cfg["prob_ema_span"] = int(cfg_spec["prob_ema_span"])
    regime_cfg["routing_mode"] = str(cfg_spec["routing_mode"]).lower()
    regime_cfg["label_method"] = str(cfg_spec["label_method"]).lower()
    atr_cfg = regime_cfg.setdefault("atr_sideways_filter", {})
    if cfg_spec.get("atr_threshold") is not None:
        atr_cfg["enabled"] = True
        atr_cfg["threshold"] = float(cfg_spec["atr_threshold"])
    else:
        atr_cfg["enabled"] = False

    # Point at the most recent fold's models
    fold_tag = f"fold_{SOURCE_FOLD}"
    models_root = "walk_forward_reward_v2"
    cfg["models"]["regime_classifier"] = str(
        Path("models") / models_root / fold_tag / "regime_classifier"
    )
    cfg["models"]["ppo_agents"] = str(
        Path("models") / models_root / fold_tag / "ppo_agents"
    )
    return cfg


# ---------------------------------------------------------------------------
# Buy & Hold from raw OHLCV
# ---------------------------------------------------------------------------

def buy_and_hold(csv_path: Path, start: str, end: str) -> Dict:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.loc[start:end]
    close = df["close"].astype(float).values
    rets = close[1:] / close[:-1] - 1.0
    pv = close / close[0]
    m = compute_advanced_metrics(rets, pv)
    m["window"] = [start, end]
    return m, pd.Series(pv, index=df.index[: len(pv)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = PROJECT_ROOT / "results" / "audit" / "oos_2024_forward"
    out_dir.mkdir(parents=True, exist_ok=True)

    bh_csv = PROJECT_ROOT / OOS_OHLCV
    if not bh_csv.exists():
        logger.error("OOS OHLCV missing: %s. Run scripts/fetch_oos_2024_data.py first.", bh_csv)
        return 1

    bh_metrics, bh_pv_series = buy_and_hold(bh_csv, WINDOW_START, WINDOW_END)
    (out_dir / "buy_and_hold.json").write_text(json.dumps(bh_metrics, indent=2), encoding="utf-8")
    logger.info(
        "B&H: Sharpe=%.2f Sortino=%.2f Calmar=%.2f MDD=%.1f%% CumRet=%.1f%% CVaR=%.4f%%",
        bh_metrics["Sharpe Ratio"], bh_metrics["Sortino Ratio"],
        bh_metrics["Calmar Ratio"], 100 * bh_metrics["Maximum Drawdown"],
        100 * bh_metrics["Cumulative Return"], 100 * bh_metrics["CVaR 95% (per-bar)"],
    )

    patch_deterministic()
    os.environ["ESWA_RAW_MODE"] = "1"
    os.environ.pop("ESWA_KEEP_INVERT", None)

    from scripts.train_and_verify import load_config, step3_backtest, step4_compare
    base_cfg = load_config()

    cfg_results: Dict[str, Dict] = {}
    equity_curves: Dict[str, pd.Series] = {"buy_and_hold": bh_pv_series}

    for spec in RERUN_CONFIGS:
        logger.info("==== Running OOS 2024 backtest: %s ====", spec["id"])
        cfg = build_oos_cfg(base_cfg, spec)
        results = step3_backtest(cfg)
        _, agg_metrics = step4_compare(results, cfg, raw=True)

        trading_history = results["trading_history"]
        pv = np.array([row["portfolio_value"] for row in trading_history], dtype=float)
        if len(pv) < 2:
            logger.error("Trading history empty for %s", spec["id"])
            continue
        rets = pv[1:] / pv[:-1] - 1.0
        rets = np.where(np.isfinite(rets), rets, 0.0).astype(float)
        m = compute_advanced_metrics(rets, pv)
        m["window"] = [WINDOW_START, WINDOW_END]
        m["rerun_metrics_aggregated"] = {k: float(v) for k, v in agg_metrics.items()}
        m["routing_diagnostics"] = results.get("routing_diagnostics", {})

        cfg_results[spec["id"]] = {
            "label": spec["label"],
            "config_spec": {k: v for k, v in spec.items() if k != "id"},
            "source_fold": SOURCE_FOLD,
            "source_models": "walk_forward_reward_v2",
            "advanced_metrics": m,
        }
        (out_dir / f"{spec['id']}_metrics.json").write_text(
            json.dumps(cfg_results[spec["id"]], indent=2), encoding="utf-8"
        )
        # Equity curve
        ts_index = pd.DatetimeIndex([row["timestamp"] for row in trading_history])
        equity_curves[spec["id"]] = pd.Series(pv, index=ts_index, name=spec["id"])
        logger.info(
            "%s OOS 2024: Sharpe=%.2f Sortino=%.2f Calmar=%.2f MDD=%.1f%% CumRet=%.1f%% CVaR=%.4f%%",
            spec["id"], m["Sharpe Ratio"], m["Sortino Ratio"],
            m["Calmar Ratio"], 100 * m["Maximum Drawdown"],
            100 * m["Cumulative Return"], 100 * m["CVaR 95% (per-bar)"],
        )

    # Combined equity curves
    eq_df = pd.DataFrame(equity_curves).sort_index()
    eq_df.to_csv(out_dir / "equity_curves.csv")
    logger.info("Wrote equity_curves.csv (%d rows × %d cols)", len(eq_df), eq_df.shape[1])

    # Master summary
    payload = {
        "window": [WINDOW_START, WINDOW_END],
        "annualization_factor": BARS_PER_YEAR,
        "honest_mode": True,
        "deterministic_actions": True,
        "source_fold": SOURCE_FOLD,
        "source_models": "walk_forward_reward_v2",
        "training_cutoff": "2023-08-19",
        "oos_gap_months": 6,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "buy_and_hold": bh_metrics,
        "configurations": cfg_results,
    }
    (out_dir / "advanced_metrics.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )

    md_lines: List[str] = []
    md_lines.append("# OOS 2024 Forward Test — Advanced Risk Metrics")
    md_lines.append("")
    md_lines.append(f"_Generated: {payload['generated']}_  ")
    md_lines.append(
        f"**Window:** {WINDOW_START} → {WINDOW_END}    "
        f"**Training cutoff (fold {SOURCE_FOLD}):** 2023-08-19    "
        f"**OOS gap:** ≥ 6 months    "
        f"**Mode:** `ESWA_RAW_MODE=1` deterministic, Backtester v2.0.1"
    )
    md_lines.append("")
    md_lines.append("## Headline comparison")
    md_lines.append("")
    md_lines.append(
        "| Config | Sharpe | **Sortino** | **Calmar** | CumRet | MDD | "
        "**CVaR 95%/bar** | Pain Idx | Ulcer Idx |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    def _r(label: str, m: Dict) -> str:
        f = lambda k: ("%.2f" % m[k]) if np.isfinite(m.get(k, np.nan)) else "—"
        pct = lambda k: ("%.1f%%" % (100 * m[k])) if np.isfinite(m.get(k, np.nan)) else "—"
        pcpb = lambda k: ("%.3f%%" % (100 * m[k])) if np.isfinite(m.get(k, np.nan)) else "—"
        return (
            f"| {label} | {f('Sharpe Ratio')} | **{f('Sortino Ratio')}** | "
            f"**{f('Calmar Ratio')}** | {pct('Cumulative Return')} | "
            f"{pct('Maximum Drawdown')} | **{pcpb('CVaR 95% (per-bar)')}** | "
            f"{pcpb('Pain Index')} | {pcpb('Ulcer Index')} |"
        )

    md_lines.append(_r("Buy & Hold (passive)", bh_metrics))
    for cfg in cfg_results.values():
        md_lines.append(_r(cfg["label"], cfg["advanced_metrics"]))
    md_lines.append("")
    md_lines.append("## Routing diagnostics")
    md_lines.append("")
    for cfg in cfg_results.values():
        rd = cfg["advanced_metrics"].get("routing_diagnostics", {})
        md_lines.append(
            f"- **{cfg['label']}** — "
            f"n_steps={rd.get('n_steps')}, "
            f"sideways_pct={rd.get('sideways_pct')}, "
            f"regime_switch_count={rd.get('regime_switch_count')}, "
            f"atr_filter_pct={rd.get('atr_filter_pct')}, "
            f"routing_accuracy={rd.get('routing_accuracy')}"
        )
    md_lines.append("")
    md_lines.append("## Sentiment caveat")
    md_lines.append("")
    md_lines.append(
        "The OOS 2024 forward test uses a synthetic neutral news placeholder "
        f"(`{OOS_NEWS}`) generated at one article every 3 hours. This mirrors "
        "the placeholder-style schema observed in the original 2021-2023 dataset "
        "and removes any chance of look-ahead bias from a 2025-era LLM having "
        "re-scored 2024 headlines. Consequently, the sentiment feature contribution "
        "during OOS 2024 is effectively zero, isolating the price-dynamics + "
        "regime-classifier pathway."
    )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_dir / "summary.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
