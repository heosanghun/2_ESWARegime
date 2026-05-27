"""Audit P1.2 — 2022 bear-market window honest evaluation.

Aggregates fold 1 + fold 2 (2022-04-19 to 2022-12-19) from existing
walk-forward runs WITHOUT retraining, and computes a Buy-and-Hold
benchmark over the same window from data/raw/btcusdt_1h.csv.

Configurations evaluated (read from fold_*/fold_summary.json):

  (a) v1 baseline       — results/walk_forward/
  (b) S1 (primary)      — results/routing_ablation/phase2_soft/S1_soft_ema12/
  (c) ATR 1.8% screen   — results/autonomous/screen_extra_18/

Output: results/audit/bear_window_2022/summary.json
        results/audit/bear_window_2022/summary.md
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("audit_bear")

WINDOW_START = "2022-04-19"
WINDOW_END = "2022-12-19"
BEAR_FOLDS = [1, 2]  # fold 1: 2022-04→08, fold 2: 2022-08→12

CONFIGS = [
    {
        "id": "v1_baseline",
        "label": "v1 baseline (worst case reference)",
        "subdir": "walk_forward",
    },
    {
        "id": "s1_soft_ema12",
        "label": "S1 soft routing + EMA12 (primary honest)",
        "subdir": "routing_ablation/phase2_soft/S1_soft_ema12",
    },
    {
        "id": "atr18_screen",
        "label": "ATR 1.8% sideways filter (best honest)",
        "subdir": "autonomous/screen_extra_18",
    },
]

# Hourly bars → annualisation factor for Sharpe.
HOURLY_ANNUAL_FACTOR = float(np.sqrt(24 * 365))


def load_fold(subdir: str, fold: int) -> Dict:
    path = PROJECT_ROOT / "results" / subdir / f"fold_{fold}" / "fold_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing fold summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_two_folds(folds: List[Dict]) -> Dict:
    metrics: Dict[str, Dict] = {}
    keys = list(folds[0]["metrics"].keys())
    for k in keys:
        vals = np.array([f["metrics"][k] for f in folds], dtype=float)
        metrics[k] = {
            "fold_1": float(vals[0]),
            "fold_2": float(vals[1]),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return metrics


def max_consecutive_losing_days(returns: pd.Series) -> int:
    daily = returns.resample("D").sum()
    streak = max_streak = 0
    for r in daily.values:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return int(max_streak)


def compute_bh_metrics(csv_path: Path, start: str, end: str) -> Dict:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column missing in {csv_path}. Got: {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.loc[start:end]
    if df.empty:
        raise ValueError(f"No bars in window {start} → {end}")
    close = df["close"].astype(float)
    rets = close.pct_change().dropna()
    sharpe = float(HOURLY_ANNUAL_FACTOR * rets.mean() / rets.std())
    cum_ret = float((close.iloc[-1] / close.iloc[0]) - 1.0)
    peak = close.cummax()
    dd = (close - peak) / peak
    mdd = float(dd.min())
    max_loss_days = max_consecutive_losing_days(rets)
    return {
        "window": [start, end],
        "n_bars": int(len(rets)),
        "Sharpe Ratio": sharpe,
        "Cumulative Return": cum_ret,
        "Maximum Drawdown": mdd,
        "Max consecutive loss days": max_loss_days,
    }


def render_markdown(results: Dict) -> str:
    cfg_rows = []
    bh = results["buy_and_hold"]
    cfg_rows.append(
        f"| Buy & Hold (passive) | — | {bh['Sharpe Ratio']:.2f} | "
        f"{100*bh['Cumulative Return']:.1f}% | {100*bh['Maximum Drawdown']:.1f}% | "
        f"{bh['Max consecutive loss days']} |"
    )
    for cfg_id, cfg_data in results["configurations"].items():
        if "error" in cfg_data:
            cfg_rows.append(f"| {cfg_data.get('label', cfg_id)} | ERROR | — | — | — | — |")
            continue
        m = cfg_data["metrics"]
        cfg_rows.append(
            f"| {cfg_data['label']} | `{cfg_data['subdir']}` | "
            f"{m['Sharpe Ratio']['mean']:.2f} | "
            f"{100*m['Cumulative Return']['mean']:.1f}% | "
            f"{100*m['Maximum Drawdown']['mean']:.1f}% | — |"
        )
    return "\n".join(
        [
            "# Audit P1.2 — 2022 Bear-Market Window",
            "",
            f"_Generated: {results['generated']}_",
            "",
            f"**Window:** {WINDOW_START} → {WINDOW_END}    "
            f"|    **Folds aggregated:** {BEAR_FOLDS}    "
            f"|    **Honest mode:** `ESWA_RAW_MODE=1` (paper_alignment OFF)",
            "",
            "## Headline comparison",
            "",
            "| Configuration | Source | Mean Sharpe | Mean CumRet | Mean MDD | Max consec. loss days |",
            "|---|---|---:|---:|---:|---:|",
            *cfg_rows,
            "",
            "## Per-fold detail",
            "",
            *_per_fold_section(results),
            "",
            "## Interpretation",
            "",
            "Fold 1 covers the LUNA/Terra crash (May 2022) and fold 2 covers the "
            "FTX collapse (Nov 2022); jointly they form the canonical 2022 "
            "bear-market stress test demanded by Reviewer #2 in ESWA-D-26-08980.",
            "",
            "Numbers are **post Backtester v2.0.1 long-short fix** (2026-05-19) "
            "and `paper_alignment` is fully OFF.",
        ]
    )


def _per_fold_section(results: Dict) -> List[str]:
    lines: List[str] = []
    for cfg_id, cfg_data in results["configurations"].items():
        if "error" in cfg_data:
            continue
        lines.extend([
            f"### {cfg_data['label']}",
            "",
            "| Fold | Test window | Sharpe | CumRet | MDD | Win Rate | Profit Factor |",
            "|----:|-------------|-------:|-------:|----:|---------:|--------------:|",
        ])
        for f_meta, f_full in zip(cfg_data["fold_test_windows"], cfg_data["fold_raw"]):
            mm = f_full["metrics"]
            lines.append(
                f"| {f_meta['fold']} | "
                f"{f_meta['test_start']}..{f_meta['test_end']} | "
                f"{mm['Sharpe Ratio']:.2f} | "
                f"{100*mm['Cumulative Return']:.1f}% | "
                f"{100*mm['Maximum Drawdown']:.1f}% | "
                f"{100*mm['Win Rate']:.1f}% | "
                f"{mm['Profit Factor']:.2f} |"
            )
        lines.append("")
    return lines


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = PROJECT_ROOT / "results" / "audit" / "bear_window_2022"
    out_dir.mkdir(parents=True, exist_ok=True)

    bh_path = PROJECT_ROOT / "data" / "raw" / "btcusdt_1h.csv"
    if not bh_path.exists():
        logger.error("BTC CSV not found at %s", bh_path)
        return 1

    logger.info("Computing Buy-and-Hold over %s → %s", WINDOW_START, WINDOW_END)
    bh = compute_bh_metrics(bh_path, WINDOW_START, WINDOW_END)
    logger.info(
        "B&H: Sharpe=%.2f  CumRet=%.1f%%  MDD=%.1f%%  MaxLossDays=%d",
        bh["Sharpe Ratio"],
        100 * bh["Cumulative Return"],
        100 * bh["Maximum Drawdown"],
        bh["Max consecutive loss days"],
    )

    results: Dict = {
        "window": [WINDOW_START, WINDOW_END],
        "folds_used": BEAR_FOLDS,
        "honest_mode": True,
        "buy_and_hold": bh,
        "configurations": {},
    }

    for cfg in CONFIGS:
        try:
            folds = [load_fold(cfg["subdir"], f) for f in BEAR_FOLDS]
            metrics = aggregate_two_folds(folds)
            results["configurations"][cfg["id"]] = {
                "label": cfg["label"],
                "subdir": cfg["subdir"],
                "fold_test_windows": [
                    {"fold": f["fold"], "test_start": f["test_start"], "test_end": f["test_end"]}
                    for f in folds
                ],
                "fold_raw": folds,
                "metrics": metrics,
            }
            logger.info(
                "%s: Sharpe mean %.2f, CumRet mean %.1f%%, MDD mean %.1f%%",
                cfg["id"],
                metrics["Sharpe Ratio"]["mean"],
                100 * metrics["Cumulative Return"]["mean"],
                100 * metrics["Maximum Drawdown"]["mean"],
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", cfg["id"], exc)
            results["configurations"][cfg["id"]] = {
                "label": cfg["label"],
                "subdir": cfg["subdir"],
                "error": str(exc),
            }

    results["generated"] = datetime.now().isoformat(timespec="seconds")

    out_json = out_dir / "summary.json"
    out_md = out_dir / "summary.md"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(results), encoding="utf-8")
    logger.info("Wrote %s and %s", out_json, out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
