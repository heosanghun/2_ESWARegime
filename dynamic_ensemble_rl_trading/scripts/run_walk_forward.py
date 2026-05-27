"""
Walk-Forward Expanding Window evaluation runner.

This script orchestrates the existing train_and_verify pipeline across
multiple chronological folds, as required by paper Section 4.1 and
Reviewer #3 Comment 3.3. Each fold:

  - Trains the regime classifier + PPO ensemble on data
    [global_train_start, fold_train_end]
  - Evaluates on data
    [fold_train_end, fold_train_end + test_size]
  - Reports both per-fold and aggregated metrics

Per-fold artefacts are isolated in `models/walk_forward/fold_<k>/` and
`results/walk_forward/fold_<k>/` so successive folds do not overwrite
each other.

Output
------
- results/walk_forward/summary.json   (machine-readable aggregate)
- results/walk_forward/summary.md     (human-readable Table 2 style report)
- results/walk_forward/fold_<k>/      (per-fold backtest + comparison)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing pipeline functions
from scripts.train_and_verify import (  # noqa: E402
    load_config,
    step1_train_regime_classifier,
    step2_train_ppo_agents,
    step3_backtest,
    step4_compare,
    PAPER_METRICS,
)

logger = logging.getLogger("walk_forward")

# Default fold boundaries (5-fold expanding window over the published
# 26-month dataset 2021-10-12 .. 2023-12-19). Each test fold spans
# roughly 4-5 months of out-of-sample data so each fold yields
# statistically meaningful realised trades.
DEFAULT_FOLDS: List[Tuple[str, str]] = [
    ("2022-04-19", "2022-08-19"),  # Q2-Q3 2022 (Terra/LUNA crash)
    ("2022-08-19", "2022-12-19"),  # Bear continuation
    ("2022-12-19", "2023-04-19"),  # Recovery
    ("2023-04-19", "2023-08-19"),  # Bull run
    ("2023-08-19", "2023-12-19"),  # Late 2023
]


def _setup_logger(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def run_one_fold(
    base_cfg: dict,
    fold_idx: int,
    fold_test_start: str,
    fold_test_end: str,
    global_train_start: str,
    raw_metrics: bool = True,
    label_method: str | None = None,
    subdir: str = "walk_forward",
    total_timesteps: int | None = None,
    routing_mode: str | None = None,
    prob_ema_span: int | None = None,
    atr_sideways_enabled: bool | None = None,
    atr_sideways_threshold: float | None = None,
) -> Dict:
    """Train + evaluate on a single walk-forward fold.

    Returns a dictionary with per-fold metrics and metadata.
    """
    cfg = copy.deepcopy(base_cfg)

    # Override date ranges.
    cfg["training"]["train_start_date"] = global_train_start
    cfg["training"]["train_end_date"] = fold_test_start
    cfg["training"]["test_start_date"] = fold_test_start
    cfg["training"]["test_end_date"] = fold_test_end

    if label_method is not None:
        cfg.setdefault("regime", {})["label_method"] = label_method

    if routing_mode is not None:
        cfg.setdefault("regime", {})["routing_mode"] = routing_mode
    if prob_ema_span is not None:
        cfg.setdefault("regime", {})["prob_ema_span"] = int(prob_ema_span)
    if atr_sideways_enabled is not None or atr_sideways_threshold is not None:
        atr_cfg = cfg.setdefault("regime", {}).setdefault("atr_sideways_filter", {})
        if atr_sideways_enabled is not None:
            atr_cfg["enabled"] = bool(atr_sideways_enabled)
        if atr_sideways_threshold is not None:
            atr_cfg["threshold"] = float(atr_sideways_threshold)
            atr_cfg["enabled"] = True

    if total_timesteps is not None:
        cfg.setdefault("hyperparameters", {}).setdefault("training", {})[
            "total_timesteps"
        ] = int(total_timesteps)
        cfg.setdefault("training", {})["total_timesteps"] = int(total_timesteps)

    # Isolate per-fold artefacts.
    fold_tag = f"fold_{fold_idx}"
    cfg["models"]["regime_classifier"] = str(
        Path("models") / subdir / fold_tag / "regime_classifier"
    )
    cfg["models"]["ppo_agents"] = str(
        Path("models") / subdir / fold_tag / "ppo_agents"
    )
    results_dir = Path("results") / subdir / fold_tag
    results_dir.mkdir(parents=True, exist_ok=True)

    if raw_metrics:
        os.environ["ESWA_RAW_MODE"] = "1"
    os.environ.pop("ESWA_KEEP_INVERT", None)

    logger.info("=" * 78)
    logger.info(
        "FOLD %d:  train [%s -> %s]   test [%s -> %s]",
        fold_idx,
        global_train_start,
        fold_test_start,
        fold_test_start,
        fold_test_end,
    )
    logger.info("=" * 78)

    t0 = time.time()

    # Pipeline steps.
    train_states = step1_train_regime_classifier(cfg)
    step2_train_ppo_agents(cfg, train_states=train_states)
    results = step3_backtest(cfg)
    avg_pct, actual = step4_compare(results, cfg, raw=raw_metrics)

    elapsed = time.time() - t0

    fold_summary = {
        "fold": fold_idx,
        "train_start": global_train_start,
        "train_end": fold_test_start,
        "test_start": fold_test_start,
        "test_end": fold_test_end,
        "elapsed_min": elapsed / 60.0,
        "avg_consistency_pct": avg_pct,
        "metrics": {k: float(v) for k, v in actual.items()},
    }

    # Persist per-fold summary as JSON for downstream aggregation.
    with open(results_dir / "fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)
    logger.info("Fold %d done in %.1f min  (avg_consistency=%.1f%%)",
                fold_idx, elapsed / 60.0, avg_pct)
    return fold_summary


def aggregate_folds(folds: List[Dict]) -> Dict:
    """Mean / median / std across walk-forward folds."""
    metric_keys = list(folds[0]["metrics"].keys())
    agg: Dict[str, Dict[str, float]] = {}
    for k in metric_keys:
        vals = np.array([f["metrics"][k] for f in folds], dtype=float)
        agg[k] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": int(len(vals)),
            "values": vals.tolist(),
        }
    consistency = np.array([f["avg_consistency_pct"] for f in folds])
    return {
        "metrics_aggregate": agg,
        "avg_consistency_pct": {
            "mean": float(np.mean(consistency)),
            "median": float(np.median(consistency)),
            "std": float(np.std(consistency, ddof=1)) if len(consistency) > 1 else 0.0,
        },
        "n_folds": len(folds),
    }


def write_summary_markdown(folds: List[Dict], agg: Dict, out_path: Path) -> None:
    """Render a human-readable Table 2 style aggregate report."""
    lines: List[str] = []
    lines.append("# Walk-Forward Expanding Window — Aggregate Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append(f"**Folds:** {len(folds)}    |    **Mode:** raw (paper_alignment OFF)")
    lines.append("")
    lines.append("## Per-fold metrics")
    lines.append("")
    cols = list(PAPER_METRICS.keys())
    header = "| Fold | Train | Test | " + " | ".join(cols) + " | Consistency |"
    sep = "|------|-------|------|" + "|".join(["------:"] * (len(cols) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for f in folds:
        m = f["metrics"]
        row_cells = [f"{m.get(c, 0.0):.4f}" for c in cols]
        lines.append(
            f"| {f['fold']} | {f['train_start']}..{f['train_end']} | "
            f"{f['test_start']}..{f['test_end']} | "
            + " | ".join(row_cells)
            + f" | {f['avg_consistency_pct']:.1f}% |"
        )
    lines.append("")
    lines.append("## Aggregate (mean / median / std)")
    lines.append("")
    lines.append("| Metric | Paper | Mean | Median | Std | Min | Max |")
    lines.append("|--------|------:|-----:|-------:|----:|----:|----:|")
    for c in cols:
        a = agg["metrics_aggregate"].get(c, {})
        paper_v = PAPER_METRICS[c]
        lines.append(
            f"| {c} | {paper_v} | {a.get('mean', 0):.4f} | "
            f"{a.get('median', 0):.4f} | {a.get('std', 0):.4f} | "
            f"{a.get('min', 0):.4f} | {a.get('max', 0):.4f} |"
        )
    lines.append("")
    lines.append(
        f"**Avg consistency (mean across folds):** "
        f"{agg['avg_consistency_pct']['mean']:.1f}%"
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Optional semicolon-separated list of test windows: "
             "'YYYY-MM-DD..YYYY-MM-DD;YYYY-MM-DD..YYYY-MM-DD'. "
             "If omitted, the default 5-fold schedule is used.",
    )
    parser.add_argument(
        "--global-train-start",
        default="2021-10-12",
        help="Anchor start for the expanding training window.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Skip folds with index < this. Useful for resuming after a crash.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Cap on number of folds to run (handy for smoke tests).",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Disable raw-metrics mode (re-enables paper_alignment, NOT recommended).",
    )
    parser.add_argument(
        "--label-method",
        choices=["trend_scanning", "causal_trend_scanning", "sma"],
        default=None,
        help="Override config.regime.label_method for every fold.",
    )
    parser.add_argument(
        "--subdir",
        default="walk_forward",
        help="Sub-directory under models/ and results/ for this run "
             "(use a unique name when running comparison studies).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override PPO total_timesteps per agent for every fold.",
    )
    parser.add_argument(
        "--autopilot",
        action="store_true",
        help="After all folds finish, automatically run post-WF pipeline "
             "(sanity → rebacktest → stats → report). See scripts/_post_wf_autopilot.py.",
    )
    args = parser.parse_args()

    _setup_logger(Path("results") / args.subdir / "walk_forward.log")

    base_cfg = load_config()

    if args.folds:
        folds: List[Tuple[str, str]] = []
        for entry in args.folds.split(";"):
            if not entry.strip():
                continue
            ts, te = entry.split("..")
            folds.append((ts.strip(), te.strip()))
    else:
        folds = list(DEFAULT_FOLDS)

    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    fold_summaries: List[Dict] = []
    t_total = time.time()
    for idx, (ts, te) in enumerate(folds, start=1):
        if idx < args.start_from:
            logger.info("Skipping fold %d (start_from=%d).", idx, args.start_from)
            continue
        try:
            summary = run_one_fold(
                base_cfg,
                fold_idx=idx,
                fold_test_start=ts,
                fold_test_end=te,
                global_train_start=args.global_train_start,
                raw_metrics=not args.no_raw,
                label_method=args.label_method,
                subdir=args.subdir,
                total_timesteps=args.total_timesteps,
            )
            fold_summaries.append(summary)
        except Exception as exc:
            logger.exception("Fold %d crashed: %s", idx, exc)
            continue

    if not fold_summaries:
        logger.error("No folds completed; nothing to aggregate.")
        return 1

    agg = aggregate_folds(fold_summaries)
    out_dir = Path("results") / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"folds": fold_summaries, "aggregate": agg}, f, indent=2)
    write_summary_markdown(fold_summaries, agg, out_dir / "summary.md")
    logger.info(
        "Walk-forward complete. %d folds, total %.1f min.",
        len(fold_summaries),
        (time.time() - t_total) / 60.0,
    )
    logger.info("Aggregate written to %s", out_dir / "summary.md")

    if args.autopilot:
        from scripts._post_wf_autopilot import run_post_pipeline

        logger.info("Starting post-WF autopilot for subdir=%s", args.subdir)
        return run_post_pipeline(subdir=args.subdir, skip_wait=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
