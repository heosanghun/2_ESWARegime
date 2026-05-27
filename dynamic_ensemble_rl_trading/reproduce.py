"""One-command reproduction of the audit headline numbers (ESWA-D-26-08980).

This script is the *single source of truth* for reproducing every headline
result cited in the rebuttal and the revised manuscript. It performs no
retraining — every model artefact is loaded from ``models/`` and only the
backtest layer is re-executed with `paper_alignment` strictly disabled
and PPO actions monkey-patched to ``deterministic=True``.

By default the script runs in *audit-fast* mode and reproduces:

  1. 5-fold walk-forward Bonferroni-corrected 95% bootstrap CIs for S1
     (primary honest configuration) and the ATR 1.8% sideways-filter
     screen.
  2. 2022 bear-market window (LUNA fold 1 + FTX fold 2) advanced risk
     metrics (Sortino, Calmar, CVaR, Pain Index, Ulcer Index) versus
     passive Buy & Hold.
  3. OOS 2024 forward-test (2024-03-01 .. 2024-08-31) on the latest
     fold-5 weights — pure out-of-sample, ≥ 6 months after training cutoff.

Usage::

    python reproduce.py            # all three blocks
    python reproduce.py --skip-oos # 1 + 2 only (no network needed)
    python reproduce.py --only ci  # just block 1
    python reproduce.py --only bear
    python reproduce.py --only oos
    python reproduce.py --download-data  # also re-pull OOS 2024 OHLCV

Outputs:

    results/audit/s1_statistical_tests.json/.md          (block 1a)
    results/audit/atr_screen_statistical_tests.json/.md  (block 1b)
    results/audit/bear_window_2022/advanced_metrics_deterministic.*   (block 2)
    results/audit/bear_window_2022/advanced_metrics_synthesis.md      (block 2 summary)
    results/audit/oos_2024_forward/advanced_metrics.json + summary.md (block 3)

Reviewer #4 reference: this is the *one* command intended to be invoked to
verify each headline number reported in the revised manuscript on the
reviewer's own hardware. ESWA_RAW_MODE=1 is set and the
reach_100_percent_autonomous.py post-processing optimizer is *not* invoked
(it is hard-guarded since 2026-05-15).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

logger = logging.getLogger("reproduce")

# Explicit hard contract:
os.environ["ESWA_RAW_MODE"] = "1"
os.environ.pop("ESWA_KEEP_INVERT", None)


def _run(label: str, cmd: list[str]) -> int:
    logger.info("\n=== %s ===", label)
    logger.info("    %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy())
    if proc.returncode != 0:
        logger.error("    [FAIL] exit=%d", proc.returncode)
    else:
        logger.info("    [OK]")
    return proc.returncode


def block_ci() -> int:
    """Block 1 — Bonferroni-corrected 95% bootstrap CIs for the 5-fold WF.

    Reproduces results/audit/{s1,atr_screen}_statistical_tests.{json,md}.
    """
    rc = 0
    rc |= _run(
        "Block 1a — S1 (primary honest) bootstrap CI",
        [
            sys.executable, "scripts/_stat_walk_forward.py",
            "--subdir", "routing_ablation/phase2_soft/S1_soft_ema12",
            "--out-dir", "audit",
            "--output-prefix", "s1_",
            "--label", "S1 soft routing + EMA12 (primary honest)",
        ],
    )
    rc |= _run(
        "Block 1b — ATR 1.8% sideways filter screen bootstrap CI",
        [
            sys.executable, "scripts/_stat_walk_forward.py",
            "--subdir", "autonomous/screen_extra_18",
            "--out-dir", "audit",
            "--output-prefix", "atr_screen_",
            "--label", "ATR 1.8% sideways filter screen (no retrain)",
        ],
    )
    return rc


def block_bear() -> int:
    """Block 2 — 2022 bear window advanced risk metrics (Sortino/Calmar/CVaR)."""
    rc = 0
    rc |= _run(
        "Block 2a — 2022 bear-window aggregate (existing fold summaries)",
        [sys.executable, "scripts/run_audit_bear_2022.py"],
    )
    rc |= _run(
        "Block 2b — 2022 bear-window deterministic advanced metrics",
        [sys.executable, "scripts/run_audit_bear_advanced_metrics.py", "--deterministic"],
    )
    return rc


def block_oos(download_data: bool) -> int:
    """Block 3 — OOS 2024 forward test (data download + deterministic backtest)."""
    rc = 0
    if download_data:
        rc |= _run(
            "Block 3a — Download BTC 1h OOS 2024 (Binance) + synthetic neutral news",
            [sys.executable, "scripts/fetch_oos_2024_data.py"],
        )
    else:
        ohlcv = ROOT / "data" / "raw" / "btcusdt_1h_oos2024.csv"
        if not ohlcv.exists():
            logger.warning(
                "OOS OHLCV not found at %s. Re-run with --download-data to pull it.", ohlcv,
            )
            return 1
    rc |= _run(
        "Block 3b — OOS 2024 forward-test backtest (deterministic, no retrain)",
        [sys.executable, "scripts/run_oos_2024_backtest.py"],
    )
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["ci", "bear", "oos"],
        default=None,
        help="Run only one block. Default: run all blocks in order.",
    )
    parser.add_argument(
        "--skip-oos",
        action="store_true",
        help="Skip the OOS 2024 forward test (use if no internet / cached data).",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="(Re-)download Binance OHLCV for the OOS 2024 window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Reachable banner for Reviewer #4 inspection.
    logger.info("ESWA-D-26-08980 — Audit reproducibility entry point")
    logger.info("    ESWA_RAW_MODE = %s (paper_alignment OFF)", os.environ.get("ESWA_RAW_MODE"))
    logger.info("    Working directory: %s", ROOT)
    logger.info("    Models root      : models/walk_forward_reward_v2/")

    rc = 0
    if args.only == "ci":
        rc |= block_ci()
    elif args.only == "bear":
        rc |= block_bear()
    elif args.only == "oos":
        rc |= block_oos(args.download_data)
    else:
        rc |= block_ci()
        rc |= block_bear()
        if not args.skip_oos:
            rc |= block_oos(args.download_data)

    if rc == 0:
        logger.info("\n[reproduce.py] All requested blocks completed successfully.")
    else:
        logger.warning("\n[reproduce.py] One or more blocks reported non-zero exit codes.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
