"""
Bootstrap CI, Ledoit-Wolf Sharpe equality test, Bonferroni correction
(Reviewer #2 / #4, item #9).

Requires the artifacts produced by ``train_and_verify.py`` so we have
strategy returns and a benchmark series to compare against. If a
prepared returns pickle is missing, we regenerate by replaying the
backtest in ``--backtest-only`` mode.

Outputs:
  - ``results/verification/statistical_tests.md``
  - ``results/verification/statistical_tests.json``
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistical_tests import comprehensive_report  # noqa: E402

logger = logging.getLogger(__name__)


def _strategy_returns() -> np.ndarray:
    from scripts import train_and_verify as tv  # type: ignore

    cfg = tv.load_config()
    results = tv.step3_backtest(cfg)
    rets = np.asarray(results.get("returns", []), dtype=float)
    return rets


def _benchmark_buy_and_hold() -> np.ndarray:
    from scripts import train_and_verify as tv  # type: ignore
    from src.data.data_processor import MarketDataHandler

    cfg = tv.load_config()
    dh = MarketDataHandler(cfg["data"]["ohlcv_path"])
    ohlcv = dh.load_data(
        start_date=cfg["training"]["test_start_date"],
        end_date=cfg["training"]["test_end_date"],
    )
    close = ohlcv["close"].astype(float).values
    rets = np.diff(close) / close[:-1]
    return rets.astype(float)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    t0 = time.time()
    out_md = PROJECT_ROOT / "results" / "verification" / "statistical_tests.md"
    out_json = out_md.with_suffix(".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    strat = _strategy_returns()
    bh = _benchmark_buy_and_hold()

    # Equal-length window for the test (last min(len) bars)
    n = int(min(len(strat), len(bh)))
    if n < 50:
        logger.error("Not enough samples for statistical tests (n=%d)", n)
        return 1
    strat = strat[-n:]
    bh = bh[-n:]
    zero = np.zeros_like(strat)  # risk-free 0 return benchmark

    report = comprehensive_report(
        strategy_returns=strat,
        benchmarks={"buy_and_hold": bh, "zero_rf": zero},
        n_boot=2000,
        block=24,
        confidence=0.95,
        seed=42,
    )
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Statistical Robustness Report",
        "",
        f"_Generated in {time.time() - t0:.1f}s, n={n} hourly returns_",
        "",
        "## Bootstrap 95% Confidence Intervals (block=24h, 2000 resamples)",
        "",
        "| Metric | Estimate | 95% CI |",
        "|--------|---------:|:------:|",
        f"| Sharpe (annualized) | {report['bootstrap']['sharpe']['estimate']:.3f} | "
        f"[{report['bootstrap']['sharpe']['lower']:.3f}, {report['bootstrap']['sharpe']['upper']:.3f}] |",
        f"| Cumulative return | {report['bootstrap']['cumulative_return']['estimate']:.4f} | "
        f"[{report['bootstrap']['cumulative_return']['lower']:.4f}, "
        f"{report['bootstrap']['cumulative_return']['upper']:.4f}] |",
        "",
        "## Ledoit-Wolf Sharpe equality test (vs. benchmarks) + Bonferroni",
        "",
        "| Benchmark | z | raw p | Bonferroni p |",
        "|-----------|--:|------:|-------------:|",
    ]
    for name, d in report["ledoit_wolf_vs_benchmarks"].items():
        md.append(
            f"| {name} | {d['z']:+.3f} | {d['p_value']:.4g} | {d['p_value_bonferroni']:.4g} |"
        )

    md += [
        "",
        "## Interpretation",
        "",
        "* A 95% bootstrap CI for the annualized Sharpe that excludes 0 ",
        "  rejects the null of zero risk-adjusted return.",
        "* Significant (Bonferroni-adjusted) p-values vs. benchmarks ",
        "  indicate the system outperforms beyond chance, while accounting ",
        "  for the multiple comparisons inflation.",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")
    logger.info("Wrote %s", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
