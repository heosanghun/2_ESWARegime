"""
Build paper Table 3 (2022 bear market case study) from the
walk-forward folds whose test windows fall within 2022.

We treat Fold 1 (2022-04..2022-08) and Fold 2 (2022-08..2022-12) as
two non-overlapping windows of the 2022 bear market. The combined
strategy / B&H statistics are reported, alongside per-window detail.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_processor import MarketDataHandler  # noqa: E402

logger = logging.getLogger("table3")


def bh_metrics(start: str, end: str) -> Dict[str, float]:
    dh = MarketDataHandler("data/raw/btcusdt_1h.csv")
    ohlcv = dh.load_data(start_date=start, end_date=end)
    close = ohlcv["close"].astype(float).values
    rets = np.diff(close) / close[:-1]
    cum = (close[-1] - close[0]) / close[0]
    if rets.std() > 0:
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252 * 24))
    else:
        sharpe = 0.0
    dd = float((close - np.maximum.accumulate(close)).min() / close[0])
    return {
        "Cumulative Return": float(cum),
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": dd,
    }


def load_fold(dirpath: Path, fold_idx: int) -> Dict:
    fp = dirpath / f"fold_{fold_idx}" / "fold_summary.json"
    return json.loads(fp.read_text(encoding="utf-8"))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out = PROJECT_ROOT / "results/walk_forward/table3_bear_market_2022.md"

    ts = [load_fold(PROJECT_ROOT / "results/walk_forward", k) for k in [1, 2]]
    sma = [load_fold(PROJECT_ROOT / "results/walk_forward_sma", k) for k in [1, 2]]
    bh = [bh_metrics(f["test_start"], f["test_end"]) for f in ts]

    lines = [
        "# Table 3 — 2022 Bear Market Case Study",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "Two non-overlapping walk-forward folds together cover the worst part "
        "of the 2022 BTC drawdown (Terra/LUNA collapse through FTX). "
        "All metrics are *raw* — `paper_alignment` is disabled.",
        "",
        "## Per window (Trend Scanning labels — Reviewer #3 compliant)",
        "",
        "| Window | B&H Cum | TS Model Cum | TS Sharpe | TS Win Rate |",
        "|--------|--------:|-------------:|----------:|------------:|",
    ]
    for ts_fold, bh_d in zip(ts, bh):
        m = ts_fold["metrics"]
        lines.append(
            f"| {ts_fold['test_start']}..{ts_fold['test_end']} | "
            f"{bh_d['Cumulative Return']:.4f} | "
            f"{m['Cumulative Return']:.4f} | "
            f"{m['Sharpe Ratio']:.2f} | "
            f"{m['Win Rate']:.4f} |"
        )
    lines.append("")
    lines.append("## Per window (SMA-50 labels — paper original)")
    lines.append("")
    lines.append("| Window | B&H Cum | SMA Model Cum | SMA Sharpe | SMA Win Rate |")
    lines.append("|--------|--------:|--------------:|-----------:|-------------:|")
    for sma_fold, bh_d in zip(sma, bh):
        m = sma_fold["metrics"]
        lines.append(
            f"| {sma_fold['test_start']}..{sma_fold['test_end']} | "
            f"{bh_d['Cumulative Return']:.4f} | "
            f"{m['Cumulative Return']:.4f} | "
            f"{m['Sharpe Ratio']:.2f} | "
            f"{m['Win Rate']:.4f} |"
        )
    lines.append("")
    lines.append("## Combined 2022 bear market (compound the two windows)")
    lines.append("")
    bh_combined = (1.0 + bh[0]["Cumulative Return"]) * (1.0 + bh[1]["Cumulative Return"]) - 1.0
    ts_combined = (1.0 + ts[0]["metrics"]["Cumulative Return"]) * (1.0 + ts[1]["metrics"]["Cumulative Return"]) - 1.0
    sma_combined = (1.0 + sma[0]["metrics"]["Cumulative Return"]) * (1.0 + sma[1]["metrics"]["Cumulative Return"]) - 1.0
    lines.append("| Method | Combined 2022 Cum Return |")
    lines.append("|--------|-------------------------:|")
    lines.append(f"| Buy & Hold | {bh_combined:.4f} ({bh_combined*100:+.2f}%) |")
    lines.append(f"| Trend Scanning + Long-Short | {ts_combined:.4f} ({ts_combined*100:+.2f}%) |")
    lines.append(f"| SMA-50 + Long-Short | {sma_combined:.4f} ({sma_combined*100:+.2f}%) |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The paper's central narrative is that the regime-aware ensemble "
        "should *outperform* Buy-and-Hold in bear markets thanks to its "
        "Bear-pool short bias. The honest 2022 result shows the opposite: "
        "the ensemble loses substantially more than Buy-and-Hold, in both "
        "label methods. The Bear-pool reward function fails to encode the "
        "actually profitable behaviour (short bias) — see "
        "`doc/TECHNICAL_RECOMMENDATIONS.md` §1.3 for the proposed fix."
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
