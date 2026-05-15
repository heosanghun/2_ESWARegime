"""
Post-hoc diagnostics for walk-forward fold runs.

For each fold:
  - Read trading_history (recorded via step3 ⇒ fold_summary JSON does NOT
    persist it, but we can re-run a fast backtest using the saved models).
  - For now we work off the saved fold_summary.json plus aggregate metrics
    that already exist, and write a compact diagnostic markdown that flags
    obvious failure modes (e.g., regime classifier always Bull / always
    Bear, action distribution skew, sign of return vs sign of B&H).

Usage
-----
    python scripts/_diagnose_walk_forward.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_processor import MarketDataHandler  # noqa: E402

logger = logging.getLogger("wf_diag")


def buy_and_hold_return(start: str, end: str) -> float:
    dh = MarketDataHandler("data/raw/btcusdt_1h.csv")
    ohlcv = dh.load_data(start_date=start, end_date=end)
    if len(ohlcv) < 2:
        return 0.0
    return (ohlcv["close"].iloc[-1] - ohlcv["close"].iloc[0]) / ohlcv["close"].iloc[0]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    wf_dir = PROJECT_ROOT / "results" / "walk_forward"
    folds: List[Dict] = []
    for fp in sorted(wf_dir.glob("fold_*/fold_summary.json")):
        try:
            folds.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", fp, exc)
    if not folds:
        logger.error("No fold_summary.json found under %s", wf_dir)
        return 1

    rows = []
    for f in folds:
        bh = buy_and_hold_return(f["test_start"], f["test_end"])
        model_ret = f["metrics"].get("Cumulative Return", 0.0)
        # "Aligned with market direction" if signs agree (or both ~0).
        aligned = "OK" if np.sign(bh) == np.sign(model_ret) or abs(bh) < 0.02 else "WRONG-SIGN"
        excess = model_ret - bh
        rows.append({
            "fold": f["fold"],
            "test_period": f"{f['test_start']}..{f['test_end']}",
            "B&H return": bh,
            "Model return": model_ret,
            "Excess": excess,
            "Sharpe": f["metrics"].get("Sharpe Ratio", 0.0),
            "Win Rate": f["metrics"].get("Win Rate", 0.0),
            "Direction": aligned,
            "Consistency %": f.get("avg_consistency_pct", 0.0),
        })
    df = pd.DataFrame(rows)

    out_path = wf_dir / "diagnostics.md"
    lines = [
        "# Walk-Forward Diagnostics",
        "",
        f"_Generated from {len(folds)} fold(s)_",
        "",
        "## Per-fold vs Buy & Hold",
        "",
        df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Aggregate",
        "",
        f"- Mean B&H return: **{df['B&H return'].mean()*100:+.2f}%**",
        f"- Mean Model return: **{df['Model return'].mean()*100:+.2f}%**",
        f"- Mean Excess (Model - B&H): **{df['Excess'].mean()*100:+.2f}%**",
        f"- Folds in correct direction: {(df['Direction']=='OK').sum()} / {len(df)}",
        f"- Mean Sharpe: {df['Sharpe'].mean():.2f}",
        f"- Mean consistency: {df['Consistency %'].mean():.1f}%",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Diagnostics written to %s", out_path)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
