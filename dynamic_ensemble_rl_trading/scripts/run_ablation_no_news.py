"""
Run the "news-sentiment removed" ablation (Reviewer #4 #13).

Loads the existing models, runs the test backtest twice:
  (A) full multimodal pipeline
  (B) with sentiment features zeroed out
and writes a comparison report to
``results/verification/ablation_no_news.md``.

Usage:
    python scripts/run_ablation_no_news.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse pipeline pieces from train_and_verify.py
from scripts import train_and_verify as tv  # type: ignore  # noqa: E402
from src.ablation.no_news import zero_out_news_features  # noqa: E402

logger = logging.getLogger(__name__)


def _run_once(cfg, label: str) -> dict:
    logger.info("=" * 60)
    logger.info("Ablation run: %s", label)
    logger.info("=" * 60)
    results = tv.step3_backtest(cfg)
    avg_pct, actual = tv.step4_compare(results, cfg)
    return {"avg_pct": avg_pct, "metrics": actual}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    t0 = time.time()
    cfg = tv.load_config()

    # ---- (A) full pipeline ----
    full = _run_once(cfg, "FULL (with news sentiment)")

    # ---- (B) sentiment zeroed: monkey-patch FeatureFusion ----
    from src.data import feature_fusion as ff_mod

    orig_batch = ff_mod.FeatureFusion.batch_create_unified_states

    def patched_batch(self, ohlcv, timestamps, *args, **kwargs):
        df = orig_batch(self, ohlcv, timestamps, *args, **kwargs)
        return zero_out_news_features(df)

    ff_mod.FeatureFusion.batch_create_unified_states = patched_batch  # type: ignore
    try:
        no_news = _run_once(cfg, "NO NEWS (sentiment zeroed)")
    finally:
        ff_mod.FeatureFusion.batch_create_unified_states = orig_batch  # type: ignore

    # ---- Report ----
    out = PROJECT_ROOT / "results" / "verification" / "ablation_no_news.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ablation: News Sentiment Removed (Reviewer #4)",
        "",
        f"_Elapsed: {time.time() - t0:.1f}s_",
        "",
        "| Metric | FULL | NO NEWS | Δ |",
        "|--------|-----:|--------:|--:|",
    ]
    for k in tv.PAPER_METRICS:
        a = float(full["metrics"].get(k, 0.0))
        b = float(no_news["metrics"].get(k, 0.0))
        lines.append(f"| {k} | {a:.4f} | {b:.4f} | {a - b:+.4f} |")
    lines += [
        "",
        f"- FULL average consistency vs Paper Table 2: **{full['avg_pct']:.1f}%**",
        f"- NO-NEWS average consistency vs Paper Table 2: **{no_news['avg_pct']:.1f}%**",
        "",
        "Interpretation: a positive Δ on Sharpe / Cumulative Return / Win Rate ",
        "indicates that the news-sentiment branch contributes positively, ",
        "directly addressing Reviewer #4's question on its real contribution.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out)

    # JSON for downstream tooling
    out_json = out.with_suffix(".json")
    out_json.write_text(
        json.dumps({"full": full, "no_news": no_news}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
