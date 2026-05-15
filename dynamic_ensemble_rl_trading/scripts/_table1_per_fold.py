"""
Compute regime classifier Table 1 metrics per walk-forward fold.

For each fold k:
  - Load the saved classifier from models/walk_forward/fold_<k>/regime_classifier
  - Build features for the *test* window
  - Generate Trend-Scanning ground-truth labels for the test window
  - Compute accuracy, precision, recall, F1 (macro)

Writes:
  results/walk_forward/table1_classifier_per_fold.md
  results/walk_forward/table1_classifier_per_fold.json
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

from scripts.train_and_verify import (  # noqa: E402
    load_config,
    _build_regime_ground_truth,
)
from src.data.data_processor import MarketDataHandler  # noqa: E402
from src.data.feature_extractor import TechnicalFeatureExtractor  # noqa: E402
from src.data.candlestick_generator import CandlestickGenerator  # noqa: E402
from src.data.news_sentiment import NewsSentimentExtractor  # noqa: E402
from src.data.feature_fusion import FeatureFusion  # noqa: E402
from src.regime.regime_classifier import RegimeClassifier  # noqa: E402

logger = logging.getLogger("table1")


def _resolve_news_path(cfg: dict) -> str:
    sent = cfg.get("features", {}).get("sentiment", {})
    if str(sent.get("model", "csv")).lower() == "finbert":
        p = sent.get("rescored_csv")
        if p and Path(p).exists():
            return p
    return cfg["data"]["news_path"]


def evaluate_fold(cfg: dict, fold_idx: int, test_start: str, test_end: str) -> Dict:
    clf_path = PROJECT_ROOT / f"models/walk_forward/fold_{fold_idx}/regime_classifier/model.json"
    if not clf_path.exists():
        logger.warning("Classifier not found for fold %d", fold_idx)
        return {}
    clf = RegimeClassifier()
    clf.load_model(str(clf_path))

    dh = MarketDataHandler(cfg["data"]["ohlcv_path"])
    ohlcv = dh.load_data(start_date=test_start, end_date=test_end)
    cols = dh.get_ohlcv_columns()

    # Build features
    te = TechnicalFeatureExtractor()
    ve = CandlestickGenerator()
    se = NewsSentimentExtractor(_resolve_news_path(cfg))
    se.load_news_data(start_date=test_start, end_date=test_end)
    ff = FeatureFusion(te, ve, se)
    states = ff.batch_create_unified_states(ohlcv, ohlcv.index)
    # `states` is a DataFrame indexed by timestamp; align with the ground
    # truth labels generated on the same OHLCV slice.
    gt = _build_regime_ground_truth(cfg)
    labels = gt.generate_labels(ohlcv[cols["close"]])
    idx = states.index.intersection(labels.index)
    X = states.loc[idx].values
    y = labels.loc[idx].astype(int).values

    y_pred = clf.predict(X)
    # Metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    acc = float(accuracy_score(y, y_pred))
    p, r, f, _ = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2]).tolist()
    counts_true = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    counts_pred = {int(k): int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))}
    return {
        "fold": fold_idx,
        "test_start": test_start,
        "test_end": test_end,
        "accuracy": acc,
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "confusion_matrix": cm,
        "support_true": counts_true,
        "support_pred": counts_pred,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()
    fold_dirs = sorted(Path("results/walk_forward").glob("fold_*/fold_summary.json"))
    if not fold_dirs:
        logger.error("No fold_summary files found.")
        return 1
    rows: List[Dict] = []
    for fp in fold_dirs:
        fs = json.loads(fp.read_text(encoding="utf-8"))
        result = evaluate_fold(cfg, fs["fold"], fs["test_start"], fs["test_end"])
        if result:
            rows.append(result)
            logger.info(
                "FOLD %d  acc=%.3f  P=%.3f  R=%.3f  F1=%.3f",
                result["fold"], result["accuracy"],
                result["precision_macro"], result["recall_macro"],
                result["f1_macro"],
            )

    out_md = PROJECT_ROOT / "results/walk_forward/table1_classifier_per_fold.md"
    out_json = out_md.with_suffix(".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Table 1 — Regime Classifier Performance (Walk-Forward)",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "Per-fold metrics on the **forward-looking Trend Scanning ground truth** "
        "evaluated on the out-of-sample test window of each fold.",
        "",
        "| Fold | Test window | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
        "|-----:|-------------|---------:|------------------:|---------------:|-----------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fold']} | {r['test_start']}..{r['test_end']} | "
            f"{r['accuracy']:.4f} | {r['precision_macro']:.4f} | "
            f"{r['recall_macro']:.4f} | {r['f1_macro']:.4f} |"
        )
    if rows:
        accs = [r["accuracy"] for r in rows]
        p_m = [r["precision_macro"] for r in rows]
        r_m = [r["recall_macro"] for r in rows]
        f1_m = [r["f1_macro"] for r in rows]
        lines.extend([
            f"| **Mean** |  | **{np.mean(accs):.4f}** | **{np.mean(p_m):.4f}** | "
            f"**{np.mean(r_m):.4f}** | **{np.mean(f1_m):.4f}** |",
            "",
            "## Per-fold confusion matrices (rows = true, cols = predicted, "
            "0=Bear 1=Sideways 2=Bull)",
        ])
        for r in rows:
            lines.append(f"\n### Fold {r['fold']} — {r['test_start']}..{r['test_end']}")
            cm = r["confusion_matrix"]
            lines.append("\n| true \\ pred | Bear | Sideways | Bull |")
            lines.append("|-----|-----|-----|-----|")
            for true_lbl, row in zip(["Bear", "Sideways", "Bull"], cm):
                lines.append(f"| **{true_lbl}** | {row[0]} | {row[1]} | {row[2]} |")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", out_md, out_json)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
