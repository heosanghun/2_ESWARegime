"""Classifier SHAP audit — quantify visual / sentiment / technical
feature contributions to the regime classifier (Reviewer #3 #4 item #3).

Loads the trained XGBoost regime classifier for fold_5 of
``walk_forward_reward_v2`` (= the model the OOS 2024 test uses), runs SHAP
on a representative test slice, and aggregates absolute SHAP values into
three feature groups:

  - technical : 19-dim hand-engineered indicators
  - visual    : 512-dim ResNet-18 candlestick image embedding
  - sentiment : 8-dim FinBERT aggregate news features

The output deliverable is a small table that lets Reviewer #3 (who flagged
the ResNet candlestick branch as a likely look-ahead / noise contributor)
read off the percentage SHAP contribution of each branch.

Output:
  results/audit/shap_audit/shap_summary.json
  results/audit/shap_audit/shap_summary.md
  results/audit/shap_audit/per_feature_top.csv
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

logger = logging.getLogger("shap_audit")

# Fold 5 of walk_forward_reward_v2 — the most recently trained classifier
# and the one used as the OOS 2024 forward-test reference model.
SOURCE_FOLD = 5
MODEL_DIR = ROOT / "models" / "walk_forward_reward_v2" / f"fold_{SOURCE_FOLD}" / "regime_classifier"
TEST_START = "2023-08-19"  # fold 5 test window
TEST_END = "2023-12-19"

OUT_DIR = ROOT / "results" / "audit" / "shap_audit"


def feature_groups(dim_total: int) -> List[Tuple[str, int, int]]:
    """Return [(group_name, start_idx, end_idx)] non-overlapping ranges.

    Schema mirrors src/data/feature_fusion.py: technical(19) + visual(512)
    + sentiment(8) = 539. We accept dim_total == 27 (use_visual=False) and
    dim_total == 539 (default) and fall back to a 'unknown' bucket if
    something else is encountered.
    """
    if dim_total == 539:
        return [
            ("technical", 0, 19),
            ("visual", 19, 19 + 512),
            ("sentiment", 19 + 512, 539),
        ]
    if dim_total == 27:
        return [
            ("technical", 0, 19),
            ("sentiment", 19, 27),
        ]
    logger.warning("Unknown feature dim=%d; falling back to all-technical bucket.", dim_total)
    return [("unknown", 0, dim_total)]


def build_unified_state_slice(cfg: dict) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    from src.data.data_processor import MarketDataHandler
    from src.data.feature_extractor import TechnicalFeatureExtractor
    from src.data.candlestick_generator import CandlestickGenerator
    from src.data.news_sentiment import NewsSentimentExtractor
    from src.data.feature_fusion import FeatureFusion
    from scripts.train_and_verify import _resolve_news_path

    dh = MarketDataHandler(cfg["data"]["ohlcv_path"])
    ohlcv = dh.load_data(start_date=TEST_START, end_date=TEST_END)
    te = TechnicalFeatureExtractor()
    ve = CandlestickGenerator()
    se = NewsSentimentExtractor(_resolve_news_path(cfg))
    se.load_news_data(start_date=TEST_START, end_date=TEST_END)
    use_visual = cfg.get("features", {}).get("use_visual", True)
    ff = FeatureFusion(te, ve, se, use_visual=use_visual)
    states = ff.batch_create_unified_states(ohlcv, ohlcv.index)
    return np.asarray(states, dtype=float), ohlcv.index


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_DIR.exists():
        logger.error("Regime classifier model directory missing: %s", MODEL_DIR)
        return 1

    try:
        import shap  # type: ignore
    except ImportError:
        logger.error(
            "shap is not installed. Install with: pip install shap. "
            "Falling back to gain-based feature importance only."
        )
        shap = None  # type: ignore

    os.environ["ESWA_RAW_MODE"] = "1"
    from scripts.train_and_verify import load_config, _build_regime_classifier
    cfg = load_config()

    logger.info("Loading regime classifier from %s", MODEL_DIR)
    rc = _build_regime_classifier(cfg)
    rc.load_model(str(MODEL_DIR / "model.json"))
    rc.reset_smoothing()

    logger.info("Constructing feature matrix for SHAP slice (%s..%s)", TEST_START, TEST_END)
    X, ts_index = build_unified_state_slice(cfg)
    dim_total = X.shape[1]
    logger.info("Feature matrix: %d samples x %d features", X.shape[0], dim_total)

    # Use a stratified subsample for SHAP — 500 is enough for stable
    # group aggregates and fits comfortably in RAM.
    rng = np.random.default_rng(seed=42)
    n_sample = min(500, X.shape[0])
    sample_idx = rng.choice(X.shape[0], size=n_sample, replace=False)
    X_sample = X[sample_idx]
    logger.info("SHAP sample size: %d", n_sample)

    groups = feature_groups(dim_total)
    payload: Dict = {
        "source_fold": SOURCE_FOLD,
        "test_window": [TEST_START, TEST_END],
        "feature_dim": int(dim_total),
        "groups": [{"name": g, "start": s, "end": e, "size": e - s} for g, s, e in groups],
        "n_sample": n_sample,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    # ── Gain-based importance (always available) ─────────────────────
    try:
        booster = rc.model.get_booster()
        gain = booster.get_score(importance_type="gain")  # {'f12': 3.4, ...}
        # Convert to dense array
        per_feat_gain = np.zeros(dim_total)
        for k, v in gain.items():
            idx = int(k.lstrip("f"))
            if 0 <= idx < dim_total:
                per_feat_gain[idx] = float(v)
        total_gain = float(per_feat_gain.sum()) or 1.0
        group_gain = {}
        for g, s, e in groups:
            grp_total = float(per_feat_gain[s:e].sum())
            group_gain[g] = {
                "sum_gain": grp_total,
                "pct_of_total": grp_total / total_gain,
                "mean_per_feature": grp_total / max(e - s, 1),
            }
        payload["gain_importance"] = {
            "group_contributions": group_gain,
            "total_gain": total_gain,
        }
        logger.info("Gain-based group contributions:")
        for g, v in group_gain.items():
            logger.info("  %-12s : sum=%.2f  pct=%.2f%%", g, v["sum_gain"], 100 * v["pct_of_total"])
    except Exception as e:
        logger.warning("Could not compute gain importance: %s", e)

    # ── SHAP-based (preferred when available) ────────────────────────
    if shap is not None:
        try:
            logger.info("Running TreeExplainer on XGBoost classifier ...")
            explainer = shap.TreeExplainer(rc.model)
            shap_values = explainer.shap_values(X_sample)
            # SHAP version differences: list-of-arrays (older) or single
            # ndarray with class dimension (newer ≥0.40). Normalise to
            # exactly 2D (n_sample, n_feat) by averaging absolute values
            # across whichever axis represents the class.
            sv = shap_values
            if isinstance(sv, list):
                stacked = np.stack(sv, axis=0)  # (n_class, n_sample, n_feat)
                abs_per_sample_feat = np.mean(np.abs(stacked), axis=0)
            else:
                arr = np.asarray(sv)
                if arr.ndim == 3:
                    # Could be (n_sample, n_feat, n_class) or (n_class, n_sample, n_feat)
                    if arr.shape[0] == X_sample.shape[0]:
                        abs_per_sample_feat = np.mean(np.abs(arr), axis=-1)
                    else:
                        abs_per_sample_feat = np.mean(np.abs(arr), axis=0)
                else:
                    abs_per_sample_feat = np.abs(arr)
            mean_abs_per_feat = np.asarray(np.mean(abs_per_sample_feat, axis=0)).reshape(-1)
            total_abs = float(mean_abs_per_feat.sum()) or 1.0
            shap_group = {}
            for g, s, e in groups:
                grp_total = float(mean_abs_per_feat[s:e].sum())
                shap_group[g] = {
                    "sum_mean_abs_shap": grp_total,
                    "pct_of_total": grp_total / total_abs,
                    "mean_per_feature": grp_total / max(e - s, 1),
                }
            payload["shap_importance"] = {
                "group_contributions": shap_group,
                "total_mean_abs_shap": total_abs,
            }
            logger.info("Mean-|SHAP| group contributions:")
            for g, v in shap_group.items():
                logger.info("  %-12s : sum=%.4f  pct=%.2f%%", g, v["sum_mean_abs_shap"], 100 * v["pct_of_total"])

            # Per-feature top contributors as a CSV
            ranking = np.argsort(mean_abs_per_feat)[::-1][:50]
            rows = []
            for rank, idx in enumerate(ranking):
                ii = int(idx)
                group_name = next((g for g, s, e in groups if s <= ii < e), "unknown")
                rows.append({
                    "rank": rank + 1,
                    "feature_index": ii,
                    "group": group_name,
                    "mean_abs_shap": float(mean_abs_per_feat[ii]),
                    "pct_of_total": float(mean_abs_per_feat[ii] / total_abs),
                })
            pd.DataFrame(rows).to_csv(OUT_DIR / "per_feature_top.csv", index=False)
            logger.info("Wrote per-feature top contributors → %s", OUT_DIR / "per_feature_top.csv")
        except Exception as e:
            logger.error("SHAP computation failed: %s", e, exc_info=True)
            payload.setdefault("shap_importance", {})["error"] = str(e)

    out_json = OUT_DIR / "shap_summary.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown
    md = [
        "# Classifier SHAP Audit (Reviewer #3 item #4)",
        "",
        f"_Generated: {payload['generated']}_  ",
        f"**Source model:** `models/walk_forward_reward_v2/fold_{SOURCE_FOLD}/regime_classifier/model.json`",
        f"**Sample slice:** {TEST_START} → {TEST_END} ({n_sample} subsampled bars)",
        f"**Feature dim:** {dim_total}  (technical 19 + visual 512 + sentiment 8 by default)",
        "",
        "## Feature-group contributions (mean |SHAP|)",
        "",
    ]
    if "shap_importance" in payload and "group_contributions" in payload["shap_importance"]:
        md.append("| Group | Size | Sum mean |SHAP| | % of total | Mean / feature |")
        md.append("|---|---:|---:|---:|---:|")
        for g, v in payload["shap_importance"]["group_contributions"].items():
            md.append(
                f"| {g} | {next((e-s for gn,s,e in groups if gn==g), '?')} | "
                f"{v['sum_mean_abs_shap']:.4f} | {100*v['pct_of_total']:.2f}% | "
                f"{v['mean_per_feature']:.4e} |"
            )
        md.append("")
    if "gain_importance" in payload:
        md.append("## Feature-group contributions (XGBoost gain)")
        md.append("")
        md.append("| Group | Sum gain | % of total | Mean / feature |")
        md.append("|---|---:|---:|---:|")
        for g, v in payload["gain_importance"]["group_contributions"].items():
            md.append(
                f"| {g} | {v['sum_gain']:.2f} | {100*v['pct_of_total']:.2f}% | "
                f"{v['mean_per_feature']:.4e} |"
            )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "Reviewer #3 raised concerns about the ResNet-18 candlestick image "
        "branch contributing noise rather than predictive signal. The table "
        "above quantifies how the trained XGBoost classifier uses each "
        "feature group:"
    )
    md.append("")
    md.append(
        "* If the **visual** share is large (e.g. > 50%) AND the classifier "
        "still has only marginal-above-chance accuracy (≈ 46% on Trend-"
        "Scanning labels, 33% chance), the classifier is consuming a large "
        "amount of *capacity* on the 512-D ResNet embedding without converting "
        "that capacity into discriminative power. This is consistent with "
        "Reviewer #3's intuition that the visual branch contributes mostly "
        "noise: the model fits to it heavily during training but the resulting "
        "decision boundary does not generalise."
    )
    md.append("")
    md.append(
        "* If the **technical** share is small but the model ablation in "
        "Section 4 of the manuscript shows that *removing* the visual branch "
        "does not materially change test accuracy, this is direct evidence "
        "that the 19-D technical features carry the same information more "
        "compactly. Recommended action: report the SHAP share as a "
        "complement to the ablation accuracy table, framing the visual "
        "branch as **redundant capacity rather than zero contribution**."
    )
    (OUT_DIR / "shap_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("Wrote %s and %s", out_json, OUT_DIR / "shap_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
