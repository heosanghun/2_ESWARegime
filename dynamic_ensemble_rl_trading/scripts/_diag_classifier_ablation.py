"""
Quick classifier ablation:
  (A) Full features (visual 512 + tech 19 + sentiment 8 = 539)
  (B) Tech + Sentiment only (27)
  (C) Tech only (19)
  (D) Simple rolling-return baseline (1-feature: sign of past 24h return)

For each subset we report in-sample AND out-of-sample accuracy on
Fold 1 of the walk-forward schedule, to disentangle "the classifier
overfits" from "the labels are unlearnable from these features".
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_and_verify import (  # noqa: E402
    load_config,
    _build_regime_ground_truth,
    _resolve_news_path,
)
from src.data.data_processor import MarketDataHandler  # noqa: E402
from src.data.feature_extractor import TechnicalFeatureExtractor  # noqa: E402
from src.data.candlestick_generator import CandlestickGenerator  # noqa: E402
from src.data.news_sentiment import NewsSentimentExtractor  # noqa: E402
from src.data.feature_fusion import FeatureFusion  # noqa: E402
from src.regime.regime_classifier import RegimeClassifier  # noqa: E402

logger = logging.getLogger("clf_diag")


def build_window(cfg: dict, start: str, end: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    dh = MarketDataHandler(cfg["data"]["ohlcv_path"])
    ohlcv = dh.load_data(start_date=start, end_date=end)
    te = TechnicalFeatureExtractor()
    ve = CandlestickGenerator()
    se = NewsSentimentExtractor(_resolve_news_path(cfg))
    se.load_news_data(start_date=start, end_date=end)
    ff = FeatureFusion(te, ve, se)
    states = ff.batch_create_unified_states(ohlcv, ohlcv.index)
    gt = _build_regime_ground_truth(cfg)
    labels = gt.generate_labels(ohlcv[dh.get_ohlcv_columns()["close"]])
    idx = states.index.intersection(labels.index)
    return states.loc[idx], labels.loc[idx], ohlcv.loc[idx]


def slice_features(X: np.ndarray, name: str) -> np.ndarray:
    # Layout based on src/data/feature_fusion.py: [visual(512), tech(19), sentiment(8)] = 539
    if name == "full":
        return X
    if name == "tech_senti":
        return X[:, 512:]
    if name == "tech":
        return X[:, 512:512 + 19]
    if name == "rolling_ret":
        # 1-feature baseline: 24-bar return computed from OHLCV close
        return None
    raise ValueError(name)


def rolling_return_baseline(ohlcv: pd.DataFrame, window: int = 24) -> np.ndarray:
    close = ohlcv["close"].astype(float).values
    r = np.zeros_like(close)
    for i in range(len(close)):
        if i - window >= 0:
            r[i] = (close[i] - close[i - window]) / close[i - window]
    return r.reshape(-1, 1)


def fit_eval(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
) -> Dict[str, float]:
    clf = RegimeClassifier(n_estimators=200, max_depth=5, confidence_threshold=0.35)
    clf.fit(X_tr, y_tr)
    p_tr = clf.predict(X_tr)
    p_te = clf.predict(X_te)
    return {
        "train_acc": float((p_tr == y_tr).mean()),
        "test_acc": float((p_te == y_te).mean()),
    }


def build_window_sma(cfg: dict, start: str, end: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Same as build_window but uses lagging SMA labels."""
    cfg_sma = {k: (v if k != "regime" else dict(v)) for k, v in cfg.items()}
    cfg_sma["regime"] = dict(cfg["regime"])
    cfg_sma["regime"]["label_method"] = "sma"
    dh = MarketDataHandler(cfg["data"]["ohlcv_path"])
    ohlcv = dh.load_data(start_date=start, end_date=end)
    te = TechnicalFeatureExtractor()
    ve = CandlestickGenerator()
    se = NewsSentimentExtractor(_resolve_news_path(cfg))
    se.load_news_data(start_date=start, end_date=end)
    ff = FeatureFusion(te, ve, se)
    states = ff.batch_create_unified_states(ohlcv, ohlcv.index)
    gt = _build_regime_ground_truth(cfg_sma)
    labels = gt.generate_labels(ohlcv[dh.get_ohlcv_columns()["close"]])
    idx = states.index.intersection(labels.index)
    return states.loc[idx], labels.loc[idx], ohlcv.loc[idx]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config()

    logger.info("Building fold-1 train window 2021-10-12..2022-04-19 (Trend Scanning)")
    X_tr_df, y_tr_s, ohlcv_tr = build_window(cfg, "2021-10-12", "2022-04-19")
    logger.info("Building fold-1 test window 2022-04-19..2022-08-19 (Trend Scanning)")
    X_te_df, y_te_s, ohlcv_te = build_window(cfg, "2022-04-19", "2022-08-19")

    X_tr_full = X_tr_df.values.astype(float)
    X_te_full = X_te_df.values.astype(float)
    y_tr = y_tr_s.astype(int).values
    y_te = y_te_s.astype(int).values

    results = {}
    for name in ["full", "tech_senti", "tech"]:
        Xtr = slice_features(X_tr_full, name)
        Xte = slice_features(X_te_full, name)
        r = fit_eval(Xtr, y_tr, Xte, y_te)
        results[name] = r
        logger.info("[%s]  train=%.3f  test=%.3f", name, r["train_acc"], r["test_acc"])

    # Rolling-return baseline.
    Xtr = rolling_return_baseline(ohlcv_tr)
    Xte = rolling_return_baseline(ohlcv_te)
    r = fit_eval(Xtr, y_tr, Xte, y_te)
    results["rolling_ret_24h"] = r
    logger.info("[rolling_ret_24h]  train=%.3f  test=%.3f", r["train_acc"], r["test_acc"])

    # ------------------------------------------------------------------
    # SMA-50 (lagging) labels, full features — does the easier label
    # scheme allow the classifier to generalise?
    # ------------------------------------------------------------------
    logger.info("\n=== SMA-50 labels (lagging baseline) ===")
    logger.info("Building fold-1 train window with SMA labels")
    X_tr_sma_df, y_tr_sma_s, _ = build_window_sma(cfg, "2021-10-12", "2022-04-19")
    logger.info("Building fold-1 test window with SMA labels")
    X_te_sma_df, y_te_sma_s, _ = build_window_sma(cfg, "2022-04-19", "2022-08-19")
    X_tr_sma = X_tr_sma_df.values.astype(float)
    X_te_sma = X_te_sma_df.values.astype(float)
    y_tr_sma = y_tr_sma_s.astype(int).values
    y_te_sma = y_te_sma_s.astype(int).values
    r_sma_full = fit_eval(X_tr_sma, y_tr_sma, X_te_sma, y_te_sma)
    results["sma_full"] = r_sma_full
    logger.info("[sma_full]  train=%.3f  test=%.3f",
                r_sma_full["train_acc"], r_sma_full["test_acc"])
    r_sma_ts = fit_eval(X_tr_sma[:, 512:], y_tr_sma, X_te_sma[:, 512:], y_te_sma)
    results["sma_tech_senti"] = r_sma_ts
    logger.info("[sma_tech_senti]  train=%.3f  test=%.3f",
                r_sma_ts["train_acc"], r_sma_ts["test_acc"])

    out = PROJECT_ROOT / "results/walk_forward/clf_feature_ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Classifier Feature Ablation — Fold 1", ""]
    lines.append("Two label schemes are tested:")
    lines.append("- **Trend Scanning** (forward-looking, the paper's revised method)")
    lines.append("- **SMA-50** (lagging, the original method)")
    lines.append("")
    lines.append("| Label scheme | Feature subset | Train acc | Test acc | Gap (overfit) |")
    lines.append("|---|---|---:|---:|---:|")
    for k, v in results.items():
        scheme = "SMA-50 (lagging)" if k.startswith("sma_") else "Trend Scanning (forward)"
        subset = k.replace("sma_", "") if k.startswith("sma_") else k
        lines.append(
            f"| {scheme} | {subset} | {v['train_acc']:.3f} | {v['test_acc']:.3f} | "
            f"{v['train_acc'] - v['test_acc']:.3f} |"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
