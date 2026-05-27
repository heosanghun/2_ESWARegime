# ESWA-D-26-08980 — Reviewer Navigation Index

**Repository:** https://github.com/heosanghun/2_ESWARegime  
**Manuscript ID:** ESWA-D-26-08980  
**One-command reproduction:** `python reproduce.py`

This index lists the **minimum set of artefacts** intended for ESWA reviewers. Internal meeting notes, draft PDFs, and experimental phase reports are **not** included in this repository.

---

## Start here (5 minutes)

| Order | Document | Purpose |
|------:|----------|---------|
| 1 | [`README.md`](../README.md) — Honesty Statement | `paper_alignment` disclosure + honest Table 2 numbers |
| 2 | [`doc/Response_Letter_v2_english.md`](Response_Letter_v2_english.md) | Point-by-point reviewer responses (§0 disclosure first) |
| 3 | [`reproduce.py`](../reproduce.py) | Reproduce headline audit numbers on your machine |
| 4 | [`.github/workflows/audit_ci.yml`](../.github/workflows/audit_ci.yml) | CI contract: `paper_alignment` must stay OFF |

---

## Audit evidence (headline numbers)

| Claim | Artefact |
|-------|----------|
| Paper Table 2 not reproducible under honest methodology | [`results/audit/s1_statistical_tests.json`](../results/audit/s1_statistical_tests.json) |
| Bonferroni-corrected bootstrap CIs | [`results/audit/s1_statistical_tests.md`](../results/audit/s1_statistical_tests.md) |
| ATR 1.8% screen (best honest config) | [`results/audit/atr_screen_statistical_tests.json`](../results/audit/atr_screen_statistical_tests.json) |
| 2022 bear window (Sortino / Calmar / CVaR / MDD) | [`results/audit/bear_window_2022/advanced_metrics_deterministic.json`](../results/audit/bear_window_2022/advanced_metrics_deterministic.json) |
| OOS 2024 forward test (6-month gap) | [`results/audit/oos_2024_forward/advanced_metrics.json`](../results/audit/oos_2024_forward/advanced_metrics.json) |
| Classifier SHAP (Reviewer #3 visual branch) | [`results/audit/shap_audit/shap_summary.json`](../results/audit/shap_audit/shap_summary.json) |
| Gap decomposition (paper +1.89 → honest −20.5) | [`doc/gap_decomposition_refined.md`](gap_decomposition_refined.md) |
| `paper_alignment` timeline | [`doc/audit_paper_alignment_timeline.md`](audit_paper_alignment_timeline.md) |

---

## Code locations (Reviewer #3 / #4)

| Mechanism | File |
|-----------|------|
| `paper_alignment` config (disabled) | [`config/config.yaml`](../config/config.yaml) — `paper_alignment:` block |
| Raw-metrics / honest backtest | [`scripts/train_and_verify.py`](../scripts/train_and_verify.py) — `ESWA_RAW_MODE=1` |
| Deprecated metric optimizer (hard-guarded) | [`scripts/reach_100_percent_autonomous.py`](../scripts/reach_100_percent_autonomous.py) |
| FinBERT sentiment (look-ahead fix) | [`src/data/finbert_sentiment.py`](../src/data/finbert_sentiment.py) |
| Trend Scanning labels | [`src/regime/trend_scanning.py`](../src/regime/trend_scanning.py) |
| Walk-forward CV | [`src/validation/walk_forward_cv.py`](../src/validation/walk_forward_cv.py) |
| ATR 1.8% sideways filter | [`src/regime/atr_sideways_filter.py`](../src/regime/atr_sideways_filter.py) |
| Backtester v2.0.1 long-short fix | [`src/backtest/backtester.py`](../src/backtest/backtester.py) |

---

## Reproduction commands

```bash
# Full audit bundle (CI + bear window + OOS 2024; requires data download for OOS)
python reproduce.py

# Individual blocks
python reproduce.py --only ci      # Bootstrap CIs only (~1 min)
python reproduce.py --only bear    # 2022 bear advanced metrics (~15 min)
python reproduce.py --only oos --download-data   # OOS 2024 (~20 min)
```

Data: OHLCV via `scripts/download_hourly_data.py`; news CSV via paper Google Drive or `scripts/regenerate_news_sentiment_finbert.py`.

---

## Manuscript revision support

| Document | Use |
|----------|-----|
| [`doc/Manuscript_Revision_Guide.md`](Manuscript_Revision_Guide.md) | Paste-ready revised sections (audit + risk-management application) |
| [`doc/oos_2024_forward_report.md`](oos_2024_forward_report.md) | OOS 2024 narrative + Table 7 |
| [`doc/Rebuttal_Letter_v2_honest.md`](Rebuttal_Letter_v2_honest.md) | Korean rebuttal (same content as English letter) |
| [`CHANGELOG.md`](../CHANGELOG.md) | v2.0.0 `paper_alignment` disclosure; v2.0.1 Backtester fix |

---

## What is intentionally excluded

See [`doc/GITHUB_업로드_제외_목록.md`](GITHUB_업로드_제외_목록.md). Large CSV data, trained model weights, internal meeting notes, manuscript PDFs, and experimental phase-2 ablation logs are excluded to keep the repository focused on **reproducible audit evidence**.
