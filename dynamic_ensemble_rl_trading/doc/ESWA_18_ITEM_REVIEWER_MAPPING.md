# ESWA-D-26-08980 — 18 Reviewer Items Mapping

**Repository:** https://github.com/heosanghun/2_ESWARegime  
**Decision:** Major Revision  
**Updated:** 2026-05-27 (honest-measurement audit pass)

This document maps all 18 reviewer items to code paths and documentation artefacts.

---

## Category 1 — Methodology (6 items)

| # | Item | Artefact | Status |
|---|------|----------|--------|
| 1 | Look-ahead bias (LLM → FinBERT) | `src/data/finbert_sentiment.py`, `scripts/regenerate_news_sentiment_finbert.py` | Done |
| 2 | Time-series CV (walk-forward) | `src/validation/walk_forward_cv.py`, `scripts/run_walk_forward.py` | Done |
| 3 | Forward-looking labels (Trend Scanning) | `src/regime/trend_scanning.py`, `src/regime/ground_truth.py` | Done |
| 4 | ResNet-18 domain gap | `src/ablation/no_visual_features.py`, `doc/manuscript_revisions/section4_ablation_visual.md` | Done |
| 5 | Model theory / notation | `doc/manuscript_revisions/section3_methodology.md`, `section3_notation.md` | Done |
| 6 | Hard regime rigidity | `doc/manuscript_revisions/section3_hard_regime.md` | Done |

## Category 2 — Experiments (7 items)

| # | Item | Artefact | Status |
|---|------|----------|--------|
| 7 | Unrealistic slippage | `src/backtest/slippage.py`, `config.training.slippage_model: atr` | Done |
| 8 | Market frictions | `doc/manuscript_revisions/section4_slippage_and_frictions.md` | Done |
| 9 | Statistical tests (Bootstrap + Bonferroni) | `src/evaluation/statistical_tests.py`, `results/audit/s1_statistical_tests.json` | Done |
| 10 | Overfitting / regularization | `doc/manuscript_revisions/section4_overfitting.md` | Done |
| 11 | Generalization limits | `doc/manuscript_revisions/section4_generalization.md`, `doc/oos_2024_forward_report.md` | Done |
| 12 | Latency / complexity | `scripts/measure_computational_complexity.py`, `results/verification/computational_complexity.md` | Done |
| 13 | News / LLM sentiment limits | `src/ablation/no_news.py`, `scripts/run_ablation_no_news.py` | Done |

## Category 3 — Manuscript structure (3 items)

| # | Item | Artefact | Status |
|---|------|----------|--------|
| 14 | Introduction / conclusion / implications | `doc/manuscript_revisions/section1_intro_revisions.md` | Done |
| 15 | Notation consistency | `doc/manuscript_revisions/section3_notation.md` | Done |
| 16 | Related work + Figure 1 | `doc/manuscript_revisions/related_work_additions.md`, `figure1_redesign.md` | Done |

## Category 4 — Citation requests (2 items)

| # | Item | Artefact | Status |
|---|------|----------|--------|
| 17 | Decline unrelated citations (R#2) | `doc/Response_Letter_v2_english.md` §2 | Done |
| 18 | Decline bearing RUL citation (R#4) | `doc/Response_Letter_v2_english.md` §4 | Done |

## Category 5 — Reproducibility (1 item)

| # | Item | Artefact | Status |
|---|------|----------|--------|
| 19 | Reproducibility statement | `doc/Reproducibility_Statement.md`, `reproduce.py`, `.github/workflows/audit_ci.yml` | Done |

---

## Honest measurement summary (May 2026 audit)

> **Important:** Early verification runs used `paper_alignment` post-processing and reported ~100% Table 2 match. Those numbers are **invalid** under honest methodology. Current headline results:

| Artefact | Key result |
|----------|------------|
| 5-fold walk-forward (S1) | Mean Sharpe **−12.76**, paper value outside Bonferroni CI |
| ATR 1.8% screen | Mean Sharpe **−2.30**; MDD only metric overlapping paper CI |
| 2022 bear (ATR, deterministic) | MDD **−24.0%** vs B&H **−63.3%** |
| OOS 2024 (ATR) | Sharpe **+1.96**, MDD **−1.9%** vs B&H **−32.3%** |
| Latency (CPU, end-to-end) | median ~5 ms per bar |
| Memory (full pipeline) | ~891 MB RSS |

See `results/audit/`, `doc/gap_decomposition_refined.md`, and `doc/Response_Letter_v2_english.md` §0.

---

## Priority checklist

| Priority | Scope | Status |
|----------|-------|--------|
| P1 | Methodology fixes + slippage + statistics + reproducibility | Done |
| P2 | Latency, ablations, manuscript supplements | Done |
| P3 | External English proofreading | Pending |

All 18 reviewer items plus reproducibility (#19) have code or documentation artefacts. Remaining work is manuscript text integration and external copy-editing.
