# Editor Extension Request — ESWA-D-26-08980

**Manuscript ID:** ESWA-D-26-08980
**Title:** A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes
**Authors:** Sanghoon Heo, Youngbae Hwang
**Decision under appeal:** Major Revision
**Subject of this letter:** Request for an extension to the revision deadline + disclosure of substantive findings discovered during the revision.

---

Dear Professor [Editor-in-Chief / Handling Editor],

Thank you again for the constructive Major Revision decision on
manuscript ESWA-D-26-08980 and for the rigorous comments from all three
reviewers. We have spent the past several weeks working through every
one of the eighteen mapped review items, and we are writing to make two
related requests.

## 1. Request for a deadline extension

We respectfully request an extension of the revision deadline by
**90 days**.

During the revision we discovered that the manuscript's previously
reported Table 2 results were not reproducible under the time-series
methodology Reviewer #3 correctly demanded (FinBERT pre-2020 sentiment,
forward-looking Trend-Scanning labels, walk-forward expanding-window
cross-validation, ATR-scaled dynamic slippage). Replacing each
component in turn led us to identify three concrete reasons:

1. **The original "ground truth" regime labels were lagging
   (SMA-50)** and are a deterministic function of the same past prices
   that the technical features see. The 90% classifier accuracy in the
   submitted draft therefore reflects an information-leak artefact
   rather than genuine prediction. Switching to Trend-Scanning labels
   collapses classifier accuracy to ~46% (see new Table 1 in the
   revised manuscript).

2. **A post-processing layer in our code base** (`config.paper_alignment`)
   was inverting policy actions, blending the buy-and-hold trajectory
   into the strategy returns, scaling positions ×1.76, and capping
   Sharpe ratio at the target value before metrics were reported. We
   have permanently disabled this layer; the revised Table 2 reports
   the raw policy's measurements with bootstrap 95% CIs and a
   Bonferroni multiple-comparison correction.

3. **The PPO reward functions** (Bull = Sharpe, Bear = Sortino,
   Sideways = Bull−5·tx) do not, in our hourly setting, lead to
   regime-specialised behaviour: even with the lagging labels giving
   ~90% classifier accuracy on Fold 1, the Bear-pool ensemble lost
   −53% during the 2022 LUNA collapse while passive Buy-and-Hold lost
   −44% on the same window.

We believe that adequately addressing these findings — by re-framing
the manuscript's contribution and, where time permits, partially
re-engineering the architecture (sequence-model regime classifier,
direction-aligned reward shaping, lower-frequency bars) — requires
roughly **90 days of focused work**. The first 30 days will be spent
on manuscript re-writing; the remaining 60 days on architectural
re-engineering and re-running the walk-forward benchmarks.

We are happy to provide a more detailed week-by-week plan upon request.

## 2. Disclosure of substantive findings already in hand

In the interest of full transparency to the editorial team — and so
that the requested extension can be evaluated against concrete
deliverables already in place — we summarise below what is already
complete and publicly available in our GitHub repository
(https://github.com/heosanghun/2_ESWARegime).

### 2.1 Reviewer #3 methodology overhauls (complete)

- **FinBERT (ProsusAI/finbert, 2019)** replaces the post-2020 LLM.
  31,012 news articles re-scored; see
  `data/cryptonews_finbert_2021-10-12_2023-12-19.csv` and
  `scripts/regenerate_news_sentiment_finbert.py`.
- **Trend-Scanning forward labels** replace SMA-50 lagging labels;
  see `src/regime/trend_scanning.py`.
- **Walk-Forward Expanding-Window CV** (5 folds, paper §4.1 schedule);
  see `src/validation/walk_forward_cv.py` and
  `scripts/run_walk_forward.py`.
- **ATR-scaled dynamic slippage** replaces the fixed 0.02% slippage;
  see `src/backtest/slippage.py`.
- **Class-imbalance correction** in the XGBoost regime classifier
  (inverse-frequency sample weights); see
  `src/regime/regime_classifier.py`.

### 2.2 Reviewer #1 / #2 / #4 items (complete)

- Bootstrap CI + Bonferroni multiple-comparison correction;
  see `results/walk_forward/statistical_tests.md`.
- Computational complexity / latency / memory measurements;
  see `results/verification/computational_complexity.md`.
- Code reproducibility: the entire revision pipeline runs from a
  single command (`python scripts/run_walk_forward.py`).

### 2.3 New honest Table 2 and Table 3

Walk-Forward 5-fold means (raw, no `paper_alignment`):

| Metric | Original Table 2 claim | Revised Table 2 (fold mean) | 95% CI |
|--------|---:|---:|---|
| Sharpe Ratio | 1.89 | −20.50 | [−23.98, −16.12] |
| Cumulative Return | 0.893 | −0.737 | [−0.863, −0.586] |
| Max Drawdown | −0.162 | −0.738 | [−0.863, −0.589] |
| Win Rate | 0.678 | 0.101 | [0.045, 0.174] |
| Profit Factor | 2.34 | 0.308 | [0.199, 0.420] |

For all six metrics the original Table 2 value lies *outside* the
Bonferroni-corrected 95% confidence interval, indicating that the
original numbers cannot be obtained from the methodology described
in the paper without the post-processing artefacts above.

## 3. What the extension would deliver

During the 90-day extension we plan to:

1. **Re-frame the manuscript's contribution** from "we propose a
   regime-aware ensemble that outperforms baselines" to "we
   identify and quantify a class of look-ahead-bias failure modes
   in regime-aware reinforcement-learning trading systems, and
   propose architectural changes that address them." We believe
   this re-framing increases the paper's scientific value rather
   than diminishing it.

2. **Re-engineer the regime classifier** to use a sequence model
   (GRU or small Transformer) operating on a 96-bar rolling window
   of *normalised* technical features. Our preliminary analysis
   suggests this should partially close the gap on forward-looking
   labels.

3. **Re-design the reward functions** to include a direction-aligned
   bonus term, so the Bear-pool actively profits from confirmed
   bear sub-windows rather than passively avoiding them.

4. **Re-run walk-forward and ablations** on the re-engineered system
   and report honest numbers.

## 4. Closing

We deeply regret that the original submission contained a
post-processing layer that was not disclosed in the methodology
section. Removing it and reporting the honest measurements was the
only acceptable course of action once we were aware of the issue.
We would like to thank the reviewers — Reviewer #3 in particular —
whose insistence on time-series-safe methodology and code disclosure
caused us to find and correct these issues before publication.

We hope that the editor will grant the requested 90-day extension so
that we can submit a revision that is methodologically sound and
scientifically honest, and that we believe will be of greater
long-term value to the *Expert Systems with Applications* readership
than the original draft.

Sincerely,

**Sanghoon Heo**, *corresponding author*
**Youngbae Hwang**, *advisor*

---

_Attachments referenced in this letter:_

- `doc/AUTONOMOUS_OVERNIGHT_REPORT.md` — detailed honest reproduction
- `doc/TECHNICAL_RECOMMENDATIONS.md` — re-engineering plan
- `results/walk_forward/` — per-fold backtests, CIs, classifier metrics
- `results/walk_forward_sma/label_method_comparison.md` — labelling diagnostic
- GitHub repository: https://github.com/heosanghun/2_ESWARegime
