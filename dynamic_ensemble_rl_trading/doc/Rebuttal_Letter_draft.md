# Rebuttal Letter (Draft)

**Manuscript ID:** ESWA-D-26-08980  
**Title:** Market Regime-aware Trading via Dynamic Ensemble Reinforcement Learning  
**Decision:** Major Revision  
**Authors:** Sanghoon Heo / Advisor: Youngbae Hwang  
**Repository:** https://github.com/heosanghun/2_ESWARegime  
**Drafted:** 2026-05-13

---

Dear Editor and Reviewers,

We sincerely thank the editor and three reviewers for the constructive
and detailed feedback. The criticisms raised — particularly around
look-ahead bias, time-series cross-validation, lagging ground-truth
labels, slippage realism, and statistical robustness — were
methodologically substantive and have meaningfully improved the
manuscript. Below we respond to every point in the reviewers'
comments. Code changes accompany every response and are pushed to the
public repository above; the file paths referenced in each response
correspond to that repository.

A summary mapping of all 18 mapped items is given at the end of this
letter.

---

## Response to Reviewer #3

### #3.1 — Look-ahead bias from LLM sentiment (item #1)

**Action taken.** We removed the 2025-released LLM completely from
the sentiment pipeline. News headlines are now re-scored by
**FinBERT (`ProsusAI/finbert`, released 2019-08)** which strictly
predates the 2021-2023 backtest window. The implementation is in
`src/data/finbert_sentiment.py` and is exposed through
`scripts/regenerate_news_sentiment_finbert.py`. The
`config/config.yaml` exposes `features.sentiment.model: finbert`. A
fallback to the legacy CSV is documented and logged so that researchers
without GPU access can still reproduce the pipeline structure.

### #3.2 — Time-series cross-validation (item #2)

**Action taken.** We replaced standard K-Fold with two time-series-safe
splitters implemented in `src/validation/walk_forward_cv.py`:

* `WalkForwardExpandingCV(n_splits=5, test_size=0.1, gap=0)` — anchored
  expanding-window walk-forward;
* `PurgedKFold(n_splits=5, embargo=0.01)` — López de Prado's purged
  K-fold with an embargo period to remove leakage from overlapping
  labels.

Hyper-parameter selection for the regime classifier now uses these
splitters via `tune_regime_classifier(...)`. `config/config.yaml`
exposes `validation.cv_method: walk_forward`.

### #3.3 — Lagging ground-truth labels (item #3)

**Action taken.** We replaced the SMA-50 normalised slope with the
**Trend Scanning** algorithm (López de Prado, 2018, *AFML*, Ch.5),
which scans forward horizons \(L\in[5,20]\) and selects the horizon
with the largest \(|t\text{-value}|\) of the OLS slope. The
implementation is in `src/regime/trend_scanning.py`; the legacy SMA
method is kept under `method='sma'` for ablation purposes. The
`config/config.yaml` exposes `regime.label_method: trend_scanning`.

### #3.4 — ResNet-18 domain gap (item #4)

**Action taken.** A "Visual-removed" ablation already exists
(`src/ablation/no_visual_features.py`) and we re-execute it in the
revised pipeline. We additionally added a *Domain Adaptation*
paragraph in §5 explaining our freeze-and-fine-tune protocol — see
`doc/manuscript_revisions/section4_ablation_visual.md`.

### #3.5 — Unrealistic slippage (item #7)

**Action taken.** Fixed 0.02 % slippage is replaced by an **ATR-scaled
dynamic model** (`src/backtest/slippage.py:ATRSlippageModel`):
\(\mathrm{slip}_t = \mathrm{clip}(b + \kappa\cdot ATR_{14}(t)/p_t, s_{\min}, s_{\max})\)
with \(b=0.0002, \kappa=0.5, s_{\max}=0.005\). A
**conservative fixed 0.10 %** model is also provided as the upper-bound
sanity check the reviewer requested. The backtester
(`src/backtest/backtester.py`) consumes a time-indexed series of
per-bar rates, so the comparison is exact.

### #3.6 — Cross-asset generalisation (item #11)

**Action taken.** We added an explicit *Generalisation Discussion*
documenting tick-size, lot-size, funding-rate, calendar, and book-depth
differences that single-volatility scaling cannot absorb. The
manuscript now frames the result as a *framework demonstrated on
BTC/USDT* and outlines a four-step adaptation protocol — see
`doc/manuscript_revisions/section4_generalization.md`.

---

## Response to Reviewer #1

### #1.1 — Mathematical foundations of multimodal fusion (item #5)

**Action taken.** A new subsection §3.0–§3.3 formalises the
frequency-domain decomposition assumption, the contiguous-block
masking strategy, and the PPO clipped surrogate. See
`doc/manuscript_revisions/section3_methodology.md`.

### #1.2 — Notation consistency (item #15)

**Action taken.** A complete notation table is added in §3 head,
unifying \(s_t, r_t, R_t, \pi_\theta, \theta_{\text{conf}}, w_k\) and
giving explicit formulae for Accuracy/Precision/Recall/F1. See
`doc/manuscript_revisions/section3_notation.md`.

### #1.3 — Intro / Conclusion / Managerial implications (item #14)

**Action taken.** A four-point contribution paragraph, a quantitative
summary, a new *Practical / Managerial Implications* subsection, and a
strengthened conclusion are provided in
`doc/manuscript_revisions/section1_intro_revisions.md`.

### #1.4 — Literature updating, Figure 1, English editing (item #16)

**Action taken.**

* A ten-reference *Related Work Additions* set covering 2018–2024
  literature is provided in
  `doc/manuscript_revisions/related_work_additions.md`.
* A redrawn Figure 1 brief (resolution, lanes, equations, captions) is
  provided in `doc/manuscript_revisions/figure1_redesign.md`.
* The revised manuscript will be sent for professional English
  editing (Editage) before resubmission.

### #1.5 — Latency / complexity (item #12)

**Action taken.** `scripts/measure_computational_complexity.py`
reports CPU latency for the regime classifier, PPO agent, and the
end-to-end regime switch, plus memory footprint and training cost.
Numbers are written to `results/verification/computational_complexity.md`.

---

## Response to Reviewer #2

### #2.1 — Hard regime assignment (item #6)

**Action taken.** We clarified two mechanisms already present
(confidence-gated switching, within-regime soft mixing) and listed a
full fuzzy-assignment variant as future work. See
`doc/manuscript_revisions/section3_hard_regime.md`.

### #2.2 — Sensitivity, confidence intervals, hypothesis tests (item #9)

**Action taken.** `scripts/run_statistical_tests.py` produces:

* Stationary-bootstrap 95 % CIs for Sharpe and cumulative return,
* A Ledoit–Wolf Sharpe-equality test against Buy-and-Hold and zero,
* Bonferroni correction over the two benchmark comparisons.

See `results/verification/statistical_tests.md` and
`doc/manuscript_revisions/section4_statistical_tests.md`.

### #2.3 — Overfitting & regularization (item #10)

**Action taken.** §4 documents Walk-Forward CV, Purged K-Fold,
PPO entropy bonus, clip range, weight decay \(10^{-4}\), seed-diverse
agents, and dynamic weighting as model averaging. See
`doc/manuscript_revisions/section4_overfitting.md`.

### #2.4 — Unrelated citation requests (item #17)

**Response — politely declined.** We have carefully reviewed the four
suggested papers (Adeleke 2023; Adeleke 2025; Hammed 2025; Joshua &
Kim 2025). They concern smart-city operations, battery-AI maintenance,
and digital-twin systems. While methodologically interesting, they do
not share scope with our financial time-series reinforcement-learning
study, and including them would not strengthen the manuscript's
argument. We respectfully decline to cite these works.

---

## Response to Reviewer #4

### #4.1 — Market frictions, partial fills, market impact (item #8)

**Action taken.** A dedicated subsection now enumerates market impact,
partial fills, liquidity caps, and venue heterogeneity as explicit
limitations. See
`doc/manuscript_revisions/section4_slippage_and_frictions.md`.

### #4.2 — LLM sentiment limitations (item #13)

**Action taken.** Beyond replacing the LLM with FinBERT (item #1), we
add a *News-removed ablation* (`src/ablation/no_news.py`) and an
execution script (`scripts/run_ablation_no_news.py`). The resulting
report `results/verification/ablation_no_news.md` shows the marginal
contribution of the sentiment branch on every paper-table metric.

### #4.3 — Bearing remaining-useful-life citation (item #18)

**Response — politely declined.** The suggested citation
("Estimation of remaining useful life of bearings…", DOI
10.1007/s40998-018-0108-y) is a mechanical-engineering reliability
study without methodological overlap with financial time-series
reinforcement learning. We respectfully decline this citation
request.

---

## Reproducibility (item #19)

**Action taken.** A dedicated reproducibility statement is added at
`doc/Reproducibility_Statement.md` and linked from `README.md`. It
specifies data sources, random seeds, package versions, system
specifications, and a DOI placeholder to be filled at acceptance.

---

## Summary table — 18 mapped items

| # | Category | Item | Status |
|---|----------|------|--------|
| 1 | Methodology | LLM look-ahead bias | Addressed (FinBERT code) |
| 2 | Methodology | Time-series CV | Addressed (`walk_forward_cv.py`) |
| 3 | Methodology | Lagging labels | Addressed (Trend Scanning) |
| 4 | Methodology | ResNet domain gap | Ablation + DA paragraph |
| 5 | Methodology | Math foundations | New §3.0-§3.3 |
| 6 | Methodology | Hard regime | Clarified + Future Work |
| 7 | Experiments | Slippage realism | ATR + conservative model |
| 8 | Experiments | Market frictions | Limitations §4 |
| 9 | Experiments | Stat. robustness | Bootstrap+LW+Bonferroni |
| 10 | Experiments | Overfitting | New §4 paragraph |
| 11 | Experiments | Generalisation | New Discussion subsection |
| 12 | Experiments | Latency/complexity | Auto-generated report |
| 13 | Experiments | LLM limitations | No-news ablation |
| 14 | Structure | Intro / Conclusion | New paragraphs |
| 15 | Structure | Notation | New table |
| 16 | Structure | Lit + Figure + Editing | Brief + 10 refs + Editage |
| 17 | Citation | Unrelated #2 refs | Declined |
| 18 | Citation | Bearing-RUL ref | Declined |
| 19 | Reproducibility | Statement & artefacts | Released |

We thank the reviewers for the thorough engagement and look forward to
their further comments.

Sincerely,  
Sanghoon Heo, Youngbae Hwang
