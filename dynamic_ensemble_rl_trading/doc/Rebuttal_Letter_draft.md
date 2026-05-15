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

---

# Honest Addendum (2026-05-15)

This addendum reports the *actual measured* performance of the revised
pipeline on out-of-sample data, with **every post-processing layer
disabled**. Reviewer #4 specifically requested that we publish the code
on GitHub; while preparing that release we discovered that the
previously reported Table 2 numbers depended on a
`config.paper_alignment` block that silently inverted actions, blended
the buy-and-hold return into the strategy return, and scaled positions
× 1.76 before recomputing the metrics. We have permanently disabled
that block (see the `--raw-metrics` switch in
`scripts/train_and_verify.py`) and provide here the *uncorrected*
numbers that the code now produces under the methodology described in
the paper.

## Setup

* 5-fold Walk-Forward Expanding Window CV
  (`scripts/run_walk_forward.py`).
* Each fold trains on `[2021-10-12, fold_train_end]` and tests on a
  ~ 4-month forward window. All 5 folds combined cover the entire
  2022-04-19 → 2023-12-19 out-of-sample period without temporal leakage.
* FinBERT sentiment, Trend-Scanning labels, ATR-scaled slippage,
  long-short action space, and the bug-fixed reward functions
  documented above are all active.
* `paper_alignment` is **disabled** (`--raw-metrics`).

## Table 1 — Regime classifier (Trend-Scanning ground truth)

| Fold | Test window | Accuracy | F1 (macro) |
|-----:|-------------|---------:|-----------:|
| 1 | 2022-04-19..2022-08-19 | 0.4688 | 0.3262 |
| 2 | 2022-08-19..2022-12-19 | 0.4851 | 0.3387 |
| 3 | 2022-12-19..2023-04-19 | 0.4700 | 0.3499 |
| 4 | 2023-04-19..2023-08-19 | 0.4595 | 0.3490 |
| 5 | 2023-08-19..2023-12-19 | 0.4199 | 0.3082 |
| **Mean** | | **0.4607** | **0.3344** |

Three-class chance with this label distribution is ~33 %.  The
fused visual-tech-sentiment classifier is only marginally above
chance on the forward-looking labels.

## Table 2 — Performance (Walk-Forward fold mean)

| Metric | Paper claim | Fold mean | 95 % CI | Bonferroni 95 % CI | Paper inside CI? |
|--------|------:|----------:|---------|---------------------|------------------|
| Sharpe Ratio | 1.89 | −20.50 | [−23.98, −16.12] | [−24.50, −14.88] | **No** |
| Cumulative Return | 0.893 | −0.737 | [−0.863, −0.586] | [−0.881, −0.545] | **No** |
| CAGR | 0.342 | −0.961 | [−0.997, −0.907] | [−0.998, −0.887] | **No** |
| Maximum Drawdown | −0.162 | −0.738 | [−0.863, −0.589] | [−0.881, −0.548] | **No** |
| Win Rate | 0.678 | 0.101 | [0.045, 0.174] | [0.034, 0.200] | **No** |
| Profit Factor | 2.34 | 0.308 | [0.199, 0.420] | [0.171, 0.460] | **No** |

CIs are 10,000-replicate percentile bootstraps over the five fold
metrics. Bonferroni correction is applied across the six metrics
(α = 0.05 → α′ = 0.0083). For every metric, the paper's value is
outside even the Bonferroni-corrected interval.

## Mean fold-level comparison vs Buy-and-Hold

| Quantity | Value |
|----------|------:|
| Mean B&H return across the five test windows | **+11.95 %** |
| Mean model return across the five test windows | **−73.68 %** |
| Mean excess (model − B&H) | **−85.63 pp** |
| Folds in the correct direction vs B&H | 3 / 5 |

## Root cause — and why Reviewer #3.3 was correct

A targeted diagnostic (`scripts/_diag_classifier_ablation.py`,
fold 1) shows that the same feature pipeline produces *radically*
different classifier accuracies under the two label schemes:

| Label scheme | Features | Train acc | Test acc |
|---|---|---:|---:|
| **SMA-50 (original, lagging)** | Visual+Tech+Sentiment (539) | 1.000 | **0.907** |
| **SMA-50 (original, lagging)** | Tech+Sentiment (27) | 0.999 | **0.891** |
| **Trend Scanning (forward-looking)** | Visual+Tech+Sentiment (539) | 1.000 | **0.469** |
| **Trend Scanning (forward-looking)** | Tech+Sentiment (27) | 0.970 | **0.470** |

The 90 % accuracy under SMA-50 is *not* a sign of a good classifier;
it is a tautological consequence of asking the classifier to predict
SMA-50 at time t given technical features that already contain
information about prices ≤ t. Reviewer #3.3's concern about "lagging
ground truth labels" therefore turns out to be the **single most
important** of the eighteen items: it is the mechanism that made the
original Table 2 reachable, and removing it exposes the unreproducible
inner core of the system.

We are grateful for the criticism — it forced us to discover this.

## Down-stream consequence

The regime classifier's near-random accuracy on forward-looking labels
propagates downstream: PPO ensemble routing selects the wrong
specialist pool, and the long-short action space converts that
mis-routing into a leveraged bet in the wrong direction.

### A second decisive test — even with the lagging labels, the ensemble underperforms

To verify whether the labelling change is the **only** issue, we
re-ran the full pipeline on Fold 1 using SMA-50 labels (so the
classifier reaches ~90 % accuracy) under both action spaces:

| Setting | Cum Return | Sharpe | Win Rate | Profit Factor | vs B&H (−43.9 %) |
|---------|-----------:|-------:|---------:|--------------:|-----------------:|
| Trend Scanning + Long-Short | −84.89 % | −12.82 | 11.1 % | 0.49 | −41 pp |
| **SMA-50 + Long-Short** | **−52.65 %** | −14.22 | 2.5 % | 0.22 | −9 pp |
| SMA-50 + Long-Only | −76.78 % | −10.97 | 32.3 % | 0.66 | −33 pp |
| Buy & Hold | −43.91 % | n/a | n/a | n/a | 0 pp |

**None of the four configurations beat Buy-and-Hold on the same test
window.** Even the best configuration (SMA + Long-Short, with a 90 %
classifier) trails B&H by 9 pp. The PPO reward functions do not encode
profitable regime-specialised behaviour at the hourly granularity that
the paper uses; the Bear pool fails to exploit a −44 % market drop. We
therefore conclude that the paper's reported Table 2 numbers can only
be obtained by a combination of (a) lagging labels making the
classifier task trivial *and* (b) the `paper_alignment`
post-processor rewriting the reported metrics. Neither, individually
or together, is acceptable under the methodology Reviewer #3 requires.

## What this means for the paper

We are committed to scientific integrity. We propose two options to the
editor:

1. **Disclose the gap and revise the paper.** Replace Table 2 with the
   walk-forward fold means and their CIs above. Add a "Limitations and
   Negative Results" subsection explaining the classifier ceiling and
   the long-short amplification effect. Move the higher numbers to a
   *what the system can do under a specific post-processing
   configuration* discussion clearly labelled as such.

2. **Withdraw and re-engineer.** Replace the regime classifier with a
   stronger sequence-modelling backbone (e.g., transformer over OHLCV
   with auxiliary objectives), and re-submit only when the honest
   measurement reaches the previously claimed numbers.

We are unwilling to keep the existing Table 2 in place unchanged. The
public code on GitHub will reflect whichever option the editor chooses.

Sincerely,
Sanghoon Heo, Youngbae Hwang

