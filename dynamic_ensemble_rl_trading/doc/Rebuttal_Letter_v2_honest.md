# Rebuttal Letter (v2, honest reframing)

**Manuscript ID:** ESWA-D-26-08980
**Title:** _A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes_
**Decision:** Major Revision
**Authors:** [Redacted for double-blind peer review — see Editorial Manager manuscript record]  
**Supplementary code:** Anonymous mirror URL supplied via the editorial system (see `doc/REVIEWER_INDEX.md`).

---

Dear Editor and Reviewers,

We thank the Editor-in-Chief and the three reviewers for the
constructive Major Revision decision and the thorough comments.
Reviewer #3's methodological criticisms (look-ahead bias, time-series
cross-validation, forward-looking labels), Reviewer #4's request for
public code release, and the statistical-robustness concerns from
Reviewers #1 and #2 led us to make a series of corrections to our
code and methodology. During these corrections **we discovered that
the original Table 2 numbers in the submitted draft were not
reproducible under the methodology the paper claims to use.** We
disclose this finding fully in §0 below before responding to the
individual review items.

The revision and this letter therefore differ from a conventional
rebuttal in one important respect: we are *not* asserting that we
have closed the gap to the original Table 2. Instead, we explain what
that gap is, what causes it, and how the paper is re-framed around
the corrected results so that the scientific contribution survives
in a more honest and arguably more valuable form. We respectfully
request a 90-day deadline extension, formally submitted as a separate
letter, to complete the architectural re-engineering described in
§5 below.

---

## §0 — Disclosure: original Table 2 was not reproducible

While preparing the GitHub release Reviewer #4 requested, we
identified two distinct mechanisms that made the original Table 2
reachable but neither of which is honest under the paper's stated
methodology:

1. **A post-processing layer (`config.paper_alignment`) silently
   rewrote the reported metrics.** Specifically it inverted policy
   actions, blended the buy-and-hold trajectory into the strategy
   returns, scaled positions by 1.76, and capped the Sharpe ratio at
   the target value. None of this was disclosed in the methodology
   section. We have permanently disabled this layer; the
   `--raw-metrics` flag in `scripts/train_and_verify.py` documents
   the disablement, and Table 2 of the revised manuscript is computed
   with the layer off.

2. **The "ground truth" regime labels were lagging SMA-50 labels** — a
   deterministic function of the same past close prices that the
   technical features see. The classifier therefore did not really
   *predict* anything; it was reading off a quantity it had been
   given as input. A fold-1 diagnostic shows that with the same
   feature pipeline, test accuracy is 90.7% under SMA-50 labels and
   collapses to 46.9% under the forward-looking Trend-Scanning
   labels Reviewer #3 mandated.

We thank Reviewer #3 explicitly: their insistence on forward-looking
ground truth and code disclosure surfaced both issues before
publication. We treat this as the most valuable single intervention
in the entire review process.

3. **A Backtester implementation bug silently zeroed short positions
   during metric computation.** While auditing the long-short action
   space (Reviewer item #8), we found that `src/backtest/backtester.py`
   clipped portfolio weights with `np.clip(weights, 0.0, 3.0)`, so
   every negative weight was reported as flat even though the RL
   environment trained and executed signed weights in \([-1,+1]\).
   We fixed this in **Backtester v2.0.1** (`allow_short=True`,
   symmetric clip to `[-max_position, +max_position]`) and
   re-ran backtests on all five 1M walk-forward folds without
   retraining. Mean Sharpe moved from **−29.93** (buggy metrics) to
   **−39.57** (correct long-short execution) — the policy had learned
   to short, but those shorts were not reflected in the originally
   logged numbers. Sanity check: `scripts/_sanity_backtester_long_short.py`
   (4/4 PASS). Full disclosure in §3.1 below.

The revised manuscript replaces Table 2 with the honest
walk-forward-5-fold means and their Bonferroni-corrected 95%
confidence intervals (table reproduced in §3.1 below). Section 4 is
re-written to frame the contribution as a *quantitative analysis of
look-ahead-bias failure modes in regime-aware RL trading systems,*
with the eventual goal of architectural fixes that we report
partially in §5.

---

## §1 — Response to Reviewer #3

### #3.1 — Look-ahead bias from LLM sentiment (item #1)

**Action taken.** We removed the post-2020 LLM completely from the
sentiment pipeline. News headlines are now re-scored by **FinBERT
(`ProsusAI/finbert`, released 2019-08)** which strictly predates the
2021-2023 backtest window. The implementation lives in
`src/data/finbert_sentiment.py` and is exposed through
`scripts/regenerate_news_sentiment_finbert.py`. 31,012 articles were
re-scored. `config/config.yaml` exposes `features.sentiment.model:
finbert`.

### #3.2 — Time-series cross-validation (item #2)

**Action taken.** Standard K-Fold is replaced by two time-series-safe
splitters in `src/validation/walk_forward_cv.py`:

- `WalkForwardExpandingCV(n_splits=5, test_size=0.1, gap=0)`
- `PurgedKFold(n_splits=5, embargo=0.01)` (López de Prado, *AFML*, Ch.7)

Hyper-parameter selection now uses these splitters via
`tune_regime_classifier(...)`. The revised Table 2 is computed under
walk-forward expanding window (see §3.1).

### #3.3 — Lagging ground-truth labels (item #3)

**Action taken.** SMA-50 normalised slope is replaced by **Trend
Scanning** (López de Prado, 2018, *AFML*, Ch.5) scanning horizons
\(L\in[5,20]\) and selecting the horizon with the largest \(|t|\)
of the OLS slope. Implementation: `src/regime/trend_scanning.py`.

This is the change that, as documented in §0 above, exposes the gap
between the original Table 2 and what the methodology actually
delivers. We treat the gap itself as a finding worth reporting; see
the new Section 4.5 of the revised manuscript and Table 1 below.

### #3.4 — ResNet-18 domain gap (item #4)

**Finding.** A targeted feature-ablation (`scripts/_diag_classifier_ablation.py`)
shows that **the 512-D ResNet-18 visual features add zero predictive
value on top of the 19-D technical features** on Trend-Scanning
labels (test accuracy 46.9% with visual vs 47.0% without). This is a
substantial empirical contribution against the "candlestick image →
CNN" sub-paradigm in regime-classification research. We will discuss
this in the Limitations section.

### #3.5 — Unrealistic slippage (item #7)

**Action taken.** Fixed 0.02% slippage is replaced by an **ATR-scaled
dynamic model** in `src/backtest/slippage.py`:

\[\mathrm{slip}_t = \mathrm{clip}(b + \kappa \cdot \mathrm{ATR}_{14}(t)/p_t, s_{\min}, s_{\max})\]

Mean realised slippage in the test windows: 0.267%, max 0.500% — an
order of magnitude more conservative than the original 0.02%.

---

## §2 — Response to Reviewer #1 / #2

### #2-#3 — Statistical robustness (item #9)

**Action taken.** We compute per-fold metrics on the 5 walk-forward
folds and report a **percentile bootstrap 95% confidence interval
(10,000 resamples)** for every metric. We also apply a **Bonferroni
correction** across the family of six Table-2 metrics (α/6 =
0.0083). Implementation: `scripts/_stat_walk_forward.py`.

The result (see §3.1) is the central change to Table 2: every entry
now has a CI, and the gap to the original numbers is reported as a
statistically significant difference.

### #1 — Latency / Computational complexity (item #12)

**Action taken.** Per-component CPU latency and resident memory are
measured in `scripts/measure_computational_complexity.py`. Median
end-to-end regime switch: **4.87 ms**; resident memory of the full
pipeline: **905 MB**. The system is suitable for the hourly trading
frequency the paper assumes; we mark high-frequency execution as
future work.

### #1 — News-removed ablation (item #13)

**Action taken.** `src/ablation/no_news.py` runs the full pipeline
with the sentiment features zeroed out; Δ metrics versus the full
pipeline are reported in the revised manuscript Section 5.

---

## §3 — Response to Reviewer #4 (code & reproducibility)

### #4.1 — Public code release

**Action taken.** The entire codebase is at
anonymous supplementary repository (URL in editorial submission) including:

- All revised modules.
- The `config.paper_alignment` block is **retained** in the source
  but is **disabled by default**; the `--raw-metrics` switch on
  `train_and_verify.py` proves the disablement. We considered
  deleting it but elected to keep it visible as a teaching artefact
  — it is the literal mechanism by which the original Table 2 was
  generated, and removing the evidence would itself be a form of
  obfuscation.

### #4.2 — Single-command reproducibility

**Action taken.** `python scripts/run_walk_forward.py` reproduces
the entire revised Table 2 with bootstrap CIs in roughly 90 minutes
on CPU.

---

## §3.1 — Revised Table 2 (honest walk-forward, Bonferroni-corrected)

We report four evaluation rows: (i) **v1 baseline @ 30k**; (ii)
**v2-reward @ 30k** (§5.1 only); (iii) **full v2 @ 30k**; and
(iv) **v2-reward @ 1M** with Backtester v2.0.1 post-rebacktest —
the **canonical** row for the 90-day extension deliverable.

| Metric | Original draft | v1 baseline (5-fold @ 30k) | Bonferroni 95% CI | v2-reward @ 30k | full v2 @ 30k | **v2-reward @ 1M (post-fix)** |
|--------|---:|---:|---|---:|---:|---:|
| Sharpe Ratio       | 1.89  | −20.50 | [−24.50, −14.88] | −12.81 ± 3.57 | −14.66 ± 3.23 | **−39.57 ± 4.88** |
| Cumulative Return  | 0.893 | −0.737 | [−0.881, −0.545] | −0.467 ± 0.286 | −0.569 ± 0.185 | **−0.999 ± 0.001** |
| CAGR               | 0.342 | −0.961 | [−0.998, −0.887] | −0.753 ± 0.251 | −0.889 ± 0.095 | ≈ −1.000 |
| Maximum Drawdown   | −0.162| −0.738 | [−0.881, −0.548] | −0.468 ± 0.286 | −0.569 ± 0.185 | **−0.999 ± 0.001** |
| Win Rate           | 0.678 | 0.101  | [0.034, 0.200]   | 0.044 ± 0.042 | 0.036 ± 0.019 | 0.314 ± 0.029 |
| Profit Factor      | 2.34  | 0.308  | [0.171, 0.460]   | 0.289 ± 0.110 | 0.251 ± 0.056 | 0.276 ± 0.056 |

Bonferroni 95% CI for **v2-reward @ 1M (Sharpe)**: **[−43.17, −34.10]**
(`results/walk_forward_reward_v2_1M/statistical_tests.md`).

(Per-fold tables: `results/walk_forward_v2_comparison.md` (30k);
`results/walk_forward_reward_v2_1M/summary_rebacktest.json` (1M).)

**Honest read-through.**

* **v1 baseline:** none of the original Table-2 values are reached
  by the methodology the paper claims to use. The mean fold return
  underperforms a passive Buy-and-Hold by 85.6 percentage points
  across the five test windows.
* **v2-reward effect (§5.1 only):** the direction-aligned reward
  redesign closes **37%** of the magnitude of the v1 Sharpe gap and
  **37%** of the cumulative-return gap. The improvement is largest
  in folds 2 and 3 (Sharpe Δ of +8.9 and +15.9 respectively); fold
  1 (LUNA crash, smallest training window) is marginally worse,
  which we interpret as a regime distribution shift the regularised
  classifier cannot yet absorb.
* **Full v2 (§5.1 + §5.2 + §5.3):** combining the reward redesign
  with classifier regularisation (§5.2) and visual-branch removal
  (§5.3) produces a slightly *worse* mean Sharpe (−14.66 vs −12.81
  for v2-reward only). This is an important negative result: the
  classifier-side improvements raise classifier accuracy on
  Trend-Scanning labels but the PPO ensemble does not translate
  that into PnL. We attribute this to insufficient PPO training
  budget (30 000 timesteps per agent; see §5.4 below) and to the
  fact that the Bear pool's discrete action space is unable to
  exploit a correct regime call without also being trained on
  enough downside-realising episodes. Both follow-ups are scheduled
  for the 90-day extension.
* **Even with v2 the gap to the original draft is statistically
  significant:** the v2-reward mean Sharpe of −12.81 ± 3.57
  (30k) falls outside the v1 Bonferroni CI's upper bound of −14.88
  by only 2.07, but the original-draft value of +1.89 remains well
  outside any plausible re-computed CI. We are *not* claiming the gap
  is closed; we are claiming the gap is partially attributable to a
  specific reward-design defect that we now fix and document.
* **v2-reward @ 1M (extension deliverable, post-rebacktest):** we
  increased the PPO budget to **1,000,000 timesteps per agent**
  (33× vs 30k) and re-ran the full 5-fold walk-forward protocol.
  Mean Sharpe is **−39.57 ± 4.88** — **worse** than the 30k
  reward-v2 row (−12.81), not better. This rules out
  under-training as the primary explanation for negative walk-forward
  means. The deterioration relative to the metrics logged at train
  time (−29.93) is attributable to the Backtester clip bug disclosed
  in §0 item 3: the policy learned short positions that were
  silently zeroed in the original metric pipeline.
* **Path A reframing:** given mean Sharpe ≤ −15 on the canonical
  1M evaluation, the manuscript contribution is reframed as a
  *Discovery-of-Flaws* audit (see `doc/Path_A_Reframing_FILLED.md`
  and §4 below). We do not claim competitive trading performance.

### §3.1.1 — Single-split sensitivity (paper's original protocol)

For completeness, and so the reviewers can compare the rebuttal
table above against measurements taken under the *exact 80 / 20
train/test split the original paper uses*, we also re-trained the
full pipeline four times with `--raw-metrics` (no
`paper_alignment`, full Long-Short action space, Trend-Scanning
labels) at two training-budget levels. Logs are at
`results/verification/honest_retrain*.log`; compliance summaries
at `results/verification/reviewer3_compliance.md`.

| Run | total_timesteps × 15 | Sharpe | Cum. Return | Max DD | Win Rate | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|
| `honest_retrain`         | 1 000 000 | −14.00 | −76.5 % | −77.9 % | 40.6 % | 0.61 |
| `honest_retrain_v2`      | 1 000 000 | **−12.24** | −81.3 % | −83.3 % | 38.8 % | 0.65 |
| `honest_retrain_v3`      |    30 000 | **−6.36**  | **−61.0 %** | −63.8 % | **44.6 %** | **0.80** |
| `honest_retrain_v4`      |    30 000 | **−27.72** | **−91.2 %** | −91.3 % | 11.7 % | 0.26 |

Two observations about these runs that we discuss in the revised
manuscript Section 4.4:

1. **PPO seed instability on this benchmark is large.** Runs
   `_v3` and `_v4` use identical code, identical config, identical
   seed pool, and differ only in stochastic execution between two
   consecutive launches. Their Sharpe values differ by **21.4**.
   The two 1-million-timestep runs (`_v1` and `_v2`) differ by
   only **1.76** Sharpe, suggesting that **higher training budget
   damps seed variance** but does not eliminate it. The revised
   manuscript reports this seed-instability band as a separate
   uncertainty source in Section 4.4 and recommends that future
   work on the same benchmark either (a) train PPO to convergence
   with > 1 M timesteps per agent, or (b) report the multi-seed
   distribution rather than a single point estimate.

2. **The best single-seed realisation (`_v3`, Sharpe −6.36) is
   still worse than passive Buy-and-Hold** over the same test
   window (B&H Sharpe ≈ +1.92, B&H cumulative return ≈ +29.4 %).
   We are therefore not in a position to claim the strategy
   outperforms a buy-and-hold baseline on BTC-USDT 1-hour bars
   under any of the four configurations.

This sensitivity table is provided to clarify the meaning of the
walk-forward CV numbers in §3.1: **walk-forward CV at 30 k
timesteps per agent is a stricter test that pays an explicit
data-budget cost** (~20 % less training data per fold than the
single split) **on top of the same PPO seed instability**, so its
worse mean values are consistent with the single-split spread of
`_v3` … `_v4`.

---

## §3.2 — New Table 1 (regime classifier per fold)

| Fold | Test window | Accuracy | F1 (macro) |
|-----:|-------------|---------:|-----------:|
| 1 | 2022-04..2022-08 | 0.469 | 0.326 |
| 2 | 2022-08..2022-12 | 0.485 | 0.339 |
| 3 | 2022-12..2023-04 | 0.470 | 0.350 |
| 4 | 2023-04..2023-08 | 0.460 | 0.349 |
| 5 | 2023-08..2023-12 | 0.420 | 0.308 |
| **Mean** | | **0.461** | **0.334** |

3-class chance with this label distribution ≈ 33%; the trained
classifier is only marginally above chance on the forward-looking
Trend-Scanning labels.

---

## §4 — Re-framed manuscript contribution

In light of §0, the manuscript's Section 1 (Introduction) and
Section 4 (Experiments) are rewritten so that the paper's
contribution is now stated as:

> **"We identify and quantify a class of look-ahead-bias failure
> modes in regime-aware reinforcement-learning trading systems, and
> propose architectural changes that begin to address them. We
> demonstrate that (i) lagging SMA-based regime labels create an
> apparent ~90% classifier accuracy that collapses to ~46% under
> forward-looking Trend-Scanning labels with the same feature
> pipeline; (ii) downstream PPO ensembles under a long-short action
> space amplify rather than absorb classifier errors; and (iii)
> common post-processing transformations (action inversion, B&H
> blending, position scaling) can fully mask these failures in the
> reported metrics. We provide a fully open-source pipeline that
> exposes both the failure modes and the post-processing artefacts."**

We believe this is genuinely novel and more valuable to the *Expert
Systems with Applications* readership than the original claim.

---

## §5 — Architectural improvements delivered in the revision

The revision implements three architectural changes that target the
failure modes identified above. Each is exposed as a configurable
flag in `config/config.yaml` so the editor and reviewers can
reproduce both the v1 and v2 behaviours on the same codebase.

### §5.1 — Reward function v2 with direction-aligned shaping

*File:* `src/env/rewards.py`. The v1 Bear-pool reward (Sortino over
30 hourly bars minus a small transaction-cost term) produces a
sparse, low-amplitude signal that PPO cannot credit-assign at the
hourly frequency. We replace it with a per-step composite

$$R_t = \mathrm{pv}_t + \alpha\, w_t r_t - \lambda_c\,\tilde c_t + \beta\,\mathrm{shaping}_t,$$

where $w_t\in[-1,+1]$ is the signed portfolio weight committed by
the agent at step $t$, $r_t$ is the realised next-bar simple
return, $\tilde c_t$ is the transaction cost normalised by previous
portfolio value, and $\mathrm{shaping}_t$ encodes regime-specific
preferences: a long bonus in Bull bars, a short bonus and an
asymmetric long-side penalty in Bear bars, and a position-magnitude
penalty in Sideways bars. With $(\alpha,\lambda_c,\beta)=(3.0,1.0,0.2)$
and a wrong-side coefficient $\gamma=1.5$, the offline sanity check
(`scripts/_sanity_reward_v2.py`) reports a correct-vs-wrong-direction
reward spread of approximately 4.1× (Bull), 4.85× (Bear), and 4.0×
(Sideways), versus approximately 1.0× under v1.

### §5.2 — Regularised classifier with shallower trees

*File:* `src/regime/regime_classifier.py`. The v1 defaults
(`max_depth=6`, `n_estimators=100`, no L1/L2) produced 100% train
and ≈47% test accuracy on Trend-Scanning labels — a 53-point
overfitting gap. The v2 constructor exposes the full XGBoost
regularisation surface and the pipeline driver forwards every
parameter from `config.yaml`:

- `max_depth: 4`
- `n_estimators: 200` (with `early_stopping_rounds: 30`)
- `learning_rate: 0.05`
- `colsample_bytree: 0.7`
- `subsample: 0.8`
- `reg_lambda: 1.0`, `reg_alpha: 0.0`, `min_child_weight: 1.0`

These knobs were already documented in v1's `config.yaml` but were
never read by the constructor — a defect we now fix.

### §5.3 — Visual branch made optional, off by default

*File:* `src/data/feature_fusion.py`,
`config.features.use_visual`. The 512-D ResNet-18 branch was
empirically shown to add zero predictive value (test accuracy 47.0%
without vs 46.9% with on fold 1 of the Trend-Scanning split). In
v2 it is opt-in; when disabled, the unified state vector becomes
27-D and the PPO observation noise floor is reduced by an order of
magnitude.

### §5.4 — Reproducibility of v2 results

The revised Table 2 is computed at
`results/walk_forward_reward_v2/` and is generated by:

```bash
python scripts/run_walk_forward.py \
    --label-method trend_scanning \
    --subdir walk_forward_reward_v2
```

The runtime is approximately 90 min on CPU; per-fold artefacts,
the aggregate JSON, and the human-readable markdown summary are
all version-controlled under that directory.

---

## §6 — Citation responses (items #17, #18)

Two reviewer-requested citation blocks (one from Reviewer #1/#2 and
one from Reviewer #4) ask us to incorporate references whose subject
matter does not overlap with the present study. We have read each of
the suggested works in full and have weighed (i) topical relevance,
(ii) methodological transferability, and (iii) the risk that
unrelated additions would dilute the manuscript's narrative. After
this assessment we decline both blocks for the specific reasons given
below.

### §6.1 — Item #17 (Reviewer #1 / #2): unrelated citation cluster

Reviewers #1/#2 suggested the four following works be cited:

- Adeleke, T. A. (2023). *Smart-city operations and AI-driven
  service-level reliability.*
- Adeleke, T. A. (2025). *Battery-AI maintenance and prognostics in
  energy storage systems.*
- Hammed, O. (2025). *Digital-twin frameworks for industrial
  cyber-physical systems.*
- Joshua, K., & Kim, S. (2025). *Multi-modal AI for autonomous
  infrastructure monitoring.*

**Response — politely declined.** These works concern smart-city
operations, battery prognostics and health management, digital-twin
frameworks, and infrastructure monitoring. None of them addresses
financial time-series modelling, market regime classification, or
reinforcement learning for portfolio management. The methodological
intersection with the present manuscript is limited to the generic
notion of "AI in time-stamped data," which is too broad a hook to
justify inclusion in *Expert Systems with Applications*'s scope as a
financial methodology paper. Including them would also break the
manuscript's literature-review structure (regime-aware RL trading →
sentiment-based finance NLP → time-series cross-validation), which is
the structure the revised Introduction has been written around. We
therefore respectfully decline to cite these four works. We would be
happy to reconsider if the reviewers can identify a specific
methodological argument from any of the suggested works that should
be answered in our manuscript.

### §6.2 — Item #18 (Reviewer #4): bearing-RUL citation

Reviewer #4 suggested:

- *"Estimation of remaining useful life of bearings using neural
  networks ..."*, DOI `10.1007/s40998-018-0108-y`
  (mechanical-engineering reliability study).

**Response — politely declined.** The cited work is a
mechanical-engineering remaining-useful-life (RUL) study on rolling
bearings. While the underlying neural-network and time-series
framing is interesting, there is no shared methodology with our
study: the bearing-RUL task is a *survival/regression* problem on
monotonically degrading vibration signals, whereas our problem is
*discrete regime classification + reinforcement-learning portfolio
control* on non-monotonic, non-degrading financial price series.
The two settings differ in label semantics (RUL ≠ regime), in
prediction horizon, in loss formulation, and in actionable downstream
output. Including this citation would imply a methodological lineage
that does not exist. We respectfully decline.

### §6.3 — Citations we *did* add

For completeness, the revised literature review (Section 2) was
strengthened with the following ten works that *are* methodologically
on-scope:

1. López de Prado, M. (2018). *Advances in Financial Machine
   Learning.* Chapters 5 (Trend Scanning) and 7 (Purged K-Fold).
2. Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe
   ratio. *Journal of Portfolio Management*.
3. Araci, D. (2019). *FinBERT: A pretrained language model for
   financial communications.* arXiv:1908.10063.
4. Schulman, J. et al. (2017). Proximal Policy Optimization
   algorithms. arXiv:1707.06347.
5. Bao, W., Yue, J., & Rao, Y. (2017). Deep learning framework for
   financial time series. *PLoS ONE*.
6. Yang, H. et al. (2020). Deep reinforcement learning for automated
   stock trading: an ensemble strategy. *ICAIF*.
7. Liu, X.-Y. et al. (2021). FinRL — A deep RL library for automated
   stock trading. *ICAIF*.
8. Theate, T., & Ernst, D. (2021). An application of deep RL to
   algorithmic trading. *Expert Systems with Applications.*
9. Ying, X. (2019). An overview of overfitting and its solutions.
   *Journal of Physics: Conference Series*.
10. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning:
    An Introduction* (2nd ed.). MIT Press.

These citations now anchor the methodology section to the
mainstream literature on time-series-safe validation,
domain-pretrained NLP for finance, and ensemble RL for portfolio
control — i.e. precisely the lineage the reviewers expected.

---

## §7 — Summary of all 18 mapped items

| # | Category | Item | Status |
|---|----------|------|--------|
| 1 | Methodology | Look-ahead bias / FinBERT | done |
| 2 | Methodology | Time-series CV / Walk-Forward | done |
| 3 | Methodology | Forward-looking labels / Trend Scanning | done |
| 4 | Methodology | Visual feature ablation | done — visual adds 0 |
| 5 | Methodology | Confidence threshold | retuned |
| 6 | Methodology | Reward design | partially done; §5 |
| 7 | Methodology | Slippage / ATR dynamic | done |
| 8 | Methodology | Action space (long-short) | done |
| 9 | Statistics | Bootstrap + Bonferroni | done |
| 10 | Experiments | Overfitting analysis | done — 1.000 train / 0.47 test |
| 11 | Experiments | Generalisation | new Section 4.6 |
| 12 | Experiments | Latency / complexity | done |
| 13 | Experiments | LLM limitation / No-news ablation | done |
| 14 | Structure | Intro / Conclusion | rewritten — §4 above |
| 15 | Structure | Notation | new table in §3 |
| 16 | Structure | Lit review + Figures + Editing | brief + 10 refs |
| 17 | Citation | Unrelated #2 refs | declined |
| 18 | Citation | Bearing-RUL ref | declined |
| (1) Reproducibility | Statement + Artefacts | done; GitHub |

---

We close by re-affirming our deep gratitude to Reviewer #3, whose
methodological criticisms forced the discovery in §0, and to
Reviewer #4 for the code-release requirement that made the
post-processing artefact visible. We believe the revised manuscript
is more truthful and more useful as a result.

Sincerely,
**[Authors redacted for double-blind peer review]**
