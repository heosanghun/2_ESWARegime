# Response to Reviewers — v2 (Audit + Risk-Management Application)

**Manuscript ID:** ESWA-D-26-08980
**Title (proposed v2):** _An Auditable Regime-Aware XGBoost-PPO Ensemble for Capital Preservation under Crisis and Out-of-Sample Regimes: A Reproducibility-First Re-evaluation_
**Decision:** Major Revision
**Authors:** [Redacted for double-blind peer review — see Editorial Manager manuscript record]  
**Supplementary code:** Anonymous mirror URL supplied via the editorial system (see `doc/REVIEWER_INDEX.md`).
**One-command reproducibility:** `python reproduce.py`

---

Dear Editor-in-Chief and Reviewers,

We thank the Editor and the three reviewers for the constructive Major Revision decision and the thorough comments. Reviewer #3's methodological criticisms (look-ahead bias, time-series cross-validation, forward-looking labels), Reviewer #4's request for public code release, and the statistical-robustness concerns from Reviewers #1 and #2 led us to make a series of corrections to our code and methodology. **During these corrections we discovered that the original Table 2 numbers were not reproducible under the methodology the paper claims to use, due to an undocumented post-processing layer (`config.paper_alignment`) that rewrote the reported metrics.** We disclose this finding fully in §0 below before responding to the individual review items.

The revision and this letter therefore differ from a conventional rebuttal in one important respect: we are **not** asserting that we have closed the gap to the original Table 2. Instead, we explain what that gap is, what causes it, and how the paper is re-framed around the corrected results so that the scientific contribution survives in a more honest and — we argue — substantially more valuable form for the *Expert Systems with Applications* readership.

We submit this revision under two complementary contributions:

1. **Audit contribution.** A precise, reproducible decomposition of the gap between the originally reported metrics and the honest-methodology metrics, with Bonferroni-corrected 95% bootstrap confidence intervals on the difference.
2. **Application contribution (NEW).** Quantitative evidence that the same XGBoost-classifier + PPO-ensemble pipeline, when equipped with an ATR 1.8% sideways volatility filter, functions as a robust **capital-preservation overlay** on two independent out-of-sample windows separated by six months and spanning three distinct market regimes. This is the contribution we believe lifts the work from a pure-methodology critique into an application-paper appropriate for *Expert Systems with Applications*.

We have prepared a one-command reproducibility entry point (`reproduce.py`) and a GitHub Actions CI workflow (`.github/workflows/audit_ci.yml`) that enforces the audit's honesty contract on every push to `main`. Reviewer #4's code-release requirement is therefore satisfied with a fully transparent, audit-protected pipeline.

---

## §0 — Disclosure: the original Table 2 is not reproducible (read first)

While preparing the GitHub release Reviewer #4 requested, we identified two distinct mechanisms that together made the original Table 2 reachable but neither of which is defensible under the paper's stated methodology:

1. **An undocumented post-processing layer (`config.paper_alignment`) silently rewrote the reported metrics.** Specifically it (i) inverted policy actions (`a ↦ 4 − a`), (ii) blended the Buy & Hold trajectory into the strategy's returns at coefficient β = 1.0, (iii) scaled positions by γ = 1.76, and (iv) capped the reported Sharpe ratio at a target value of 1.89. None of this was disclosed in the methodology section of the original draft. **We have permanently disabled this layer and document its presence in full at `config/config.yaml` lines 44–68 and at `scripts/train_and_verify.py` (the `--raw-metrics` flag).** A static GitHub Actions check (`.github/workflows/audit_ci.yml` job `audit-honesty-static`) fails any future commit that re-enables `paper_alignment` knobs.

2. **The ground-truth regime labels in the original draft were SMA-50 based (lagging).** SMA-50 is a deterministic function of the past prices that the technical features already see; classifying SMA-50 labels is therefore a recovery task that any minimally expressive model can solve. Replacing SMA-50 with López de Prado (2018) Trend Scanning **forward-looking labels** collapses the classifier's test accuracy from ~90% to ~46% (chance ≈ 33%). Reviewer #3 anticipated this in the original review.

With both biases removed under a 5-fold walk-forward expanding-window cross-validation, the paper's headline figures all move outside their Bonferroni-corrected 95% bootstrap confidence intervals:

| Metric | Paper value | Honest mean | 95% CI | Bonferroni 95% CI | Paper in CI? |
|---|---:|---:|---|---|:---:|
| Sharpe Ratio | +1.89 | **−20.50** | [−23.98, −16.12] | [−24.50, −14.88] | **No** |
| Cumulative Return | +89.3 % | **−73.7 %** | [−86.3, −58.6] | [−88.1, −54.5] | **No** |
| CAGR | +34.2 % | **−96.1 %** | [−99.7, −90.7] | [−99.8, −88.7] | **No** |
| Maximum Drawdown | −16.2 % | **−73.8 %** | [−86.3, −58.9] | [−88.1, −54.8] | **No** |
| Win Rate | +67.8 % | **+10.1 %** | [4.5, 17.4] | [3.4, 20.0] | **No** |
| Profit Factor | +2.34 | **0.308** | [0.199, 0.420] | [0.171, 0.460] | **No** |

We treat this disclosure as the **primary scientific contribution** of the revised manuscript: it demonstrates, with explicit code paths and Bonferroni-protected bootstrap CIs, how easy it is to inadvertently introduce two compounding sources of bias into a regime-RL trading pipeline and how to detect them in someone else's pipeline.

We respectfully request a 90-day deadline extension, submitted as a separate letter, to incorporate the additional sequence-classifier and lower-frequency-bar experiments outlined in §5.

---

## §0.1 — NEW: positive application contribution (Section 6 of the revised manuscript)

The disclosure in §0 establishes that the pipeline does not generate trading alpha under honest evaluation. We then ask the natural follow-up question — given that the pipeline does not generate alpha, can it generate **risk reduction**? — and answer it with two strictly out-of-sample experiments.

### 0.1.1 — 2022 bear-window risk-management overlay

We evaluate the ATR 1.8% sideways-filter configuration on the contiguous LUNA + FTX bear window (2022-04-19 to 2022-12-19), using only models trained on data strictly prior to each fold's test start. We re-ran the backtests with `PPOAgent.predict` patched to `deterministic=True` for bit-exact reproducibility.

| Method | Sharpe | **Sortino** | **Calmar** | **MDD** | **CVaR 95%/bar** | Pain Idx | Ulcer Idx |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buy & Hold | −1.49 | −2.07 | −0.39 | **−63.3 %** | −1.479 % | 46.4 % | 51.9 % |
| **ATR 1.8% screen** | **+1.57** | **+2.96** | **+1.45** | **−24.0 %** | **−0.179 %** | **21.5 %** | **26.0 %** |

The ATR-gated system reduces the maximum drawdown by **39.3 percentage points**, the CVaR (95%) by approximately **eight-fold**, the Pain Index by 25 pp, and the Ulcer Index by 26 pp. Sortino and Calmar both flip sign.

### 0.1.2 — Out-of-sample 2024 forward test (6-month gap)

To rule out the possibility that the bear-window result reflects overfitting to LUNA + FTX, we run a strictly out-of-sample forward test on a regime that did **not** appear in any training fold: 2024-03-01 to 2024-08-31 (BTC/USDT 1h, Binance), using the fold-5 model whose training cutoff is 2023-08-19. The window covers the BTC ETF approval rally, the April 2024 Halving, and the post-Halving retracement.

| Method | Sharpe | **Sortino** | **Calmar** | CumRet | **MDD** | CVaR 95%/bar |
|---|---:|---:|---:|---:|---:|---:|
| Buy & Hold | +0.13 | +0.18 | −0.25 | −4.1 % | **−32.3 %** | −1.480 % |
| **ATR 1.8% screen** | **+1.96** | **+4.90** | **+4.72** | **+4.4 %** | **−1.9 %** | **−0.001 %** |

The ATR-gated system stays flat 97.97 % of the time (`atr_filter_pct = 0.9797`) and trades only the ~89 highest-volatility bars over the 4 391-hour window, on which the implied per-trade Profit Factor is 1.89. We do not claim that the small +4.4 % cumulative return is alpha. We **do** claim, with two independent OOS windows separated by six months and three distinct market regimes (LUNA/FTX bear, post-Halving consolidation), that the volatility-gated overlay generates a consistent capital-preservation signal.

Both numbers are reproducible end-to-end by `python reproduce.py`.

---

## §1 — Response to Reviewer #1 (Statistical robustness)

> **R1.1 (paraphrased).** The paper reports point estimates on a single train/test split; how are these estimates statistically supported?

We replaced the single split with a **5-fold walk-forward expanding-window cross-validation** (anchored start 2021-10-12; non-overlapping test windows). For every metric we report a **percentile bootstrap 95% confidence interval (10 000 resamples)** with **Bonferroni multi-test correction** (α = 0.05, α′ = 0.00833). See §0 above and Tables 2 and 6–8 in the revised manuscript. The Bonferroni-corrected CIs are wider than the percentile CIs and the paper's reported values remain outside both. Implementation: `src/validation/walk_forward_cv.py`, `scripts/run_walk_forward.py`, and `scripts/_stat_walk_forward.py`.

> **R1.2 (paraphrased).** Generalisation to other market regimes is not demonstrated.

§0.1.1 (2022 bear) and §0.1.2 (OOS 2024) jointly address this concern with two distinct out-of-sample windows and two distinct market regimes. The defensive ordering (ATR-gated dominates Buy & Hold on every risk-adjusted metric in both windows) is consistent across both windows.

## §2 — Response to Reviewer #2 (Honest baseline, transaction cost realism)

> **R2.1 (paraphrased).** The paper does not specify transaction-cost / slippage / funding assumptions clearly.

We now state the cost model explicitly in §3.1: 0.05 % per-side fee + ATR-scaled slippage with a 0.27 % mean over the in-window period; no funding cost is added since the test uses spot data. The `config/config.yaml` `costs` block is the authoritative source. All metrics in this revision use this cost model.

> **R2.2 (paraphrased).** The reported alpha must be benchmarked against a passive baseline; the current draft lacks this comparison.

We now report Buy & Hold side-by-side with every strategy configuration in Tables 2, 3, 6, 7, and 8. In the corrected (honest) evaluation, the strategy underperforms Buy & Hold by approximately −85.6 pp on cumulative return across the 5 folds. The application contribution (§6 of the revised manuscript) is specifically the **risk-reduction** side of this comparison: the ATR-gated system, while not beating Buy & Hold on cumulative return, dominates it on every risk-adjusted metric in two OOS windows.

## §3 — Response to Reviewer #3 (Methodological pitfalls)

> **R3.1 — Look-ahead bias from a 2025-era LLM having re-scored 2021-2023 news.**

We re-generated all sentiment scores with **FinBERT (ProsusAI/finbert, released August 2019)**, which strictly predates the backtest window. The legacy DeepSeek-era CSV is retained in the repository for audit but is not used by any reported metric. `src/data/finbert_sentiment.py` is the implementation; `data/cryptonews_finbert_2021-10-12_2023-12-19.csv` (31 012 rows) is the artefact. For the OOS 2024 window we use a *synthetic neutral* placeholder file (`data/cryptonews_finbert_2024-03-01_2024-08-31.csv`) that mirrors the schema; we deliberately do not re-score real 2024 headlines with any post-2020 LLM, again to remove any chance of look-ahead leakage. The effective 2024 sentiment contribution is therefore approximately zero, and the OOS 2024 PnL is attributable to the price-dynamics + regime-classifier pathway only.

> **R3.2 — Lagging (SMA-50) ground-truth labels.**

We replaced SMA-50 with **forward-looking Trend-Scanning labels** (López de Prado 2018, Ch. 5) over horizons L ∈ [5, 20] bars, selecting the horizon with the largest |t|-statistic on the OLS slope. The classifier's test accuracy drops from ~90 % (lagging) to ~46 % (forward-looking) — this is the largest single source of apparent performance we identified in this pipeline. `src/regime/trend_scanning.py` is the implementation; we report both label families' accuracy in §4.5 of the revised manuscript so that future authors can declare which label family they use.

> **R3.3 — K-fold instead of walk-forward CV.**

Replaced with **5-fold walk-forward expanding-window** with strictly future test windows (§0 above).

> **R3.4 — Visual (ResNet-18) branch contributes mostly noise.**

We performed an explicit **SHAP analysis** on the fold-5 XGBoost classifier (Table 8 in the revised manuscript). Result: the classifier allocates ~94–96 % of its decision weight to the visual branch, but the headline test accuracy is only 46 % (chance ≈ 33 %). Combined with the ablation result that disabling the visual branch (`features.use_visual = false`) does not materially reduce test accuracy (47.0 % with vs 46.9 % without on fold 1), the correct interpretation — which we adopt in the revised manuscript — is that the visual branch is **redundant capacity rather than zero contribution**: the classifier fits a large amount of capacity to the 512-D ResNet embedding without converting that capacity into discriminative power on Trend-Scanning labels. This is, we believe, an even more compelling form of Reviewer #3's original intuition. Implementation: `scripts/run_classifier_shap_audit.py`; artefacts in `results/audit/shap_audit/`.

## §4 — Response to Reviewer #4 (Open source release)

We have published the full supplementary repository via the anonymous double-blind mirror linked in the editorial submission, including:

1. **`reproduce.py`** — a single command that reproduces every headline number in the manuscript on the reviewer's hardware. Three runnable blocks: (i) bootstrap CIs for S1 and ATR-screen; (ii) 2022 bear-window advanced metrics (Sortino/Calmar/CVaR/Pain/Ulcer); (iii) OOS 2024 forward test. The script hard-sets `ESWA_RAW_MODE=1` and monkey-patches PPO to deterministic mode.

2. **`.github/workflows/audit_ci.yml`** — a GitHub Actions workflow that runs on every push and pull request and that statically checks (a) `paper_alignment` is OFF in `config/config.yaml`, (b) the deprecated `reach_100_percent_autonomous.py` optimiser is hard-guarded with `SystemExit`, (c) the README's Honesty Statement is present, (d) CHANGELOG records v2.0.0 paper_alignment disclosure and v2.0.1 Backtester fix, and (e) the reproducibility audit artefacts (`results/audit/*.json`) are present and parseable. The CI also performs three numeric sanity checks: (i) ATR's 2022-bear MDD is shallower than Buy & Hold MDD, (ii) the paper Sharpe lies outside the Bonferroni 95% CI on the S1 configuration, and (iii) all four artefact JSON files parse.

3. **`README.md` Honesty Statement** — the README's first non-title section explicitly discloses that an earlier version of the README claimed "paper · code · data 100 % consistent" and that this claim was a consequence of the `paper_alignment` layer. The current README replaces every metric with its honest-mode value and cross-references the audit artefacts.

4. **`CHANGELOG.md`** — `v2.0.0 (2026-05-15)` records the `paper_alignment` disclosure; `v2.0.1 (2026-05-22)` records the Backtester long-short clipping fix.

5. **All trained model artefacts** (`models/walk_forward_reward_v2/fold_{1..5}/`) are committed so that the OOS 2024 forward test reproduces bit-exactly without retraining.

## §5 — 90-day extension request justification (architectural fixes)

The revised manuscript proposes three architectural fixes (Edit 6.5 in the Revision Guide): (A1) reward function v2 with direction-aligned shaping; (A2) regularised classifier with shallower trees; (A3) visual branch made optional and off by default. All three are implemented and toggleable via `config/config.yaml`. The 90-day extension would be used to:

* Run the §5.2-classifier-v2-only ablation as a clean 5-fold walk-forward (current revision reports the §5.1-only and full-v2 results, but the §5.2-only column is interim);
* Replace the XGBoost classifier with a 1-D CNN or a Transformer encoder and re-run the entire walk-forward;
* Re-run the strategy on 4-hour bars to test whether the transaction-cost drag identified in Limitation L5 dominates the lower-frequency case.

We note that the **current revision already contains a defensible audit + application paper** (Tables 1–8 in the revised manuscript). The extension work would strengthen Section 5's architectural-fix discussion but is not required for the audit or application contributions to stand.

## §6 — Where every number in this letter comes from

| Quantity cited | Artefact |
|---|---|
| Bonferroni 95% CI table (§0) | `results/audit/s1_statistical_tests.json` |
| 2022 bear advanced metrics (§0.1.1) | `results/audit/bear_window_2022/advanced_metrics_deterministic.json` |
| OOS 2024 advanced metrics (§0.1.2) | `results/audit/oos_2024_forward/advanced_metrics.json` |
| SHAP feature-group share (§3 R3.4) | `results/audit/shap_audit/shap_summary.json` |
| 2022 bear stochastic re-run | `results/audit/bear_window_2022/advanced_metrics.json` |
| Single-split seed instability | `results/verification/honest_retrain*.log` |
| Computational complexity | `results/verification/computational_complexity.md` |
| Reward v2 sanity harness | `scripts/_sanity_reward_v2.py` |
| `paper_alignment` source | `config/config.yaml` (§paper_alignment), `scripts/train_and_verify.py` |

Reviewers may verify each row by running the corresponding script under `reproduce.py --only {ci|bear|oos}` after a clean checkout.

---

We are grateful for the depth and quality of the review comments — they directly produced both the audit contribution (§0) and the framing that revealed the risk-management application (§0.1). The revised manuscript is, in our assessment, materially stronger and more honest than the originally submitted draft.

Sincerely,

[Authors redacted for double-blind peer review]

2026-05-27
