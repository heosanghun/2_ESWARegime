# Final synthesis — honest measurements vs the original draft

_Last updated 2026-05-19. This document is the single source of truth
for performance numbers cited in the rebuttal letter and the revised
manuscript. Whenever a number here disagrees with another file, treat
this document as authoritative and submit a follow-up fix to the
disagreeing file._

---

## TL;DR

1. The original draft (Table 2) reported Sharpe **+1.89**, cumulative
   return **+89.3 %**, win-rate **0.678**, profit factor **2.34**.
2. With every reviewer-mandated methodology fix in place (FinBERT,
   Trend-Scanning forward labels, Walk-Forward expanding-window CV,
   ATR-based dynamic slippage, **`paper_alignment` disabled — raw
   metrics**, and **Backtester v2.0.1 long-short fix**), the system
   **does not reproduce those numbers**. None of the original draft
   values is inside any Bonferroni-corrected 95 % CI we compute.
3. **Canonical evaluation (2026-05-19):** reward-only v2 at **1 M
   timesteps**, 5-fold walk-forward, post-rebacktest with corrected
   long-short Backtester → mean Sharpe **−39.57 ± 4.88**, mean
   cumulative return **−99.9 %**. **Path A** (Discovery-of-Flaws)
   reframing applies.
4. The **reward function redesign** (§5.1) closes **~34 %** of the
   Sharpe gap at **30 k** timesteps (+7.7 Sharpe vs v1 baseline),
   but **1 M training does not improve** the walk-forward mean
   (−39.6 vs −12.8 at 30 k) — under-training is not the primary
   explanation.
5. A **Backtester clip bug** (`np.clip(weights, 0, 3)`) silently
   zeroed short positions during metric computation, inflating
   pre-fix 1 M numbers by **~9.6 Sharpe** (mean −29.93 → −39.57).
6. **PPO seed instability** remains large at 30 k (Sharpe spread
   21.4 between `_v3` and `_v4`); at 1 M single-split the spread
   narrows to 1.76.

The remainder of this file walks through (i) the master result
tables, (ii) which row each manuscript / rebuttal table cites, and
(iii) how to reproduce each row.

---

## 1. Master result table

All numbers come from `results/verification/honest_retrain*.log`
(single-train/test-split runs) or
`results/walk_forward*/summary.json` (5-fold walk-forward CV).
Buy-and-Hold benchmark numbers come from
`scripts/run_walk_forward.py --benchmark`.

### 1.1 Single-split runs (train 2021-10-12 → 2023-06-19, test 2023-06-19 → 2023-12-19)

| Run id | total_timesteps × 15 agents | Sharpe | Cum. Return | Max DD | Win Rate | Profit Factor | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| `honest_retrain` (v1) | 1 000 000 | −14.00 | −76.52 % | −77.94 % | 40.6 % | 0.61 | 7 h 20 m |
| `honest_retrain_v2`   | 1 000 000 | **−12.24** | −81.34 % | −83.25 % | 38.8 % | 0.65 | 8 h 26 m |
| `honest_retrain_v3`   |    30 000 | **−6.36**  | **−60.95 %** | −63.84 % | **44.6 %** | **0.80** | 22 m |
| `honest_retrain_v4`   |    30 000 | −27.72 | −91.15 % | −91.28 % | 11.7 % | 0.26 | 21 m |

Reading: v3 and v4 use the *same* code, the *same* config
(`use_visual=false` ablation off — 539-D state), the *same* seed
pool (42-46, 142-146, 242-246) and **differ only in stochastic
order-of-execution effects between two consecutive runs.** The 21.4
Sharpe spread between them is the natural seed-instability range
of PPO at 30 000 timesteps on this benchmark. The two 1-million-step
runs (v1, v2) are far closer (Sharpe spread = 1.76) — more training
narrows but does not eliminate the variance.

### 1.2 5-fold walk-forward CV (Trend-Scanning labels, embargo 24 h)

| Configuration | Mean Sharpe | Mean Cum. Return | Mean MDD | Mean Win Rate | Mean PF |
|---|---:|---:|---:|---:|---:|
| v1 baseline (rwd-v1, clf-v1, visual-on) @ 30k | −20.50 ± 5.01 | −73.68 % | −73.81 % | 10.1 % | 0.308 |
| reward-only v2 (§5.1) @ 30k | −12.81 ± 3.57 | −46.73 % | −46.80 % | 4.4 %  | 0.289 |
| full v2 (§5.1 + §5.2 + §5.3) @ 30k | −14.66 ± 3.23 | −56.86 % | −56.92 % | 3.6 %  | 0.251 |
| reward-only v2 @ 1M, **pre-rebacktest** (clip bug) | −29.93 ± 4.01 | −97.0 % | −97.0 % | 16.9 % | 0.293 |
| **reward-only v2 @ 1M, post-rebacktest (canonical)** | **−39.57 ± 4.88** | **−99.9 %** | **−99.9 %** | **31.4 %** | **0.276** |

Bonferroni 95 % CI (Sharpe, 1M post-rebacktest): **[−43.17, −34.10]**.

(Per-fold detail: `results/walk_forward_v2_comparison.md`,
`results/walk_forward_reward_v2_1M/summary_rebacktest.json`,
`results/walk_forward_reward_v2_1M/autopilot_report.md`.)

### 1.3 Buy-and-Hold benchmark over the same windows

| Window | B&H Cum. Return | B&H Sharpe |
|---|---:|---:|
| Single-split test (2023-06-19 → 2023-12-19) | +29.4 % | +1.92 |
| Walk-forward 5-fold mean | +11.9 % | +0.61 |

The strategy underperforms passive Buy-and-Hold across every honest
configuration we tried.

### 1.4 Side-by-side vs the original draft

| Metric | Original draft (Table 2) | v1 baseline WF @ 30k | reward-only v2 @ 30k | **reward-only v2 @ 1M (post-fix)** |
|---|---:|---:|---:|---:|
| Sharpe          | +1.89  | −20.50 | −12.81 | **−39.57** |
| Cum. Return     | +0.893 | −0.737 | −0.467 | **−0.999** |
| CAGR            | +0.342 | −0.961 | −0.753 | ≈ −1.000 |
| Max DD          | −0.162 | −0.738 | −0.468 | **−0.999** |
| Win Rate        | +0.678 | +0.101 | +0.044 | +0.314 |
| Profit Factor   | +2.34  | +0.308 | +0.289 | +0.276 |

The reward redesign (v2) at 30 k closes ~34 % of the v1 Sharpe gap;
**1 M training with a corrected backtester confirms architectural
limits** — the gap to the original draft widens to **41.5 Sharpe
points**. We are explicit about this in §3.1 of the rebuttal and in
`doc/Path_A_Reframing_FILLED.md`.

### 1.5 Routing-gap ablation (2026-05-19)

Three-factor study (confidence threshold, probability EMA, PPO 50k) on
retrained 30k models. Full report: `doc/ROUTING_ABLATION_REPORT.md`;
matrix: `results/routing_ablation/summary_matrix.md`.

| Arm | Intervention | Mean Sharpe | Δ vs fair 30k (R0b = −24.34) |
|---|---|---:|---:|
| R0b | baseline (conf 0.35, no EMA) | −24.34 | — |
| **B2** | **prob EMA span 12** | **−16.59** | **+7.74** |
| B1 | prob EMA span 9 | −16.87 | +7.46 |
| D1 | conf 0.65 + EMA 9 | −23.07 | +1.27 |
| A2 | conf 0.75 | −25.93 | −1.59 |
| C1 | PPO 50k | −29.84 | −5.51 |
| REF | PPO 1M (canonical) | −39.57 | −15.23 |

**Takeaway:** EMA smoothing reduces regime-switch turnover (~914 → ~184
switches/fold) and improves Sharpe by ~7.5 points, but **does not**
produce positive risk-adjusted returns. Higher confidence thresholds
(0.65–0.75) **worsen** performance. PPO 50k resembles 1M overfitting
direction (−30 Sharpe).

### 1.6 Phase 2 soft routing (2026-05-19)

Prob-weighted pool blending (`regime.routing_mode: soft`) on the same
30k models. Full report: `doc/PHASE2_SOFT_ROUTING_REPORT.md`;
matrix: `results/routing_ablation/phase2_soft/summary_matrix.md`.

| Arm | routing | EMA | Mean Sharpe | Δ vs R0b | Δ vs B2 |
|---|---|---:|---:|---:|---:|
| B2 | hard | 12 | −16.59 | +7.74 | — |
| **S1** | **soft** | **12** | **−12.76** | **+11.57** | **+3.83** |
| S2 | soft | 0 | −17.37 | +6.97 | −0.78 |

**Takeaway:** Soft routing with EMA 12 is the **best honest 5-fold WF
result to date** (Sharpe −12.76, CumRet −70.7 %). It mitigates
hard-routing amplification but **does not** close the gap to Table 2
(+1.89 Sharpe; remaining gap **−14.65**). Wrong-sign folds vs B&H
unchanged at **2 / 5**.

### 1.7 LSTM sequential classifier (2026-05-19)

LSTM over 48-bar × 19 technical features, soft routing + EMA 12.
Full report: `doc/PHASE2_SEQ_CLASSIFIER_REPORT.md`;
matrix: `results/routing_ablation/phase2_lstm/summary_matrix.md`.

| Arm | Classifier | Mean Sharpe | Routing acc | Δ vs S1 |
|---|---|---:|---:|---:|
| **S1** | XGB + soft + EMA12 | **−12.76** | **46.2 %** | — |
| L1 | LSTM + soft + EMA12 | −17.61 | 35.9 % | **−4.85** |

**Takeaway:** Sequential LSTM **underperforms** tabular XGBoost under
the same routing stack. The ~47 % accuracy ceiling is a **label /
feature predictability** problem, not solved by temporal modelling
alone. **S1 remains canonical best honest config.**

### 1.8 Causal regime labels — classifier only (2026-05-19)

Backward Trend Scanning (no lookahead) classifier + soft + EMA 12.
Full report: `doc/PHASE2_CAUSAL_LABEL_REPORT.md`;
matrix: `results/routing_ablation/phase2_causal/summary_matrix.md`.

| Arm | Labels | Mean Sharpe | Routing acc | Δ vs S1 |
|---|---|---:|---:|---:|
| **S1** | forward TS | **−12.76** | 46.2 % | — |
| C1 | causal TS (clf only) | −13.72 | 75.6 % | −0.96 |

**Takeaway:** Causal labels raise routing accuracy **+29 pp** but
aggregate Sharpe is **unchanged** because PPO pools were trained on
forward labels.

### 1.9 C2 — fully aligned causal pipeline (2026-05-19)

Causal labels for **classifier + PPO GT masks** + soft + EMA 12 @ 30k.
Matrix: `results/routing_ablation/phase2_causal/C2_summary_matrix.md`.

| Arm | Labels | Mean Sharpe | Routing acc | CumRet | Δ vs S1 |
|---|---|---:|---:|---:|---:|
| **S1** | forward TS | **−12.76** | 46.2 % | **−70.7 %** | — |
| C1 | causal (clf only) | −13.72 | 75.6 % | −74.4 % | −0.96 |
| **C2** | causal (full align) | −13.34 | **75.0 %** | −84.3 % | **−0.58** |

**Takeaway:** Full label alignment **does not beat S1**. High routing
accuracy (~75 %) does not translate to better trading when the
hierarchical PPO architecture still fails vs B&H. **S1 remains the
single canonical best honest configuration.** Engineering loop closed;
Path A rebuttal is the resolution.

---

## 2. Where each number is cited

| Result row | Cited in |
|---|---|
| v1 baseline WF | `doc/Rebuttal_Letter_v2_honest.md` §3.1, Table 2 row 1; `doc/Manuscript_Revision_Guide.md` §4.3 Table 2 row 1 |
| reward-only v2 WF @ 30k | Rebuttal §3.1 row 2; §5.1 results |
| **reward-only v2 @ 1M post-rebacktest** | Rebuttal §3.1 row 4; `doc/Path_A_Reframing_FILLED.md`; autopilot report |
| full v2 WF @ 30k | Rebuttal §3.1 row 3 |
| Single-split v1 (`honest_retrain`) | Rebuttal §3.1 Box: "extended-training sensitivity" |
| Single-split v2 (`honest_retrain_v2`) | Rebuttal §3.1 Box; Manuscript §4.4 sensitivity discussion |
| Single-split v3 (`honest_retrain_v3`) — best of seed | Rebuttal §3.1 Box (best realisation); Manuscript §4.4 |
| **B2 routing ablation (hard EMA 12)** | `doc/ROUTING_ABLATION_REPORT.md`; rebuttal routing-gap discussion |
| **S1 soft routing (best honest WF)** | `doc/PHASE2_SOFT_ROUTING_REPORT.md`; rebuttal Phase 2 update |
| **L1 LSTM classifier (negative ablation)** | `doc/PHASE2_SEQ_CLASSIFIER_REPORT.md` |
| **C1 causal labels (diagnostic)** | `doc/PHASE2_CAUSAL_LABEL_REPORT.md` |

## 3. Reproduction recipe

```bash
# (a) Walk-forward CV — canonical 1M run (completed 2026-05-19)
python scripts/run_walk_forward.py --label-method trend_scanning \
    --subdir walk_forward_reward_v2_1M --total-timesteps 1000000 --raw-metrics

# (a') Re-backtest only (Backtester v2.0.1 long-short fix, no retrain)
python scripts/_rebacktest_walk_forward_folds.py \
    --subdir walk_forward_reward_v2_1M --folds 1,2,3,4,5 --raw-metrics

# (b) 30k ablation runs (historical comparison)
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward_reward_v2

# (b) Single-split sensitivity (4 retrains under varying total_timesteps)
python scripts/train_and_verify.py --reviewer3-mode --raw-metrics  # produces results/verification/reviewer3_compliance.md

# (c) Regenerate every figure cited in the rebuttal and the manuscript
python scripts/_generate_figures.py

# (d) Statistical robustness (Bootstrap CIs + Bonferroni)
python scripts/_stat_walk_forward.py
```

---

## 4. Backtester long-short clip bug (v2.0.1 disclosure)

During post-training autopilot (2026-05-19) we found that
`src/backtest/backtester.py` clipped portfolio weights with
`np.clip(weights, 0.0, 3.0)`, silently converting every short
position to flat. The RL environment and reward function correctly
used signed weights in \([-1, +1]\); only the **reported backtest
metrics** were affected.

| Stage | Mean Sharpe | Mean Cum. Return | Notes |
|---|---:|---:|---|
| 1M WF, metrics at train time (clip bug) | −29.93 | −97.0 % | Shorts zeroed |
| 1M WF, **post-rebacktest v2.0.1** | **−39.57** | **−99.9 %** | `allow_short=True`, symmetric clip |
| Δ (fix impact) | **−9.64** | **−2.9 %pt** | Policy had learned to short; shorts now execute |

Fix: `Backtester` now accepts `allow_short` and `max_position`;
`train_and_verify.py` and `evaluate.py` pass these from config.
Sanity check: `scripts/_sanity_backtester_long_short.py` (4/4 PASS).

**Important:** 30 k walk-forward numbers (−12.81 mean Sharpe) were
also computed under the clip bug. A full re-backtest of the 30 k
runs is scheduled but not yet executed; treat 1M post-rebacktest as
the most conservative canonical row.

---

## 5. Honest limitations of this evidence

1. **5-fold walk-forward CV reduces train-data per fold** by ~20 %
   relative to the 80/20 single split, which is the prescribed cost
   of removing the look-ahead bias of a single 80/20 train/test
   choice. This is *why* WF mean Sharpe (−12.81) is worse than the
   best single-split run (−6.36). We treat WF numbers as the
   canonical evaluation per Reviewer #3's instruction.

2. **PPO 30 k seed instability is large.** Single-split `_v3` and
   `_v4` differ in Sharpe by 21.4 points despite identical
   configuration. The 1-million-step retrains are far more stable
   (Sharpe spread 1.76 between `_v1` and `_v2`). The revised
   manuscript acknowledges this and reports the seed-instability
   spread as a separate uncertainty band.

3. **Walk-forward CV at 1 M timesteps confirms negative performance.**
   The 90-day extension request included 1 M WF as the primary
   deliverable; it completed 2026-05-19 (~37.7 h wall-clock).
   Mean Sharpe **−39.57** (post-rebacktest) is **worse** than the
   30 k reward-v2 mean (−12.81), ruling out under-training as the
   main explanation for negative WF means.

---

## 6. Status of pending tasks (P2 / P3 / P4)

| Item | Status | Notes |
|---|---|---|
| 1M WF reward-v2 + autopilot T0–T5 | **done** | `results/walk_forward_reward_v2_1M/` |
| Path A reframing draft | **done** | `doc/Path_A_Reframing_FILLED.md` |
| Table 5 ablation rewrite with v2 numbers | **done** | propagated to `doc/Manuscript_Revision_Guide.md` §6.5 |
| Re-backtest 30k WF folds (clip bug) | **pending** | optional; 1M post-fix is canonical |
| Multi-asset Table 4 (P3) | Not addressed | Requires ETH-USDT 1h dataset |
| 6-month paper-trading Table 7 (P3) | Not addressed | Requires live paper trader |
| Manuscript Word/LaTeX §1/§4 rewrite | **pending** | templates ready in Path A filled doc |
| Git push to remote | **Held** | Per user rule |
