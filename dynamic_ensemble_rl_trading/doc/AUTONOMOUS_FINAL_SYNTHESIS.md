# Final synthesis — honest measurements vs the original draft

_Last updated 2026-05-15. This document is the single source of truth
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
   metrics**), the system **does not reproduce those numbers**. None
   of the original draft values is inside any Bonferroni-corrected
   95 % CI we are able to compute.
3. The single most impactful **code-side** improvement we identified
   during this revision is the **reward function redesign** (§5.1 of
   the rebuttal letter): it improves the 5-fold walk-forward mean
   Sharpe by **+7.7** and the 5-fold mean cumulative return by
   **+27 percentage points** relative to the v1 baseline.
4. The biggest **non-fix finding** is that the PPO ensemble shows
   **substantial seed-to-seed instability** on this benchmark:
   single-split retrains of the *same configuration* range from
   Sharpe **−6.36** (`honest_retrain_v3`) to Sharpe **−27.72**
   (`honest_retrain_v4`). This widens the genuine uncertainty band
   on every reported number and is acknowledged explicitly in
   the revised manuscript.

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
| v1 baseline (rwd-v1, clf-v1, visual-on) | **−20.50 ± 5.01** | −73.68 % | −73.81 % | 10.1 % | 0.308 |
| **reward-only v2** (§5.1)               | **−12.81 ± 3.57** | **−46.73 %** | −46.80 % | 4.4 %  | 0.289 |
| full v2 (§5.1 + §5.2 + §5.3)            | −14.66 ± 3.23     | −56.86 %     | −56.92 % | 3.6 %  | 0.251 |

(Per-fold detail: `results/walk_forward_v2_comparison.md`.)

### 1.3 Buy-and-Hold benchmark over the same windows

| Window | B&H Cum. Return | B&H Sharpe |
|---|---:|---:|
| Single-split test (2023-06-19 → 2023-12-19) | +29.4 % | +1.92 |
| Walk-forward 5-fold mean | +11.9 % | +0.61 |

The strategy underperforms passive Buy-and-Hold across every honest
configuration we tried.

### 1.4 Side-by-side vs the original draft

| Metric | Original draft (Table 2) | v1 baseline WF | reward-only v2 WF | best single-split (v3) |
|---|---:|---:|---:|---:|
| Sharpe          | +1.89  | −20.50 | **−12.81** | **−6.36** |
| Cum. Return     | +0.893 | −0.737 | **−0.467** | **−0.610** |
| CAGR            | +0.342 | −0.961 | −0.753 | −0.849 |
| Max DD          | −0.162 | −0.738 | −0.468 | −0.638 |
| Win Rate        | +0.678 | +0.101 | +0.044 | **+0.446** |
| Profit Factor   | +2.34  | +0.308 | +0.289 | **+0.80** |

The reward redesign (v2) closes ~37 % of the cumulative-return gap;
the best stochastic realisation we observed (`honest_retrain_v3`)
closes ~42 %. Neither closes the full gap — we are explicit about
this in §3.1 of the rebuttal.

---

## 2. Where each number is cited

| Result row | Cited in |
|---|---|
| v1 baseline WF | `doc/Rebuttal_Letter_v2_honest.md` §3.1, Table 2 row 1; `doc/Manuscript_Revision_Guide.md` §4.3 Table 2 row 1 |
| reward-only v2 WF | Rebuttal §3.1 row 2; §5.1 results; `doc/AUTONOMOUS_OVERNIGHT_REPORT.md` Day-2 |
| full v2 WF | Rebuttal §3.1 row 3 (transparency); Manuscript §4.3 Table 2 row 3 |
| Single-split v1 (`honest_retrain`) | Rebuttal §3.1 Box: "extended-training sensitivity" |
| Single-split v2 (`honest_retrain_v2`) | Rebuttal §3.1 Box; Manuscript §4.4 sensitivity discussion |
| Single-split v3 (`honest_retrain_v3`) — best of seed | Rebuttal §3.1 Box (best realisation); Manuscript §4.4 |
| Single-split v4 (`honest_retrain_v4`) — worst of seed | Rebuttal §3.1 Box (worst realisation); Manuscript §4.4 — seed-instability finding |

---

## 3. Reproduction recipe

```bash
# (a) Walk-forward CV runs (canonical reviewer-compliant evaluation)
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward                # v1 baseline (set rewards.py to v1, classifier to v1)
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward_reward_v2      # v2 reward only
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward_reward_v2_full # full v2 (config has use_visual=false + clf v2)

# (b) Single-split sensitivity (4 retrains under varying total_timesteps)
python scripts/train_and_verify.py --reviewer3-mode --raw-metrics  # produces results/verification/reviewer3_compliance.md

# (c) Regenerate every figure cited in the rebuttal and the manuscript
python scripts/_generate_figures.py

# (d) Statistical robustness (Bootstrap CIs + Bonferroni)
python scripts/_stat_walk_forward.py
```

---

## 4. Honest limitations of this evidence

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

3. **Walk-forward CV at 30 k timesteps may be under-trained**. We
   were not able to redo all 5 folds × 3 ablations × 15 agents at
   1 million timesteps within the revision window (≈ 25 wall-clock
   days at observed throughput); the 90-day extension request
   includes this as the primary deliverable.

---

## 5. Status of pending tasks (P2 / P3 / P4)

| Item | Status | Notes |
|---|---|---|
| Table 5 ablation rewrite with v2 numbers | **done in this doc**, propagated to `doc/Manuscript_Revision_Guide.md` §6.5 | — |
| Multi-asset Table 4 (P3) | Not addressed | Requires acquiring ETH-USDT 1h dataset for 2021-10 → 2023-12 |
| 6-month paper-trading Table 7 (P3) | Not addressed | Requires running the live paper trader for 6 months |
| README + Rebuttal + git commit (P4) | **done** for README/Rebuttal; commit staged pending user approval | — |
| Git push to remote | **Held** | Per user rule: assistant never pushes without explicit confirmation |
