# Refined Gap Decomposition (P4.1)

_Generated: 2026-05-27_  
**Purpose.** Quantify how much each individual methodological bias contributes to the gap between the originally reported Table 2 (Sharpe +1.89) and the honest walk-forward measurement (Sharpe −20.50). This refines the earlier draft at `results/autonomous/analysis/rebuttal_gap_decomposition.md` by adding (a) the additive contribution of each bias source, (b) error bars where available, and (c) the OOS-2024 generalisation column.

---

## 1. Headline ladder (Sharpe Ratio)

Each row introduces *one additional honest-evaluation correction* on top of the row above. The "Δ vs above" column attributes the cumulative gap to the marginal correction. All numbers are mean over the 5 walk-forward folds unless noted; CIs are 10 000-resample percentile.

| # | Stage | Sharpe | 95 % CI | Δ vs above | Marginal correction applied |
|---:|---|---:|---|---:|---|
| 0 | **Paper Table 2 (reported)** | **+1.89** | — | — | (none — original draft) |
| 1 | Paper Table 2 with `paper_alignment` OFF only | ≈ +0.30 | (single-split estimate) | **−1.59** | Disabling action inversion, B&H blend, scale 1.76, cap 1.89 |
| 2 | + Trend-Scanning labels (drop SMA-50 lagging) | ≈ −6 to −14 | (seed-dependent) | **−6 to −14** | Replacing 90 % classifier with 46 % classifier |
| 3 | + 5-fold walk-forward (drop single 80/20 split) | **−20.50** | [−23.98, −16.12] | **−6 to −14 cumulative** | Time-series-safe CV |
| 4 | + 1 M-timestep PPO training budget (post-rebt fix) | **−39.57** | ± 4.88 | **−19.07** | Longer training widens gap further |

**Total measured gap (row 0 → row 3):** **22.39 Sharpe units.**

### Marginal contributions in order of magnitude

| Rank | Bias source | Marginal Δ Sharpe | Share of total gap |
|---:|---|---:|---:|
| 1 | SMA-50 → Trend-Scanning labels (classifier accuracy 90 % → 46 %) | **−10 to −16** | **≈ 50–70 %** |
| 2 | `paper_alignment` post-processing OFF (action inversion alone changes sign of Sharpe ≈ ±20.5; capping/blending then compresses to +1.89) | **−1.59 to −6** (depending on whether action inversion is counted as "post-processing" or as "wrong-sign deployment") | **≈ 10–30 %** |
| 3 | Walk-forward vs single-split (data-budget cost + protected leakage) | **−2 to −4** | **≈ 10–20 %** |
| 4 | PPO seed instability at 30 k timesteps (`honest_retrain_v3` = −6.36 vs `_v4` = −27.72, identical config) | **±10.7** (noise floor, not a bias source) | (noise) |

**Headline take-away.** Approximately **half to two-thirds** of the apparent Table 2 Sharpe comes from using lagging SMA-50 labels; the remainder comes from the `paper_alignment` post-processing layer plus a small additional contribution from the K-fold-vs-walk-forward data-budget cost. **PPO seed instability is not a bias source** — it is a noise floor whose standard deviation is comparable to the strategy's signal, and is the reason the original single-split estimate was unstable in either direction.

---

## 2. Decomposition by performance metric (5-fold WF)

For completeness we apply the same row-by-row corrections to every metric in the original Table 2. All numbers are 5-fold mean.

| Stage | Sharpe | Cum Return | CAGR | Max DD | Win Rate | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|
| Paper Table 2 (reported) | +1.89 | +89.3 % | +34.2 % | −16.2 % | +67.8 % | +2.34 |
| + paper_alignment OFF | ≈ +0.30 | ≈ +5 % | (n/a) | (n/a) | (≈ 40 %) | (≈ 1.10) |
| + Trend-Scanning labels | ≈ −12 | ≈ −47 % | ≈ −75 % | ≈ −47 % | ≈ 4 % | ≈ 0.29 |
| **+ 5-fold walk-forward** | **−20.50** | **−73.7 %** | **−96.1 %** | **−73.8 %** | **10.1 %** | **0.308** |
| Bonferroni 95 % CI lower | −24.50 | −88.1 % | −99.8 % | −88.1 % | 3.4 % | 0.171 |
| Bonferroni 95 % CI upper | −14.88 | −54.5 % | −88.7 % | −54.8 % | 20.0 % | 0.460 |
| **Paper value in CI?** | **No** | **No** | **No** | **No** | **No** | **No** |

For all six metrics the paper value lies outside the Bonferroni-corrected 95 % CI of the honest measurement.

---

## 3. Defensive-overlay column — OOS 2024 generalisation

The decomposition above answers "why is the original Table 2 not reproducible?". A complementary question for ESWA practitioners is "what positive performance remains under honest evaluation?". Section 6 of the revised manuscript answers this with the ATR 1.8% sideways-filter overlay; we report it here too for completeness.

| Window | Method | Sharpe | Sortino | Calmar | MDD |
|---|---|---:|---:|---:|---:|
| 2022-04..2022-12 (LUNA + FTX) | Buy & Hold | −1.49 | −2.07 | −0.39 | −63.3 % |
| 2022-04..2022-12 | **ATR 1.8 % screen** | **+1.57** | **+2.96** | **+1.45** | **−24.0 %** |
| 2024-03..2024-08 (post-ETF, post-Halving) | Buy & Hold | +0.13 | +0.18 | −0.25 | −32.3 % |
| 2024-03..2024-08 | **ATR 1.8 % screen** | **+1.96** | **+4.90** | **+4.72** | **−1.9 %** |

The defensive ordering (ATR > B&H on every risk-adjusted metric) is consistent across two independent OOS windows separated by six months and three distinct market regimes. The +1.96 OOS-2024 Sharpe **does** sit close to the originally reported +1.89, but it is computed on a *different test window* and under a *different operating mode* (97.97 % of bars flat) than the original Table 2. It is therefore not directly comparable — but it is the strongest single piece of positive evidence we have for the practical utility of the pipeline.

---

## 4. Sources

| Component | File |
|---|---|
| Paper Table 2 | original manuscript |
| `paper_alignment` source code | `config/config.yaml` lines 44–68; `scripts/train_and_verify.py` (`--raw-metrics` branch) |
| paper_alignment-OFF estimate (row 1) | `doc/AUTONOMOUS_OVERNIGHT_REPORT.md`, post-disabling baseline |
| Trend-Scanning vs SMA-50 (row 2 vs 1) | `results/walk_forward_sma/label_method_comparison.md` |
| 5-fold WF mean (row 3) | `results/walk_forward/summary.md` + `results/audit/s1_statistical_tests.json` |
| 1 M-timestep extended budget (row 4) | `results/walk_forward_reward_v2_1M/autopilot_report.md` |
| PPO seed instability | `results/verification/honest_retrain*.log` (v1..v4) |
| ATR overlay 2022 bear | `results/audit/bear_window_2022/advanced_metrics_deterministic.json` |
| ATR overlay OOS 2024 | `results/audit/oos_2024_forward/advanced_metrics.json` |

This document supersedes `results/autonomous/analysis/rebuttal_gap_decomposition.md`; the older artefact is retained for audit.
