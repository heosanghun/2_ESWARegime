# Audit P1.2 — 2022 Bear-Market Window

_Generated: 2026-05-26T17:56:13_

**Window:** 2022-04-19 → 2022-12-19    |    **Folds aggregated:** [1, 2]    |    **Honest mode:** `ESWA_RAW_MODE=1` (paper_alignment OFF)

## Headline comparison

| Configuration | Source | Mean Sharpe | Mean CumRet | Mean MDD | Max consec. loss days |
|---|---|---:|---:|---:|---:|
| Buy & Hold (passive) | — | -1.76 | -59.8% | -63.3% | 8 |
| v1 baseline (worst case reference) | `walk_forward` | -17.83 | -87.5% | -87.5% | — |
| S1 soft routing + EMA12 (primary honest) | `routing_ablation/phase2_soft/S1_soft_ema12` | -11.24 | -86.9% | -87.8% | — |
| ATR 1.8% sideways filter (best honest) | `autonomous/screen_extra_18` | -2.31 | -22.5% | -24.0% | — |

## Per-fold detail

### v1 baseline (worst case reference)

| Fold | Test window | Sharpe | CumRet | MDD | Win Rate | Profit Factor |
|----:|-------------|-------:|-------:|----:|---------:|--------------:|
| 1 | 2022-04-19..2022-08-19 | -12.82 | -84.9% | -84.9% | 11.1% | 0.49 |
| 2 | 2022-08-19..2022-12-19 | -22.84 | -90.2% | -90.2% | 8.4% | 0.28 |

### S1 soft routing + EMA12 (primary honest)

| Fold | Test window | Sharpe | CumRet | MDD | Win Rate | Profit Factor |
|----:|-------------|-------:|-------:|----:|---------:|--------------:|
| 1 | 2022-04-19..2022-08-19 | -11.58 | -91.0% | -91.4% | 42.9% | 0.66 |
| 2 | 2022-08-19..2022-12-19 | -10.91 | -82.9% | -84.1% | 38.8% | 0.66 |

### ATR 1.8% sideways filter (best honest)

| Fold | Test window | Sharpe | CumRet | MDD | Win Rate | Profit Factor |
|----:|-------------|-------:|-------:|----:|---------:|--------------:|
| 1 | 2022-04-19..2022-08-19 | -3.30 | -37.4% | -37.7% | 4.4% | 0.73 |
| 2 | 2022-08-19..2022-12-19 | -1.32 | -7.6% | -10.3% | 1.1% | 0.79 |


## Interpretation

Fold 1 covers the LUNA/Terra crash (May 2022) and fold 2 covers the FTX collapse (Nov 2022); jointly they form the canonical 2022 bear-market stress test demanded by Reviewer #2 in ESWA-D-26-08980.

Numbers are **post Backtester v2.0.1 long-short fix** (2026-05-19) and `paper_alignment` is fully OFF.