# Table 3 — 2022 Bear Market Case Study

_Generated: 2026-05-14T23:36:58_

Two non-overlapping walk-forward folds together cover the worst part of the 2022 BTC drawdown (Terra/LUNA collapse through FTX). All metrics are *raw* — `paper_alignment` is disabled.

## Per window (Trend Scanning labels — Reviewer #3 compliant)

| Window | B&H Cum | TS Model Cum | TS Sharpe | TS Win Rate |
|--------|--------:|-------------:|----------:|------------:|
| 2022-04-19..2022-08-19 | -0.4391 | -0.8489 | -12.82 | 0.1110 |
| 2022-08-19..2022-12-19 | -0.2685 | -0.9016 | -22.84 | 0.0844 |

## Per window (SMA-50 labels — paper original)

| Window | B&H Cum | SMA Model Cum | SMA Sharpe | SMA Win Rate |
|--------|--------:|--------------:|-----------:|-------------:|
| 2022-04-19..2022-08-19 | -0.4391 | -0.5265 | -14.22 | 0.0246 |
| 2022-08-19..2022-12-19 | -0.2685 | -0.7109 | -19.74 | 0.0748 |

## Combined 2022 bear market (compound the two windows)

| Method | Combined 2022 Cum Return |
|--------|-------------------------:|
| Buy & Hold | -0.5897 (-58.97%) |
| Trend Scanning + Long-Short | -0.9851 (-98.51%) |
| SMA-50 + Long-Short | -0.8631 (-86.31%) |

## Interpretation

The paper's central narrative is that the regime-aware ensemble should *outperform* Buy-and-Hold in bear markets thanks to its Bear-pool short bias. The honest 2022 result shows the opposite: the ensemble loses substantially more than Buy-and-Hold, in both label methods. The Bear-pool reward function fails to encode the actually profitable behaviour (short bias) — see `doc/TECHNICAL_RECOMMENDATIONS.md` §1.3 for the proposed fix.