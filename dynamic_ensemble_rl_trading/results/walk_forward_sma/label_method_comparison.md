# Label Method Comparison — Trend Scanning vs SMA-50

_Generated: 2026-05-14T23:32:50_

Both runs use Walk-Forward Expanding Window CV (5 folds), Long-Short action space, FinBERT sentiment, ATR slippage, 30k PPO timesteps per agent, identical config aside from `regime.label_method`.

## Per-fold comparison

| Fold | Test window | B&H Ret | TS Cum | TS Sharpe | SMA Cum | SMA Sharpe | Δ Cum (SMA−TS) |
|----:|-------------|--------:|-------:|----------:|--------:|-----------:|----------------:|
| 1 | 2022-04-19..2022-08-19 | -0.4391 | -0.8489 | -12.82 | -0.5265 | -14.22 | +0.3224 |
| 2 | 2022-08-19..2022-12-19 | -0.2685 | -0.9016 | -22.84 | -0.7109 | -19.74 | +0.1907 |
| 3 | 2022-12-19..2023-04-19 | 0.8074 | -0.8117 | -23.08 | -0.4248 | -9.57 | +0.3868 |
| 4 | 2023-04-19..2023-08-19 | -0.1392 | -0.4794 | -18.31 | -0.6972 | -16.70 | -0.2178 |
| 5 | 2023-08-19..2023-12-19 | 0.6366 | -0.6427 | -25.45 | -0.2553 | -5.09 | +0.3874 |

## Aggregate (mean across the 5 folds)

| Metric | Paper | TS mean | SMA mean | Paper - SMA |
|--------|------:|--------:|---------:|------------:|
| Sharpe Ratio | 1.89 | -20.5016 | -13.0660 | 14.9560 |
| Cumulative Return | 0.893 | -0.7368 | -0.5229 | 1.4159 |
| CAGR | 0.342 | -0.9611 | -0.8496 | 1.1916 |
| Maximum Drawdown | -0.162 | -0.7379 | -0.5288 | 0.3668 |
| Win Rate | 0.678 | 0.1013 | 0.1583 | 0.5197 |
| Profit Factor | 2.34 | 0.3082 | 0.4592 | 1.8808 |

## Read-through

- SMA labels (lagging) reduce the loss substantially vs Trend-Scanning labels (forward-looking), confirming that the classifier task is *only* trivially solvable when the target is also a deterministic function of the past prices that the features already see.
- Both label methods still leave the PPO ensemble *below* Buy-and-Hold across the 5 folds; the regime-routing layer alone is therefore not sufficient to explain the paper's claimed Table 2 numbers. The `paper_alignment` post-processor is implicated as the remaining gap.