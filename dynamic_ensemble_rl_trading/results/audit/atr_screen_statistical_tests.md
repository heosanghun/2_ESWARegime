# Walk-Forward Statistical Robustness Tests — ATR 1.8% sideways filter screen (no retrain)

_Generated: 2026-05-26T17:55:46_

**Source:** `results/autonomous/screen_extra_18/`    |    **N folds:** 5    |    **Family-wise α:** 0.05    |    **Bonferroni α:** 0.0083    |    **Bootstrap resamples:** 10,000

## Per-metric paired comparison vs paper Table 2

Each fold's metric is treated as one observation. We report the mean across folds, the percentile bootstrap 95% CI, and a Bonferroni-adjusted CI; the test compares the fold mean to the paper's published value (PAPER_METRICS in train_and_verify.py).

| Metric | Paper value | Fold mean | 95% CI | Bonferroni 95% CI | Paper inside CI? |
|--------|------------:|----------:|--------|--------------------|------------------|
| Sharpe Ratio | 1.89 | -2.3029 | [-3.1236, -1.4822] | [-3.3052, -1.2402] | no / Bonf:no |
| Cumulative Return | 0.893 | -0.1030 | [-0.2412, -0.0193] | [-0.3027, -0.0106] | no / Bonf:no |
| CAGR | 0.342 | -0.2317 | [-0.5022, -0.0571] | [-0.6155, -0.0318] | no / Bonf:no |
| Maximum Drawdown | -0.162 | -0.1102 | [-0.2484, -0.0213] | [-0.3060, -0.0124] | YES / Bonf:YES |
| Win Rate | 0.678 | 0.0117 | [0.0010, 0.0286] | [0.0006, 0.0355] | no / Bonf:no |
| Profit Factor | 2.34 | 0.4698 | [0.1951, 0.7236] | [0.1002, 0.7528] | no / Bonf:no |

## Interpretation

- A 'YES' in the last column means the paper-reported value falls inside the bootstrap 95% confidence interval estimated from the 5 walk-forward folds. A 'no' means the difference between paper and walk-forward estimate is statistically significant.
- 'Bonf:YES' applies the family-wise Bonferroni correction across 6 metrics (α/6); this is the multiple-comparison-safe test recommended by Reviewer #1.4 (Ledoit-Wolf + Bonferroni).