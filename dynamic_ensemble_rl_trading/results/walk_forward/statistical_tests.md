# Walk-Forward Statistical Robustness Tests

_Generated: 2026-05-14T21:18:24_

**N folds:** 5    |    **Family-wise α:** 0.05    |    **Bonferroni-adjusted α:** 0.0083

## Per-metric paired comparison vs paper Table 2

Each fold's metric is treated as one observation. We report the mean across folds, the percentile bootstrap 95% CI (10,000 resamples), and a Bonferroni-adjusted CI; the test compares the fold mean to the paper's published value.

| Metric | Paper value | Fold mean | 95% CI | Bonferroni 95% CI | Paper inside CI? |
|--------|------------:|----------:|--------|--------------------|------------------|
| Sharpe Ratio | 1.89 | -20.5016 | [-23.9810, -16.1151] | [-24.5020, -14.8760] | no / Bonf:no |
| Cumulative Return | 0.893 | -0.7368 | [-0.8625, -0.5859] | [-0.8805, -0.5447] | no / Bonf:no |
| CAGR | 0.342 | -0.9611 | [-0.9971, -0.9072] | [-0.9980, -0.8872] | no / Bonf:no |
| Maximum Drawdown | -0.162 | -0.7379 | [-0.8626, -0.5886] | [-0.8805, -0.5476] | no / Bonf:no |
| Win Rate | 0.678 | 0.1013 | [0.0448, 0.1744] | [0.0344, 0.2004] | no / Bonf:no |
| Profit Factor | 2.34 | 0.3082 | [0.1987, 0.4197] | [0.1713, 0.4597] | no / Bonf:no |

## Interpretation

- A 'YES' in the last column means the paper-reported value falls inside the bootstrap 95% confidence interval estimated from the 5 walk-forward folds. A 'no' means the difference between paper and walk-forward estimate is statistically significant.
- 'Bonf:YES' applies the family-wise Bonferroni correction across 6 metrics (α/6); this is the multiple-comparison-safe test recommended by Reviewer #1.4 (Ledoit-Wolf + Bonferroni).