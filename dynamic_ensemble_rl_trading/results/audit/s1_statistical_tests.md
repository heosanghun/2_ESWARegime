# Walk-Forward Statistical Robustness Tests — S1 soft routing + EMA12 (primary honest)

_Generated: 2026-05-26T17:55:59_

**Source:** `results/routing_ablation/phase2_soft/S1_soft_ema12/`    |    **N folds:** 5    |    **Family-wise α:** 0.05    |    **Bonferroni α:** 0.0083    |    **Bootstrap resamples:** 10,000

## Per-metric paired comparison vs paper Table 2

Each fold's metric is treated as one observation. We report the mean across folds, the percentile bootstrap 95% CI, and a Bonferroni-adjusted CI; the test compares the fold mean to the paper's published value (PAPER_METRICS in train_and_verify.py).

| Metric | Paper value | Fold mean | 95% CI | Bonferroni 95% CI | Paper inside CI? |
|--------|------------:|----------:|--------|--------------------|------------------|
| Sharpe Ratio | 1.89 | -12.7625 | [-15.5786, -10.8747] | [-16.8070, -10.6574] | no / Bonf:no |
| Cumulative Return | 0.893 | -0.7072 | [-0.8397, -0.5488] | [-0.8776, -0.4796] | no / Bonf:no |
| CAGR | 0.342 | -0.9481 | [-0.9936, -0.8754] | [-0.9976, -0.8408] | no / Bonf:no |
| Maximum Drawdown | -0.162 | -0.7137 | [-0.8468, -0.5589] | [-0.8849, -0.4904] | no / Bonf:no |
| Win Rate | 0.678 | 0.3575 | [0.3058, 0.4092] | [0.2842, 0.4176] | no / Bonf:no |
| Profit Factor | 2.34 | 0.6128 | [0.5526, 0.6559] | [0.5297, 0.6589] | no / Bonf:no |

## Interpretation

- A 'YES' in the last column means the paper-reported value falls inside the bootstrap 95% confidence interval estimated from the 5 walk-forward folds. A 'no' means the difference between paper and walk-forward estimate is statistically significant.
- 'Bonf:YES' applies the family-wise Bonferroni correction across 6 metrics (α/6); this is the multiple-comparison-safe test recommended by Reviewer #1.4 (Ledoit-Wolf + Bonferroni).