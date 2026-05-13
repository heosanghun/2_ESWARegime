# Statistical Robustness Report

_Generated in 157.6s, n=4391 hourly returns_

## Bootstrap 95% Confidence Intervals (block=24h, 2000 resamples)

| Metric | Estimate | 95% CI |
|--------|---------:|:------:|
| Sharpe (annualized) | 2.781 | [-0.099, 5.479] |
| Cumulative return | 1.2281 | [-0.1348, 4.8083] |

## Ledoit-Wolf Sharpe equality test (vs. benchmarks) + Bonferroni

| Benchmark | z | raw p | Bonferroni p |
|-----------|--:|------:|-------------:|
| buy_and_hold | +0.004 | 0.9971 | 1 |
| zero_rf | +0.000 | 1 | 1 |

## Interpretation

* A 95% bootstrap CI for the annualized Sharpe that excludes 0 
  rejects the null of zero risk-adjusted return.
* Significant (Bonferroni-adjusted) p-values vs. benchmarks 
  indicate the system outperforms beyond chance, while accounting 
  for the multiple comparisons inflation.