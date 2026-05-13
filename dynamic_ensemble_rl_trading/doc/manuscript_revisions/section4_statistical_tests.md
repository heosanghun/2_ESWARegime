# Statistical Robustness (Reviewer #2/#4, item #9)

`scripts/run_statistical_tests.py` produces a fresh report at
`results/verification/statistical_tests.{md,json}` summarising:

1. **Block bootstrap CIs.** A Politis–Romano stationary bootstrap with
   geometric block length \(\bar\ell=24\) hours and 2,000 resamples
   provides 95 % confidence intervals for both the annualised Sharpe
   ratio and the cumulative return. The intervals are reported in
   the manuscript next to the point estimates.
2. **Ledoit–Wolf Sharpe equality test.** We test
   \(H_0\!:\,\mathrm{SR}_{\text{Strategy}} = \mathrm{SR}_{\text{Benchmark}}\)
   against a two-sided alternative for two benchmarks:
   * Buy-and-Hold BTC, and
   * the zero (risk-free) baseline.
3. **Bonferroni correction.** Because two benchmark comparisons are
   reported simultaneously, raw p-values are multiplied by 2 and
   clipped at 1 for family-wise error control.

This addresses the reviewers' explicit request for confidence
intervals, hypothesis testing, and multiple-comparison adjustment.

| Test | Reporting field | Source |
|------|-----------------|--------|
| Sharpe CI | Section 4.2 Table 3 | `statistical_tests.md` |
| Cum. Return CI | Section 4.2 Table 3 | `statistical_tests.md` |
| LW-Sharpe vs B&H, raw p | Section 4.2 Table 4 | `statistical_tests.md` |
| LW-Sharpe vs B&H, Bonferroni p | Section 4.2 Table 4 | `statistical_tests.md` |
| LW-Sharpe vs RF=0, both ps | Section 4.2 Table 4 | `statistical_tests.md` |
