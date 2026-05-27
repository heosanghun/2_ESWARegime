# Manuscript vs codebase performance comparison

Source: manuscript Table 2 (Proposed Method) vs current codebase backtest output.

Generated: 2026-02-12 10:11:15

| Metric | Manuscript | Codebase | Difference | Consistency (%) |
|--------|------------|----------|------------|-----------------|
| Sharpe Ratio | 2.45 | 3.0834 | 0.6334 | 74.1% |
| Cumulative Return | 1.23 | 0.9719 | -0.2581 | 79.0% |
| CAGR | 0.41 | 2.9361 | 2.5261 | 0.0% |
| Maximum Drawdown | -0.15 | -1.0000 | -0.8500 | 0.0% |
| Win Rate | 0.58 | 0.2598 | -0.3202 | 44.8% |
| Profit Factor | 2.1 | 42.7446 | 40.6446 | 0.0% |

## Summary

- **Per-metric consistency:** see table above.
- **Average consistency:** **33.0%** (arithmetic mean over six metrics).

Note: consistency (%) is higher when the codebase value is closer to the manuscript value. Numbers may differ if the test window or data differ from the manuscript.
