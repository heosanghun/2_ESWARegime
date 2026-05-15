# Classifier Feature Ablation — Fold 1

Two label schemes are tested:
- **Trend Scanning** (forward-looking, the paper's revised method)
- **SMA-50** (lagging, the original method)

| Label scheme | Feature subset | Train acc | Test acc | Gap (overfit) |
|---|---|---:|---:|---:|
| Trend Scanning (forward) | full | 1.000 | 0.469 | 0.531 |
| Trend Scanning (forward) | tech_senti | 0.970 | 0.470 | 0.500 |
| Trend Scanning (forward) | tech | 0.942 | 0.448 | 0.494 |
| Trend Scanning (forward) | rolling_ret_24h | 0.468 | 0.347 | 0.121 |
| SMA-50 (lagging) | full | 1.000 | 0.907 | 0.093 |
| SMA-50 (lagging) | tech_senti | 0.999 | 0.891 | 0.108 |