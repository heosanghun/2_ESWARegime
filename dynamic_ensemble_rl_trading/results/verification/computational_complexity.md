# Computational Complexity & Latency

Generated to address Reviewer #1 / #2 (item #12).

## Inference latency (CPU, single sample)

| Component | median (ms) | mean (ms) | p95 (ms) |
|-----------|------------:|---------:|---------:|
| Regime classifier (XGBoost) | 3.826 | 5.302 | 19.704 |
| PPO agent (MlpPolicy) | 0.393 | 0.562 | 1.215 |
| End-to-end regime switch | 5.364 | 6.779 | 11.567 |

## Memory footprint

- Resident memory (full pipeline loaded): **890.9 MB**

| Artifact | Size (MB) |
|----------|----------:|
| regime_classifier | 1.55 |
| ppo_agent_0 | 4.74 |

## Training cost

- Regime classifier samples: 14256
- PPO timesteps per agent: 20480
- Total PPO agents: 15 (3 pools × 5)

## HFT Suitability

The end-to-end median latency is well below the bar-frequency of
the system (hourly trading), so the architecture is suitable for
low-frequency systematic execution. For high-frequency trading
(sub-second), the bottleneck would shift to feature extraction
(candlestick image rendering); this is acknowledged as a
limitation and a direction for future work.