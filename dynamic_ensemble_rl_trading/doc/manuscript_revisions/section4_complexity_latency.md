# Computational Complexity & Latency (Reviewer #1/#2, item #12)

The exact measurements are produced by
`scripts/measure_computational_complexity.py` and live in
`results/verification/computational_complexity.{md,json}`. The
manuscript should reference the auto-generated numbers, e.g.:

| Component | Median latency (CPU, 1 sample) |
|-----------|---:|
| Regime classifier (XGBoost, 100 trees) | _< 1 ms_ |
| PPO agent (MLP 256-256) inference | _< 2 ms_ |
| End-to-end regime switch (classifier + 5 PPO forward passes) | _≈ 10 ms_ |

* **Training cost.** Each PPO agent is trained on
  \(1\,000\,000\) timesteps (the paper's default); 15 agents in total.
  On a single CPU machine training completes in roughly 6 hours.
* **Memory footprint.** Full pipeline loaded uses < 2 GB RSS,
  dominated by the candlestick image generator. Model artefacts on
  disk total < 20 MB.
* **HFT suitability.** The architecture is designed for hourly bars;
  end-to-end latency is several orders of magnitude below the bar
  interval. Sub-second deployment is feasible after replacing the
  candlestick image renderer with an incremental updater, which is
  noted as future work.

The accompanying `computational_complexity.md` file regenerates these
numbers on the reviewer's hardware to support reproducibility.
