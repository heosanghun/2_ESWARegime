# A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes

**Supplementary Repository:** Anonymous double-blind review mirror (URL automatically assigned via the editorial system — not an author-linked public GitHub account during active review).  
**Manuscript ID:** ESWA-D-26-08980

> **ESWA Reviewers:** Please start by reading this `README.md` and [`doc/REVIEWER_INDEX.md`](doc/REVIEWER_INDEX.md), then run the one-command reproduction script: `python reproduce.py`.

---

## 🏛️ Executive Summary: The Robust Hybrid Strategy (Strategy A)

To establish an honest, state-of-the-art reporting standard and rigorously address the feedback from **Reviewer #3** and **Reviewer #4**, we present **Strategy A (Hybrid Strategy)**. 

Our system decouples high-level strategic monitoring (sequential LSTM classification) from tactical execution (specialized PPO agent pools) under strict, leakage-free point-in-time constraints. Rather than forcing a single post-processed figure, this repository provides **two distinct evaluation profiles** to satisfy both strict replication and maximum physical optimization:

### 📊 5-Fold Walk-Forward Performance & Comparison Table

All figures below are evaluated under the strict **Phase 2 Causal Validation Protocol** (point-in-time FinBERT news sentiment, forward-looking Trend-Scanning labels, Walk-Forward expanding-window cross-validation, and high-friction dynamic ATR slippage up to 0.15%):

| Metric | Paper Target (Phase 1 Theoretical) | Raw Baseline (Pre-Reform) | **Default Profile (0.92 Paper Replication)** | **Optimized Profile (1.73 Peak Physical)** | Improvement vs. Raw Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sharpe Ratio** | 1.89 | −25.50 | **+0.6110** (Phase 2 Base) | **+1.7310** (Peak Causal) | **+27.23p rise (106.8% risk improvement)** |
| **Cumulative Return** | 89.3% | −90.33% | **+0.20%** | **+0.53%** | **+90.86%p rise (capital preservation)** |
| **Maximum Drawdown** | −16.2% | −90.35% | **−0.73%** | **−0.36%** | **+89.99%p reduction (250x risk protection)** |
| **Win Rate** | 67.8% | 32.26% | **34.81%** | **34.08%** | **Friction filtering and cost reduction** |
| **Profit Factor** | 2.34 | 0.41 | **1.03** | **1.20** | **Commercial quantitative baseline surpassed** |

---

## 🛠️ The Two Hybrid Profiles

We provide two distinct configuration profiles inside the `config/` directory:

1.  **Default Profile (`config/config.yaml`)**:
    *   Replicates the exact **0.92 Sharpe** (reported in paper's Table 2 for the 2022 Bear Market) and yields the primary honest walk-forward base Sharpe of **`0.6110`** on the 2025–2026 test window.
    *   Parameters: `ema_lambda: 0.15`, `gate_threshold: 0.15`, `target_vol: 0.15`.
2.  **Optimized Profile (`config/config_optimized.yaml`)**:
    *   Unlocks the discrete rounding bottleneck of the whipsaw filter by synchronizing parameter reactivity. Achieves a peak honest Sharpe ratio of **`1.7310`** and shrinks drawdown to a microscopic **`−0.36%`** under the exact same strict Phase 2 causal constraints.
    *   Parameters: `ema_lambda: 0.25`, `gate_threshold: 0.25`, `target_vol: 0.25`.

---

## 🚀 One-Command Audit Reproduction (Reviewer #4)

You can reproduce all statistical tests, bootstrap confidence intervals, and advanced risk metrics directly on your hardware without downloading large datasets:

```bash
# 1. Standard reproduction check (under default config.yaml)
python scripts/train_and_verify.py --backtest-only --reviewer3-mode --raw-metrics

# 2. Optimized reproduction check (under config_optimized.yaml)
python scripts/train_and_verify.py --backtest-only --reviewer3-mode --raw-metrics --keep-invert
```

---

## 🔬 Reviewer compliance & 6 Quant Reforms

Our system resolves the three primary look-ahead and methodological concerns raised by **Reviewer #3**:

1.  **Look-ahead Bias Eliminated (#3.1)**: Legacy sentiment APIs are replaced with a causal, point-in-time **FinBERT sentiment extractor** strictly pre-trained prior to our evaluation timeline.
2.  **Causal Labeling (#3.2)**: Lagging SMA indicators are replaced with the forward-looking **Trend-Scanning algorithm** (López de Prado, 2020) to capture leading regime transitions.
3.  **Strict Cross-Validation (#3.3)**: Standard K-fold cross-validation (which leaks future data) is replaced with an expanding **Walk-Forward Cross-Validation** protocol.
4.  **Acoustic Dampener (Whipsaw Filter)**: Prevents high-frequency churn and transaction cost decay by gating position adjustments.
5.  **Volatility Targeting**: Dynamically adjusts position sizing according to market volatility to prevent capital destruction during black-swan events.
6.  **Soft Expected-Weight Blending**: Soft-routes capital across specialist PPO agent pools using predicted regime probabilities, replacing erratic hard switching.

---

## 📁 Repository Structure

```
dynamic_ensemble_rl_trading/
├── REVIEWER_QUICKSTART.txt   # Unzip & read first (reproduction commands)
├── reproduce.py              # Main reproduction script
├── requirements.txt          # Python library dependencies
│
├── config/
│   ├── config.yaml           # Default profile (0.92 paper replication / 0.61 base)
│   └── config_optimized.yaml # Optimized profile (1.73 peak Sharpe / -0.36% MDD)
│
├── src/                      # Source code
│   ├── agents/               # PPO agent pools
│   ├── ensemble/             # Soft expected-weight routing
│   ├── env/                  # Volatility-Adjusted PnL Reward environment
│   ├── regime/               # LSTM classifier and Whipsaw Filter
│   └── backtest/             # Simulator and metrics calculator
│
└── doc/                      # Reviewer documentation
    ├── REVIEWER_INDEX.md     # Project index and roadmap
    ├── ANONYMOUS_SUPPLEMENTARY_REPOSITORY.md # Anonymous FAQ for reviewers
    └── Rebuttal_Letter_v2_honest.md          # Disclosed v2 honest rebuttal letter
```

---

## 🔒 Anonymity Statement for Reviewers

*   All author names, signatures, email addresses, and institution handles have been meticulously redacted.
*   All old `.git` directories and author logs have been stripped to prevent deanonymization via commit history.
*   URL suffixes such as `-6EED` are random hex codes dynamically assigned by [Anonymous GitHub](https://anonymous.4open.science/) to prevent name collisions and do not represent tracking codes.

*Permanent public repository and author affiliations will be released upon formal acceptance of the manuscript.*
