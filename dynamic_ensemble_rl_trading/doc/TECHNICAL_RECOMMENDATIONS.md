# Technical Recommendations — How to Make This System Actually Work

_Generated: 2026-05-15  (autonomous overnight work for ESWA-D-26-08980)_

This document complements `AUTONOMOUS_OVERNIGHT_REPORT.md` by being
*constructive*. The overnight analysis showed that the current
implementation cannot reproduce the paper's claims under any honest
methodology. Below are the specific changes that would give the
project a real chance at the numbers it claims, in priority order.

---

## Tier 1 — Required for *any* genuine reproducibility

### 1.1  Replace the regime classifier backbone

**Problem.** Fold-1 diagnostics show:

| Features                             | Test Acc (TS labels) |
|--------------------------------------|---------------------:|
| Visual (512) + Tech (19) + Senti (8) | 0.469 |
| Tech + Sentiment (27)                | 0.470 |
| Tech only (19)                       | 0.448 |
| Rolling 24 h return only (1)         | 0.347 |

The 512-D ResNet-18 visual features add nothing on top of plain
technical indicators. Train accuracy is 1.000 while test accuracy is
~0.47 — the XGBoost is memorising the training set, not learning a
generalisable representation.

**Concrete fix.** Replace `RegimeClassifier(XGBoost)` with a sequence
model that has temporal inductive bias and bounded capacity, e.g.:

- A small 2-layer GRU/LSTM (hidden ≤ 64) over a 96-bar window of
  *normalised* technical features.
- Or a TST-style Time-Series Transformer with attention masks
  preventing future leakage.

Important: feed the sequence model *normalised* prices and indicators
(z-score over a rolling 30-bar window), so the regime label is learned
from local structure rather than absolute price level.

### 1.2  Stop treating SMA-50 labels as "ground truth"

SMA-50 labels at time t are a deterministic function of close prices
≤ t — and those are exactly the inputs to the classifier. This is
circular and is the *only* reason the original paper hit ~90 %
classifier accuracy. Reviewer #3.3 was correct.

**Concrete fix.** Use Trend Scanning labels (already implemented in
`src/regime/trend_scanning.py`) and accept that the classifier will be
much weaker. Set realistic accuracy expectations (50-65 %) and design
the downstream PPO to be robust to that.

### 1.3  Redesign the reward functions

The current rewards (`src/env/rewards.py`):

- Bull: Sharpe − 5·tx_norm
- Bear: Sortino − 5·tx_norm
- Sideways: Bull_reward − 5·tx_norm

Empirically these do *not* lead to regime-specialised behaviour. The
Bear pool fails to exploit a −44 % market drop even when activated
correctly (`results/walk_forward_sma/fold_1`).

**Concrete fix.** Replace with:

- **Bull reward** = excess return *and* direction-alignment bonus
  when the policy goes long during up-bars.
- **Bear reward** = excess return *and* direction-alignment bonus
  when the policy goes short during down-bars; very strong penalty
  for being long during a confirmed bear sub-window.
- **Sideways reward** = volatility-adjusted excess return with strong
  drawdown penalty (e.g., Calmar over the past N steps).

The "direction-alignment bonus" should be a fraction of the absolute
log-return when the position sign matches the bar sign. This gives the
PPO a strong gradient towards the trivially correct policy in each
regime.

---

## Tier 2 — Required for *competitive* numbers

### 2.1  Move to daily or 4-hour resolution

Hourly is too noisy for the regime concept and too short for any
trend-following PPO policy to compound gains net of transaction costs.
The paper's "26 months × hourly = 19,000 bars" with 0.05 % tx + ATR
slippage means the model needs an annualised drift of ≥ 30 % *just to
break even on fees*, which is implausibly high for any honest
strategy.

**Concrete fix.** Aggregate to 4-hour or daily bars. This also reduces
the inference cost and makes the "regime persistence" assumption hold.

### 2.2  Use a longer history with more diverse regimes

The 26-month training window is too short to contain enough
distinct regimes. Walk-forward folds keep showing the classifier
overfitting because each fold has only 4500 hourly bars.

**Concrete fix.** Train on 2017-2023 inclusive (so the classifier sees
2018 bear, 2019 sideways, 2020 crash, 2021 bull, 2022 bear, 2023 bull
recovery). This requires fetching historical Binance OHLCV — well
within reach of `scripts/download_hourly_data.py`.

### 2.3  Add explicit regime-persistence regularisation

The classifier's predictions flip every few bars in the walk-forward
runs, which makes the downstream PPO routing unstable. The paper's
confidence-threshold mechanism doesn't help because confidence itself
is overconfident on out-of-sample data.

**Concrete fix.** Smooth predictions with a 6-12 bar exponential
moving average over class probabilities and re-route only when the
smoothed class changes.

---

## Tier 3 — Required for *publication-grade* methodology

### 3.1  Properly nested CV for hyperparameter selection

`run_walk_forward.py` currently uses the *outer* CV fold to both train
and evaluate. There is no inner loop for hyperparameter selection.

**Concrete fix.** Wrap each outer fold with a 3-split inner walk-forward
that picks PPO hyperparameters (learning rate, ent_coef) on the inner
test, then retrain on the full outer train with the chosen
hyperparameters before evaluating on the outer test.

### 3.2  Multi-seed PPO with reported variance

A single 5-agent pool per regime is too small to estimate variance.
Use 10 seeds and report the per-seed Sharpe distribution.

### 3.3  Comparison benchmarks must include realistic baselines

Currently the paper compares only to Buy-and-Hold and a few other RL
baselines (DQN, A2C). Add:

- Constant-mix portfolio (60/40 cash/BTC).
- Risk-parity allocation re-balanced daily.
- Moving-average crossover with realistic costs.
- Carver-style trend ensemble.

These are the academic standard for trading-system papers.

---

## Tier 4 — `paper_alignment` deletion

The `config.paper_alignment` block and the conditional logic in
`scripts/train_and_verify.py` that consume it must be **deleted** in
the next commit. They served no scientific purpose; they only existed
to fit numbers to a target. Keeping them in the public repository will
draw reviewer suspicion and undo any goodwill from the methodological
fixes above.

---

## Time and resource estimate

| Tier | Eng. weeks | Compute (CPU-h) |
|------|-----------:|----------------:|
| 1 — required for honest reproducibility | 4-6 | 50 |
| 2 — required for competitive numbers    | 6-8 | 200 |
| 3 — required for publication grade      | 2-3 | 100 |
| 4 — `paper_alignment` deletion          | 0.5 | 0 |

Total realistic effort to bring this paper to a state where the
numbers are honest *and* competitive: **3-4 months of focused work
on the methodology**, *not* on tweaking the existing pipeline.

If the deadline does not permit this, the only intellectually honest
option for the current submission is:

> **Withdraw the manuscript, complete Tier 1-3, then resubmit.**

I am happy to help execute any of these tiers when the operator
returns. None of them have been started; this overnight session was
diagnostic and corrective, not architecturally generative.
