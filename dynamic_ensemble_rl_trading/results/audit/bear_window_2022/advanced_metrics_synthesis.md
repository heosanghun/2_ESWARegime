# Audit P1.2 — Advanced Risk Metrics (2022 Bear-Market Window): Synthesis

_Generated: 2026-05-27_

**Window:** 2022-04-19 → 2022-12-19  (LUNA fold 1 + FTX fold 2, 5 879 bars)
**Mode:** `ESWA_RAW_MODE=1` (paper_alignment OFF), Backtester v2.0.1
**Annualisation factor:** 8 760 bars/year (hourly)

## 1. Primary results — deterministic PPO actions (reproducible)

| Configuration | Sharpe | **Sortino** | **Calmar** | CumRet | **MDD** | **CVaR 95% / bar** | Pain Idx | Ulcer Idx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy & Hold (passive) | −1.76 | **−2.45** | **−1.17** | **−59.79%** | **−63.34%** | **−1.737%** | 46.25% | 48.50% |
| **ATR 1.8% screen** | **−0.96** | **−1.35** | **−0.91** | **−18.99%** | **−29.80%** | **−0.027%** | 20.76% | 22.83% |
| S1 soft + EMA12 (primary honest) | −1.64 | −2.35 | −1.02 | −46.35% | −59.29% | −1.270% | 41.16% | 45.09% |
| v1 baseline (Calmar only) | — | — | −1.14 | — | −87.5% | — | — | — |

### ATR 1.8% versus Buy & Hold — defensive contributions

| Metric | Buy & Hold | ATR 1.8% screen | Improvement | Improvement (%) |
|---|---:|---:|---:|---:|
| **Maximum Drawdown** | −63.3% | **−29.8%** | **+33.5 pp** | **53% lower DD** |
| **Cumulative Return** | −59.8% | **−19.0%** | **+40.8 pp** | **68% capital preserved** |
| **Sortino Ratio** | −2.45 | **−1.35** | +1.10 | **45% better** |
| **Calmar Ratio** | −1.17 | **−0.91** | +0.26 | **22% better** |
| **CVaR 95% (per-bar)** | −1.737% | **−0.027%** | +1.71 pp | **98% better tail** |
| **Pain Index** | 46.25% | **20.76%** | −25.5 pp | **55% less underwater** |
| **Ulcer Index** | 48.50% | **22.83%** | −25.7 pp | **53% smoother DD curve** |
| Sharpe Ratio | −1.76 | −0.96 | +0.80 | 45% better |

→ **Every single risk-adjusted metric favours the ATR-gated system over passive Buy & Hold during the 2022 bear market, on a deterministic, reproducible evaluation.**

### Per-fold detail (deterministic) — ATR 1.8% screen

| Fold | Window | Event | Sharpe | Sortino | Calmar | CumRet | MDD | CVaR 95%/bar |
|----:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 2022-04-19..2022-08-19 | LUNA / Terra crash | −1.88 | −2.57 | −1.91 | −22.8% | −28.2% | −0.927% |
| 2 | 2022-08-19..2022-12-19 | FTX collapse | **+1.02** | **+1.83** | **+2.73** | **+5.0%** | **−5.7%** | −0.008% |

**Key observation:** during the FTX-collapse window, the ATR-gated system *generated positive cumulative return* (+5.0%) with a positive Sharpe (+1.02) and Sortino (+1.83), against a Buy & Hold loss of roughly −30% over the same period. This is the strongest defensive evidence in the audit.

### Per-fold detail (deterministic) — S1 soft + EMA12

| Fold | Window | Sharpe | Sortino | Calmar | CumRet | MDD | CVaR 95%/bar |
|----:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2022-04-19..2022-08-19 | −2.80 | −3.92 | −1.48 | −43.5% | −55.1% | −1.477% |
| 2 | 2022-08-19..2022-12-19 | −0.16 | −0.24 | −0.72 | −5.1% | −20.2% | −1.006% |

S1 (the fully-active honest configuration) underperforms B&H on every metric and is provided as a control for the defensive effect of the ATR gate.

## 2. Sensitivity check — stochastic PPO actions (earlier run)

The PPO inference layer (`src/agents/pool.py:131`) calls
`agent.predict(observation, deterministic=False)`, so action selection is
stochastic. A stochastic run produced:

| Configuration | Sharpe | Sortino | Calmar | CumRet | MDD | CVaR 95%/bar |
|---|---:|---:|---:|---:|---:|---:|
| ATR 1.8% screen | −0.27 | −0.39 | −0.45 | −7.77% | −25.24% | −0.027% |
| S1 soft + EMA12 | −1.34 | −1.93 | −0.98 | −40.84% | −55.25% | −1.269% |

→ ATR screen's *defensive ordering* (better than B&H on Sortino, Calmar, MDD,
CVaR, Pain, Ulcer) is **stable across the stochastic and deterministic
samples**. The exact Sharpe magnitude varies because action sampling is
non-deterministic, which is a known instability we also report
quantitatively (cf. PPO seed Sharpe spread = 21.4 in our audit).

For the manuscript, **the deterministic results are the headline numbers**
and the stochastic numbers can be cited as a robustness check / further
evidence of inference variance.

## 3. Files

```
results/audit/bear_window_2022/
  advanced_metrics.json                 (stochastic run, primary metrics)
  advanced_metrics.md
  advanced_metrics_deterministic.json   (deterministic run, headline)
  advanced_metrics_deterministic.md
  advanced_metrics_synthesis.md         (this synthesis)
  summary.json                          (existing P1.2 aggregate, headline only)
  summary.md
```

## 4. Headline sentence for the manuscript

> "Under honest evaluation with `paper_alignment` fully disabled, the proposed
> XGBoost-classifier + PPO-ensemble system equipped with an ATR 1.8% sideways
> filter delivers materially better drawdown-based risk metrics than passive
> Buy & Hold during the 2022 LUNA + FTX bear market: maximum drawdown is
> reduced from −63.3% to −29.8% (a 33.5 percentage-point reduction), Sortino
> ratio improves from −2.45 to −1.35, Calmar ratio improves from −1.17 to
> −0.91, and per-bar CVaR(95%) shrinks from −1.74% to −0.03%. Cumulative
> capital preserved over the 8-month window is 40.8 percentage points larger
> than under Buy & Hold. The system therefore exhibits clear defensive
> value as a **risk-management overlay**, even though it does not generate
> positive alpha on its own."
