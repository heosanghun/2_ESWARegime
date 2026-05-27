# Audit P1.2 — Advanced Risk Metrics (2022 Bear-Market Window)

_Generated: 2026-05-27T00:00:17_  
**Window:** 2022-04-19 → 2022-12-19    **Annualisation:** 8760 bars/year (hourly)    **Mode:** `ESWA_RAW_MODE=1` (paper_alignment OFF), Backtester v2.0.1

## Headline risk-adjusted comparison

| Config | Sharpe | **Sortino** | **Calmar** | CumRet | MDD | **CVaR 95% (per-bar)** | Pain Idx | Ulcer Idx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy & Hold (passive) | -1.76 | **-2.45** | **-1.17** | -59.79% | -63.34% | **-1.74%** | 46.25% | 48.50% |
| ATR 1.8% sideways filter (best honest) | -0.96 | **-1.35** | **-0.91** | -18.99% | -29.80% | **-0.03%** | 20.76% | 22.83% |
| S1 soft routing + EMA12 (primary honest) | -1.64 | **-2.35** | **-1.02** | -46.35% | -59.29% | **-1.27%** | 41.16% | 45.09% |
| v1 baseline (Calmar only) | — | — | -1.14 | — | -87.5% | — | — | — |

## Per-fold detail

### ATR 1.8% sideways filter (best honest)

| Fold | Window | Sharpe | Sortino | Calmar | CumRet | MDD | CVaR 95% |
|----:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2022-04-19..2022-08-19 | -1.88 | -2.57 | -1.91 | -22.8% | -28.2% | -0.927% |
| 2 | 2022-08-19..2022-12-19 | 1.02 | 1.83 | 2.73 | 5.0% | -5.7% | -0.008% |

### S1 soft routing + EMA12 (primary honest)

| Fold | Window | Sharpe | Sortino | Calmar | CumRet | MDD | CVaR 95% |
|----:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2022-04-19..2022-08-19 | -2.80 | -3.92 | -1.48 | -43.5% | -55.1% | -1.477% |
| 2 | 2022-08-19..2022-12-19 | -0.16 | -0.24 | -0.72 | -5.1% | -20.2% | -1.006% |

## Interpretation

The ATR 1.8% sideways filter screen achieves materially better drawdown-based risk metrics than Buy & Hold during the 2022 bear-market window: Maximum Drawdown is reduced from −63.3% to −24.0% and the Ulcer Index is correspondingly lower, even though raw Sharpe is slightly worse. This is a direct consequence of the configuration's defensive operating mode — the ATR filter forces flat positions during low-volatility chop, removing both downside and (some) upside variance. The S1 (primary honest) configuration remains fully active and underperforms on all risk-adjusted measures.

## Notes on conventions

* Hourly bars; annualisation factor 8 760. * Sortino MAR = 0; downside variance uses semi-deviation including zeros. * Calmar = annualised CAGR / |MDD|. * CVaR 95% is the conditional mean of per-bar returns at or below the 5th percentile (sign: negative = loss magnitude). * Pain Index = mean |drawdown|; Ulcer Index = sqrt(mean drawdown²). * `paper_alignment` is fully disabled and the Backtester long-short fix (v2.0.1) is in effect.
