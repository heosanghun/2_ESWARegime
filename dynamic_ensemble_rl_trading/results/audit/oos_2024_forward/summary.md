# OOS 2024 Forward Test — Advanced Risk Metrics

_Generated: 2026-05-27T00:13:21_  
**Window:** 2024-03-01 → 2024-08-31    **Training cutoff (fold 5):** 2023-08-19    **OOS gap:** ≥ 6 months    **Mode:** `ESWA_RAW_MODE=1` deterministic, Backtester v2.0.1

## Headline comparison

| Config | Sharpe | **Sortino** | **Calmar** | CumRet | MDD | **CVaR 95%/bar** | Pain Idx | Ulcer Idx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy & Hold (passive) | 0.13 | **0.18** | **-0.25** | -4.1% | -32.3% | **-1.480%** | 11.735% | 13.221% |
| ATR 1.8% sideways filter (best honest) | 1.96 | **4.90** | **4.72** | 4.4% | -1.9% | **-0.001%** | 1.182% | 1.295% |
| S1 soft routing + EMA12 (primary honest) | 0.42 | **0.70** | **0.41** | 1.9% | -9.3% | **-0.202%** | 3.383% | 3.941% |

## Routing diagnostics

- **ATR 1.8% sideways filter (best honest)** — n_steps=4392, sideways_pct=0.0, regime_switch_count=173, atr_filter_pct=0.979735883424408, routing_accuracy=0.5040983606557377
- **S1 soft routing + EMA12 (primary honest)** — n_steps=4392, sideways_pct=0.0, regime_switch_count=173, atr_filter_pct=None, routing_accuracy=0.5040983606557377

## Sentiment caveat

The OOS 2024 forward test uses a synthetic neutral news placeholder (`data/cryptonews_finbert_2024-03-01_2024-08-31.csv`) generated at one article every 3 hours. This mirrors the placeholder-style schema observed in the original 2021-2023 dataset and removes any chance of look-ahead bias from a 2025-era LLM having re-scored 2024 headlines. Consequently, the sentiment feature contribution during OOS 2024 is effectively zero, isolating the price-dynamics + regime-classifier pathway.
