# Dynamic Ensemble Reinforcement Learning Trading System

**Supplementary repository:** Anonymous double-blind mirror (URL supplied via the editorial system — not an author-linked GitHub account during review).  
**Manuscript ID:** ESWA-D-26-08980

> **ESWA reviewers:** start at [`doc/REVIEWER_INDEX.md`](doc/REVIEWER_INDEX.md), then run `python reproduce.py`.

A hierarchical ensemble framework for regime-aware trading on hourly BTC/USDT data, implementing **"A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes."**

This repository supports an **audit + risk-management application** revision: we disclose that originally reported Table 2 metrics are **not reproducible** under honest methodology, and we provide reproducible evidence for an **ATR 1.8% volatility-gated capital-preservation overlay** on two out-of-sample windows (2022 bear market + 2024 forward test).

> **Note on blind review:** ESWA uses double-blind peer review. This repository is intended for **anonymous supplementary material** supplied through the editorial system (or upon request), not as a public author-linked release during active review. Do not cite author names or affiliations when sharing the link with reviewers.

---

## Honesty Statement

> An earlier version of this README claimed **"100% consistency between paper, code, and data."** That claim was incorrect. A post-processing layer named `config.paper_alignment` silently rewrote reported metrics (action inversion, Buy & Hold blending, position scaling ×1.76, Sharpe capping, etc.). While preparing the code release requested by Reviewer #4, we discovered this layer, **permanently disabled it**, and recomputed all headline numbers under honest measurement (`ESWA_RAW_MODE=1`).
>
> **Honest measurement artefacts** are preserved in `doc/REVIEWER_INDEX.md`, `results/audit/`, `doc/AUTONOMOUS_OVERNIGHT_REPORT.md`, and `results/walk_forward/summary.md`. The post-processing layer itself remains in the codebase for audit purposes at `config/config.yaml` (`paper_alignment` section) and `scripts/train_and_verify.py` (`--raw-metrics` branch). The deprecated optimizer `scripts/reach_100_percent_autonomous.py` is hard-guarded with `SystemExit`.

---

## One-command reproduction (Reviewer #4)

```bash
python reproduce.py              # Bootstrap CIs + 2022 bear + OOS 2024
python reproduce.py --only ci    # Statistical tests only (~1 min)
python reproduce.py --only bear  # 2022 bear advanced metrics (~15 min)
python reproduce.py --only oos --download-data   # OOS 2024 forward test
```

See [`doc/Response_Letter_v2_english.md`](doc/Response_Letter_v2_english.md) and [`doc/REVIEWER_INDEX.md`](doc/REVIEWER_INDEX.md).

---

## ESWA-D-26-08980 — Reviewer #3 compliance (honest measurement)

| Reviewer concern | Code location | Honest result (5-fold walk-forward, raw-metrics) |
|---|---|---|
| #3.1 Look-ahead bias (FinBERT) | `src/data/finbert_sentiment.py`, `features.sentiment.model = finbert` | DeepSeek-era CSV replaced with FinBERT-rescored CSV (31,012 rows) |
| #3.2 Lagging → forward-looking labels | `src/regime/trend_scanning.py` | Trend Scanning active; classifier accuracy **46.07%** (chance ≈ 33%) |
| #3.3 K-fold → walk-forward CV | `src/validation/walk_forward_cv.py`, `scripts/run_walk_forward.py` | 5-fold expanding window completed |

### Table 2 — honest measurement vs originally reported values

| Metric | Paper (reported) | WF fold mean | 95% CI | Bonferroni 95% CI | Paper inside CI? |
|------|---:|---:|---|---|:---:|
| Sharpe Ratio | 1.89 | **−20.50** | [−23.98, −16.12] | [−24.50, −14.88] | **No** |
| Cumulative Return | 0.893 | **−0.737** | [−0.863, −0.586] | [−0.881, −0.545] | **No** |
| CAGR | 0.342 | **−0.961** | [−0.997, −0.907] | [−0.998, −0.887] | **No** |
| Maximum Drawdown | −0.162 | **−0.738** | [−0.863, −0.589] | [−0.881, −0.548] | **No** |
| Win Rate | 0.678 | **0.101** | [0.045, 0.174] | [0.034, 0.200] | **No** |
| Profit Factor | 2.34 | **0.308** | [0.199, 0.420] | [0.171, 0.460] | **No** |

All six metrics lie **outside** the Bonferroni-corrected 95% confidence interval. The originally reported Table 2 values **cannot be reproduced** under the stated honest methodology. See `doc/gap_decomposition_refined.md` for a bias-source breakdown.

### Risk-management application (ATR 1.8% overlay)

| Window | Method | Sharpe | Sortino | MDD |
|---|---|---:|---:|---:|
| 2022 LUNA+FTX bear | Buy & Hold | −1.49 | −2.07 | −63.3% |
| 2022 LUNA+FTX bear | **ATR 1.8% screen** | **+1.57** | **+2.96** | **−24.0%** |
| 2024-03..08 OOS | Buy & Hold | +0.13 | +0.18 | −32.3% |
| 2024-03..08 OOS | **ATR 1.8% screen** | **+1.96** | **+4.90** | **−1.9%** |

Details: `results/audit/bear_window_2022/`, `results/audit/oos_2024_forward/`, `doc/oos_2024_forward_report.md`.

### v2 architecture improvements (May 2026)

| ID | Change | File | Effect |
|----|--------|------|--------|
| A1 | Reward function v2 (direction-aligned shaping) | `src/env/rewards.py` | Reward spread ≈ 1× → 5× (`scripts/_sanity_reward_v2.py`) |
| A2 | Classifier regularization v2 | `src/regime/regime_classifier.py` | Reduced train–test gap |
| A3 | Visual branch optional (off by default) | `src/data/feature_fusion.py` | Observation dim 539 → 27 |

| | v1 baseline | reward-only v2 | full v2 |
|---|---:|---:|---:|
| Mean Sharpe | −20.50 ± 5.01 | **−12.81 ± 3.57** | −14.66 ± 3.23 |
| Δ vs v1 Sharpe | — | **+7.69** | +5.83 |

```bash
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward_reward_v2
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward
```

Comparison: `results/walk_forward_v2_comparison.md`

---

## System architecture

1. **Multimodal feature fusion** — technical indicators (19-D), optional ResNet-18 candlestick embeddings (512-D), FinBERT news sentiment (8-D).
2. **Regime classification** — XGBoost, Bull / Bear / Sideways, confidence threshold θ = 0.35, optional ATR sideways filter.
3. **PPO layer** — three pools of 5 agents (Bull, Bear, Sideways), regime-specific rewards.
4. **Ensemble decision** — dynamic Sharpe-based weighting, soft or hard routing.

## Key features

- Walk-forward expanding-window cross-validation
- Transaction costs: 0.05% fee + ATR-scaled slippage (~0.27% mean)
- **`paper_alignment` disabled by default** (`ESWA_RAW_MODE=1`)
- ATR 1.8% sideways volatility filter (`src/regime/atr_sideways_filter.py`)

## Reviewer item mapping (18 items)

See `doc/ESWA_18_ITEM_REVIEWER_MAPPING.md` and `doc/Response_Letter_v2_english.md`.

| # | Item | Location |
|---|------|----------|
| 1 | FinBERT sentiment (look-ahead fix) | `src/data/finbert_sentiment.py` |
| 2 | Walk-forward CV | `src/validation/walk_forward_cv.py` |
| 3 | Trend Scanning labels | `src/regime/trend_scanning.py` |
| 7 | ATR dynamic slippage | `src/backtest/slippage.py` |
| 9 | Bootstrap + Bonferroni tests | `src/evaluation/statistical_tests.py` |
| 12 | Latency / complexity | `scripts/measure_computational_complexity.py` |

### Core audit artefacts

| File | Content |
|------|---------|
| `results/audit/` | Bootstrap CIs, 2022 bear, OOS 2024, SHAP |
| `doc/Response_Letter_v2_english.md` | Point-by-point reviewer responses |
| `doc/gap_decomposition_refined.md` | Paper vs honest gap by bias source |
| `doc/audit_paper_alignment_timeline.md` | `paper_alignment` disclosure timeline |
| `doc/Manuscript_Revision_Guide.md` | Paste-ready revised sections |
| `doc/Reproducibility_Statement.md` | Data, seeds, environment, commands |

## Requirements

- Python 3.9+
- PyTorch, Stable-Baselines3, XGBoost, pandas, NumPy, TA-Lib, Gymnasium, Matplotlib

## Project structure

```
dynamic_ensemble_rl_trading/
├── src/              # Core library
├── scripts/          # Training, backtest, audit scripts
├── config/           # config.yaml (paper_alignment OFF)
├── reproduce.py      # One-command audit reproduction
├── results/audit/    # Headline audit JSON/MD
└── doc/              # Reviewer-facing documentation
```

## Data download

Data are **not** included in the repository.

- **Google Drive (paper):** https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB
- **OHLCV:** `python scripts/download_hourly_data.py` (Binance BTC/USDT 1h, 2021-10-12 .. 2023-12-19)
- **OOS 2024 OHLCV:** `python scripts/fetch_oos_2024_data.py`

Place OHLCV at `data/raw/btcusdt_1h.csv`. Candlestick images are generated in-code from OHLCV; the chart ZIP is optional.

## Installation

```bash
# Obtain the anonymous supplementary ZIP or clone URL from the editorial submission letter.
cd dynamic_ensemble_rl_trading
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Backtest only (existing models)
python scripts/train_and_verify.py --backtest-only

# Full pipeline (train classifier + PPO + backtest)
python scripts/train_and_verify.py

# Download OHLCV
python scripts/download_hourly_data.py
```

## Performance — honest measurement (walk-forward 5-fold)

> All numbers below use `paper_alignment` **disabled**. See `doc/AUTONOMOUS_OVERNIGHT_REPORT.md`.

| Metric | Paper Table 2 | v1 baseline | reward-only v2 |
|------|---:|---:|---:|
| Sharpe | 1.89 | **−20.50** | **−12.81 ± 3.57** |
| Cumulative Return | 0.893 | **−0.737** | **−0.467 ± 0.286** |
| Maximum Drawdown | −0.162 | **−0.738** | **−0.468 ± 0.286** |

Classifier accuracy (~46%) → mis-routing → long-short amplification is the primary loss mechanism under honest labels.

## Documentation index

- **Start here:** `doc/REVIEWER_INDEX.md`
- **Anonymous mirror FAQ:** `doc/ANONYMOUS_SUPPLEMENTARY_REPOSITORY.md`
- **Upload exclusions:** `doc/GITHUB_UPLOAD_EXCLUSIONS.md`
- **Full synthesis:** `doc/AUTONOMOUS_FINAL_SYNTHESIS.md`

## Citation

```
A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes
```

## License

Research and educational use.
