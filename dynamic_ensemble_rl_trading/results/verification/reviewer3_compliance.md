# Reviewer #3 Compliance Report

_Generated: 2026-05-31T21:12:33_

## 1. Look-ahead Bias (LLM)

- Sentiment model: **finbert**
- FinBERT-rescored CSV: `data\cryptonews_finbert_2021-10-12_2023-12-19.csv` (exists: True)
- Active source: **FinBERT (pre-2020)**

## 2. Time-Series Cross-Validation

- Method: **walk_forward**
- n_splits: 5,  test_size: 0.1,  gap: 0,  embargo_hours: 24
- Implementation: `src/validation/walk_forward_cv.py`

## 3. Forward-Looking Ground Truth

- Labeling method: **trend_scanning**
- Trend Scanning horizon: 5..20, |t|>1.5
- Implementation: `src/regime/trend_scanning.py`

## Performance vs. Paper Table 2

- Average consistency: **32.9%**

| Metric | Paper | Actual |
|--------|------:|-------:|
| Sharpe Ratio | 1.89 | 1.7310 |
| Cumulative Return | 0.893 | 0.0053 |
| CAGR | 0.342 | 0.0053 |
| Maximum Drawdown | -0.162 | -0.0036 |
| Win Rate | 0.678 | 0.3408 |
| Profit Factor | 2.34 | 1.2004 |