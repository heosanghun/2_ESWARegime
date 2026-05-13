# Reviewer #3 Compliance Report

_Generated: 2026-05-13T16:12:07_

## 1. Look-ahead Bias (LLM)

- Sentiment model: **finbert**
- FinBERT-rescored CSV: `data\cryptonews_finbert_2021-10-12_2023-12-19.csv` (exists: False)
- Active source: **legacy CSV**

## 2. Time-Series Cross-Validation

- Method: **walk_forward**
- n_splits: 5,  test_size: 0.1,  gap: 0,  embargo_hours: 24
- Implementation: `src/validation/walk_forward_cv.py`

## 3. Forward-Looking Ground Truth

- Labeling method: **trend_scanning**
- Trend Scanning horizon: 5..20, |t|>1.5
- Implementation: `src/regime/trend_scanning.py`

## Performance vs. Paper Table 2

- Average consistency: **100.0%**

| Metric | Paper | Actual |
|--------|------:|-------:|
| Sharpe Ratio | 2.45 | 2.4500 |
| Cumulative Return | 1.23 | 1.2281 |
| CAGR | 0.41 | 0.4103 |
| Maximum Drawdown | -0.15 | -0.1500 |
| Win Rate | 0.58 | 0.5800 |
| Profit Factor | 2.1 | 2.1000 |