# Dynamic Ensemble Reinforcement Learning Trading System

**Repository:** https://github.com/heosanghun/2_ESWARegime

A robust hierarchical ensemble framework for responding to market regime changes in financial trading, implementing the paper **"A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes"**.

---

## 정직성 선언 (Honesty Statement)

> 이 README의 이전 버전은 "**논문·코드·데이터 100% 일치**"라고 기재되어
> 있었으나, 이는 `config.paper_alignment` 후처리 계층(액션 반전,
> Buy & Hold 혼합, 포지션 ×1.76 스케일, Sharpe 캡 등)이 측정치를 무성으로
> 재기록한 결과였습니다. ESWA 리뷰어 #4의 "GitHub 코드 공개" 요구를 적용
> 하는 과정에서 이 사실을 발견하여, 후처리 계층을 비활성화한 정직한 측정
> 으로 모든 결과를 재산출했습니다.
>
> **정직한 측정 결과**는 `doc/AUTONOMOUS_OVERNIGHT_REPORT.md`,
> `results/walk_forward/summary.md`,
> `results/walk_forward/statistical_tests.md`,
> `results/walk_forward/table1_classifier_per_fold.md` 에 보존되어 있습니다.
> 후처리 계층의 코드 위치 또한 `config/config.yaml` 의 `paper_alignment`
> 섹션과 `scripts/train_and_verify.py` 의 `--raw-metrics` 분기에서 직접
> 확인할 수 있습니다.

---

## ESWA-D-26-08980 — Reviewer 3 대응 현황 (정직한 측정 기준)

| 리뷰어 지적 | 코드 위치 | 정직한 측정 결과 (Walk-Forward 5-fold, raw-metrics) |
|---|---|---|
| #3.1 Look-ahead Bias (FinBERT 교체) | `data/cryptonews_finbert_2021-10-12_2023-12-19.csv` (재스코어 31,012행), `features.sentiment.model = finbert` | DeepSeek-R1 → FinBERT 교체 완료. 측정에 적용됨. |
| #3.2 후행 라벨 → 선행 라벨 | `src/regime/trend_scanning.py` | Trend Scanning (López de Prado) 활성. 분류기 정확도: **46.07%** (mean, 3-class chance ≈ 33%). |
| #3.3 K-fold → Walk-Forward | `src/validation/walk_forward_cv.py`, `scripts/run_walk_forward.py` | 5-fold expanding window 실행 완료 (91 분). |

### Table 2 정직한 측정값 vs 논문 발표값

| 지표 | 논문 발표값 | Walk-Forward fold 평균 | 95% CI (10,000 부트스트랩) | Bonferroni-corrected 95% CI | 논문값이 CI 안? |
|------|---:|---:|---|---|---|
| Sharpe Ratio       | 1.89  | **−20.50** | [−23.98, −16.12] | [−24.50, −14.88] | **아니오** |
| Cumulative Return  | 0.893 | **−0.737** | [−0.863, −0.586] | [−0.881, −0.545] | **아니오** |
| CAGR               | 0.342 | **−0.961** | [−0.997, −0.907] | [−0.998, −0.887] | **아니오** |
| Maximum Drawdown   | −0.162| **−0.738** | [−0.863, −0.589] | [−0.881, −0.548] | **아니오** |
| Win Rate           | 0.678 | **0.101**  | [0.045, 0.174]   | [0.034, 0.200]   | **아니오** |
| Profit Factor      | 2.34  | **0.308**  | [0.199, 0.420]   | [0.171, 0.460]   | **아니오** |

모든 6개 지표에서 **Bonferroni 보정 95% 신뢰구간 밖**에 논문 발표값이
위치합니다. 즉, 정직한 측정치와 논문 발표값의 차이는 통계적으로 유의
하며(다중 비교 보정 후에도), **현재 코드/방법론으로는 논문 발표값을
재현할 수 없음**을 의미합니다.

이 격차의 1차 원인은 분류기 정확도(46%)이며, Long-Short 액션 공간이
이 오류를 증폭시키고 있습니다. 자세한 분석과 권고는
`doc/AUTONOMOUS_OVERNIGHT_REPORT.md` 를 참고하세요.

### v2 아키텍처 개선 (May 2026)

리뷰어 지적과 정직한 측정 결과를 바탕으로 다음 세 가지를 v2로
구현했습니다 (모두 `config/config.yaml` 의 플래그로 토글 가능).

| ID | 개선 내용 | 파일 | 기대 효과 |
|----|-----------|------|----------|
| A1 | **Reward function v2** — direction-aligned shaping. 매 step 마다 `α·w·r + regime shaping − cost drag` 형태. v1 의 30-bar Sortino 가 만들던 sparse 신호 대신 dense gradient 제공. | `src/env/rewards.py` | 정답-vs-오답 reward spread ≈ **1× → 5×** (sanity: `scripts/_sanity_reward_v2.py`) |
| A2 | **Classifier regularization v2** — `max_depth=4`, `n_estimators=200` + `early_stopping=30`, `colsample_bytree=0.7`, `reg_lambda=1.0`. v1 의 `colsample`/`subsample` 설정이 생성자에 전달되지 않던 버그도 함께 수정. | `src/regime/regime_classifier.py`, `scripts/train_and_verify.py` | Train-Test 격차 축소 (100% → 더 작은 수치, walk_forward_reward_v2 에서 측정) |
| A3 | **Visual branch 선택적 제거** — `features.use_visual: false` 가 기본값. ResNet-18 512-dim 노이즈 제거. 분류기 ablation 에서 visual 기여 ≈ 0 으로 확인 완료. | `src/data/feature_fusion.py`, `config.features.use_visual` | PPO 관측 차원 539 → 27, 학습 신호 SNR 개선 |

v2 결과 (모두 정직한 측정, raw-metrics 모드):

| | v1 baseline | reward-only v2 | full v2 (rwd+clf+nv) |
|---|---:|---:|---:|
| **Mean Sharpe** | -20.50 ± 5.01 | **-12.81 ± 3.57** | -14.66 ± 3.23 |
| **Mean Cum Return** | -73.68% ± 17.4% | **-46.73% ± 28.6%** | -56.86% ± 18.5% |
| **Δ vs v1 Sharpe** | — | **+7.69** | +5.83 |

**최종 결론:** **Reward function v2 단독이 가장 효과적** (Sharpe +7.69 개선).
분류기 정규화와 visual_off는 분류 정확도 자체는 개선하지만 PPO PnL로 이어지지 않음 — PPO 학습 예산(30k steps) 부족이 원인으로 추정. 90일 연장 기간에 해결 예정.

재현:

```bash
# v2 reward only (현재 권장 설정)
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward_reward_v2

# full v2 (reward + classifier + visual_off, ablation)
# config/config.yaml 의 features.use_visual: false + classifier v2 defaults
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward_reward_v2_full

# v1 baseline 재현
python scripts/run_walk_forward.py --label-method trend_scanning --subdir walk_forward
```

세 실험 비교: `results/walk_forward_v2_comparison.md`

---

## Overview

This system implements a four-layer hierarchical architecture that adapts to market regime changes through real-time regime classification and performance-based dynamic weight allocation. The framework consists of specialized agent pools for different market conditions (Bull, Bear, Sideways) and uses ensemble learning to aggregate their decisions.

## System Architecture

1. **Multimodal Feature Fusion Layer**: Candlestick images (CNN), technical indicators, and news sentiment → unified state vector.
2. **Market Regime Classification Layer**: XGBoost, Bull/Bear/Sideways, confidence-based selection (theta = 0.6).
3. **PPO Reinforcement Learning Layer**: Three pools of 5 PPO agents each, regime-specific reward functions.
4. **Ensemble Decision Layer**: Dynamic weighting (30-day Sharpe, temperature T = 10), policy aggregation.

## Key Features

- Walk-Forward Expanding Window Cross-Validation
- Backtesting with transaction costs (0.05% fee, 0.02% slippage)
- Paper alignment options for Table 2 metric comparison

## ESWA-D-26-08980 — 18개 리뷰어 항목 전수 대응

본 저장소는 ESWA Major Revision의 18개 리뷰어 지적을 **모두** 코드 또는 문서 산출물로 대응했습니다. 자세한 매핑은 `doc/ESWA_18항목_최종정리표.md` 와 `doc/Rebuttal_Letter_draft.md` 를 참고하세요.

### 핵심 신규 모듈

| # | 항목 | 위치 |
|---|------|------|
| 1 | FinBERT 감성분석 (Look-ahead bias 제거) | `src/data/finbert_sentiment.py` |
| 2 | Walk-Forward / Purged K-Fold CV | `src/validation/walk_forward_cv.py` |
| 3 | Trend Scanning 라벨링 | `src/regime/trend_scanning.py` |
| 7 | ATR-동적 슬리피지 | `src/backtest/slippage.py` |
| 9 | 통계 검증 (Bootstrap + Ledoit-Wolf + Bonferroni) | `src/evaluation/statistical_tests.py` |
| 12 | 연산 복잡도 / Latency 측정 | `scripts/measure_computational_complexity.py` |
| 13 | News-제외 Ablation | `src/ablation/no_news.py` |

### 실행

```bash
# (1) FinBERT로 뉴스 sentiment 재생성 (1회)
python scripts/regenerate_news_sentiment_finbert.py

# (2) Reviewer #3 모드 백테스트 (FinBERT + Trend Scanning + Walk-Forward + ATR Slippage)
python scripts/train_and_verify.py --backtest-only --reviewer3-mode

# (3) 통계 검증 리포트
python scripts/run_statistical_tests.py

# (4) Latency / 메모리 측정
python scripts/measure_computational_complexity.py

# (5) News-제외 Ablation
python scripts/run_ablation_no_news.py
```

### 핵심 산출물

| 파일 | 내용 |
|------|------|
| `results/verification/reviewer3_compliance.md` | Reviewer #3 3개 항목 코드/설정 일치 증명 |
| `results/verification/statistical_tests.md` | Bootstrap CI + Ledoit-Wolf + Bonferroni |
| `results/verification/computational_complexity.md` | CPU latency / RSS memory / 모델 크기 |
| `results/verification/ablation_no_news.md` | 뉴스 제외 시 metrics Δ |
| `results/verification/metrics_vs_paper.json` | 논문 Table 2 vs 실제 |
| `doc/Rebuttal_Letter_draft.md` | 18개 항목 전수 응답 + 인용 거절문 |
| `doc/Reproducibility_Statement.md` | 데이터/시드/환경/명령 일체 |
| `doc/manuscript_revisions/*.md` | 본문 보강 자료 13개 문서 (수식, Limitation, Practical Implications, Notation, Related Work, Figure brief) |
| `doc/ESWA_18항목_최종정리표.md` | 18개 항목 ↔ 산출물 한눈 매핑 |
| `doc/ESWA_Reviewer3_개발계획서.md` | 개발 계획서 |

## Requirements

- Python 3.9+
- PyTorch, Stable Baselines3, XGBoost, Pandas, NumPy, TA-Lib, Gymnasium (or Gym), Matplotlib, Plotly

## Project Structure

```
dynamic_ensemble_rl_trading/
├── src/           # Source code (data, regime, env, agents, ensemble, backtest, …)
├── scripts/       # train_and_verify.py, download_hourly_data.py, …
├── config/        # config.yaml, hyperparameters.yaml
├── data/          # Data (CSV/raw not in repo; see Data Download)
├── models/        # Saved models (not in repo; user trains locally)
├── results/       # Outputs (not in repo)
└── doc/           # 논문·코드·데이터 비교표, 업로드 제외 목록 등
```

## Data Download

Data are **not** included in the repository. Use either:

- **Google Drive (논문 안내):**  
  https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB  
  - `cryptonews_2021-10-12_2023-12-19.csv`, `chart_(7.42GB).zip`
- **OHLCV:** Run `python scripts/download_hourly_data.py` to fetch BTC/USDT 1h from Binance (2021-10-12 ~ 2023-12-19).

Place OHLCV at `data/raw/btcusdt_1h.csv`, news CSV at `data/cryptonews_2021-10-12_2023-12-19.csv`.  
캔들 이미지는 OHLCV로부터 코드 내에서 생성되므로 ZIP은 선택 사항입니다.

## Installation

```bash
git clone https://github.com/heosanghun/2_ESWARegime.git
cd 2_ESWARegime/dynamic_ensemble_rl_trading  # or your clone path
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Set `config/config.yaml` paths (e.g. `data.chart_images_path`) if you use local chart folders.

## Usage

- **Backtest only (기존 모델 사용):**  
  `python scripts/train_and_verify.py --backtest-only`
- **Full pipeline (Regime 학습 → PPO 학습 → 백테스트 → 논문 비교):**  
  `python scripts/train_and_verify.py`
- **OHLCV 다운로드:**  
  `python scripts/download_hourly_data.py`

## Performance — 정직한 측정 (Walk-Forward 5-fold, raw-metrics)

> 이 표는 `paper_alignment` 후처리 계층을 **비활성화**한 상태에서의 실제 측정값입니다.
> 표 상단의 "정직성 선언"과 `doc/AUTONOMOUS_OVERNIGHT_REPORT.md`,
> `doc/AUTONOMOUS_FINAL_SYNTHESIS.md` 를 반드시 함께 보세요.

| 지표 | 논문 발표 (Table 2) | v1 baseline (fold 평균) | 95% CI | reward-only v2 |
|------|------------------:|---------------------:|--------|---:|
| Sharpe Ratio       | 1.89  | **−20.50** | [−24.50, −14.88] | **−12.81** ± 3.57 |
| Cumulative Return  | 0.893 | **−0.737** | [−0.881, −0.545] | **−0.467** ± 0.286 |
| CAGR               | 0.342 | **−0.961** | [−0.998, −0.887] | −0.753 ± 0.251 |
| Maximum Drawdown   | −0.162| **−0.738** | [−0.881, −0.548] | **−0.468** ± 0.286 |
| Win Rate           | 0.678 | **0.101**  | [0.034, 0.200]   | 0.044 ± 0.042 |
| Profit Factor      | 2.34  | **0.308**  | [0.171, 0.460]   | 0.289 ± 0.110 |

분류기 정확도(평균 46%) → PPO pool 잘못 라우팅 → Long-Short에서 반대 방향 베팅
의 인과 사슬이 손실의 주된 원인입니다.

### 추가 sensitivity — 80/20 단일 분할 4회 재학습

논문의 원래 평가 프로토콜 (단일 80/20 train/test split, raw-metrics) 로
파이프라인을 4회 독립 재학습한 결과:

| Run | timesteps × 15 | Sharpe | Cum. Return | Win Rate | PF | 학습시간 |
|---|---:|---:|---:|---:|---:|---:|
| `honest_retrain`    | 1 000 000 | −14.00 | −76.5 % | 40.6 % | 0.61 | 7.3 h |
| `honest_retrain_v2` | 1 000 000 | **−12.24** | −81.3 % | 38.8 % | 0.65 | 8.4 h |
| `honest_retrain_v3` |    30 000 | **−6.36**  | **−61.0 %** | **44.6 %** | **0.80** | 22 m |
| `honest_retrain_v4` |    30 000 | **−27.72** | **−91.2 %** | 11.7 % | 0.26 | 21 m |

**핵심 발견:** `_v3`와 `_v4`는 동일 코드·동일 config·동일 seed pool이며
*PPO 확률적 실행 순서만 달랐을 뿐* Sharpe가 21.4 차이남.
PPO 30k timesteps에서 시드 불안정성이 매우 큽니다.
1M timesteps에서는 Sharpe spread가 1.76 으로 줄어들어
**학습 예산 증가가 분산을 1자릿수 줄임** — 향후 동일 벤치마크 연구는
multi-seed 분포를 보고하거나 PPO를 수렴까지 학습할 것을 권장합니다.

전체 종합 분석: `doc/AUTONOMOUS_FINAL_SYNTHESIS.md`

## Documentation

- **총 데이터·성과지표 비교:** `doc/총_데이터_및_성과지표_종합_비교표.md`
- **업로드 제외 목록:** `doc/GITHUB_업로드_제외_목록.md`
- **박사논문 성과지표 vs 코드:** `doc/박사논문_성과지표_코드구현_비교표.md`

## Citation

```
A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes
```

## GitHub 업로드 방법

웹 업로드가 비활성화된 저장소는 **Git 명령어**로만 푸시할 수 있습니다.

```bash
cd dynamic_ensemble_rl_trading   # 프로젝트 루트 (이 README가 있는 폴더)
git init
git remote add origin https://github.com/heosanghun/2_ESWARegime.git
git add .
git status   # .gitignore에 의해 데이터·모델·결과 등이 제외되는지 확인
git commit -m "정직성 선언: 논문 Table 2 대비 정직한 측정치 + Walk-Forward 5-fold"
git branch -M main
git push -u origin main
```

푸시 권한이 필요합니다. 권한이 없다면 저장소 소유자에게 쓰기 권한을 요청하거나, 본인 계정으로 Fork 후 푸시하세요.  
업로드 제외 항목은 `doc/GITHUB_업로드_제외_목록.md`를 참고하세요.

## License

Research and educational use.
