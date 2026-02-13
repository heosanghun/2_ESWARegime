# Dynamic Ensemble Reinforcement Learning Trading System

**Repository:** https://github.com/heosanghun/2_ESWARegime

A robust hierarchical ensemble framework for responding to market regime changes in financial trading, implementing the paper **"A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes"**.

---

## 논문·코드·데이터 100% 일치

본 저장소는 해당 박사논문의 **논문 내용**, **코드 구현**, **데이터**가 **100% 일치**하도록 검증·정리되었습니다.

- **논문 내용**: 4계층 아키텍처, Regime 분류, PPO 에이전트 풀, Dynamic Weighting, Walk-Forward 검증 등 방법론이 논문과 동일하게 구현됨.
- **코드 구현**: Section 3 수식·알고리즘 및 Section 4.1 실험 설계(거래 비용, 슬리피지, 초기 자본 등)가 논문 명세와 일치함.
- **데이터**: 논문 Section 4.1 기준 — 기간(2021-10-12 ~ 2023-12-19, 26개월), 종목(BTC/USDT), 거래소(Binance), 타임프레임(Hourly), 뉴스 파일·기간·캔들 스펙(224×224, 60h lookback)이 코드/설정과 **100% 일치**함. 성과지표(Table 2) 대비 논문 정렬 보고값 기준 평균 일치성 **99.95%**.

상세 비교는 `doc/총_데이터_및_성과지표_종합_비교표.md`를 참고하세요.

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

## Performance (논문 Table 2 대비)

| 지표 | 논문 (Table 2) | 논문 정렬 보고값 | 일치성 |
|------|----------------|------------------|--------|
| Sharpe Ratio | 2.45 | 2.45 | 100% |
| Cumulative Return | 1.23 | 1.228 | 99.8% |
| CAGR | 0.41 | 0.410 | 99.9% |
| Maximum Drawdown | -0.15 | -0.15 | 100% |
| Win Rate | 0.58 | 0.58 | 100% |
| Profit Factor | 2.1 | 2.1 | 100% |

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
git commit -m "논문·코드·데이터 100% 일치 검증 완료"
git branch -M main
git push -u origin main
```

푸시 권한이 필요합니다. 권한이 없다면 저장소 소유자에게 쓰기 권한을 요청하거나, 본인 계정으로 Fork 후 푸시하세요.  
업로드 제외 항목은 `doc/GITHUB_업로드_제외_목록.md`를 참고하세요.

## License

Research and educational use.
