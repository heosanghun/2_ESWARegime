# ESWA Reviewer #3 대응 — 개발 계획서

**논문:** Market Regime-aware Trading via Dynamic Ensemble Reinforcement Learning (ESWA-D-26-08980)  
**작성일:** 2026-05-13  
**작성자:** 허상훈 / **지도교수:** 황영배  
**마감:** 2026-06-03 (총 21일, P1 Must)

---

## 1. 배경

Reviewer #3는 본 연구의 **방법론적 무결성**을 위협하는 세 가지 치명적 결함을 지적했다.  
사전 코드 검토 결과 해당 지적 사항은 **전부 미반영(또는 단일 분할 형태로만 부분 반영)** 상태이다.

| # | 지적 | 현재 상태 |
|---|------|-----------|
| 1 | Look-ahead Bias — DeepSeek(2025년 출시) 기반 감성 라벨 | ❌ 미반영 (sentiment CSV가 사후 LLM 산출물) |
| 2 | 시계열 CV 오류 — 표준 K-fold/단일 분할 | ⚠️ 부분 반영 (Walk-Forward 함수는 단일 분할 1회만) |
| 3 | 후행적 라벨링 — SMA-50 (lagging indicator) | ❌ 미반영 (`ground_truth.py`가 SMA-50 그대로) |

---

## 2. 목표

본 개발 계획의 산출물은 다음을 만족해야 한다.

1. **재현 가능한** Out-of-Sample 실험 (Look-ahead Bias 완전 제거).
2. **시계열 무결성**을 지키는 다중 fold CV 기반 하이퍼파라미터 탐색.
3. **선행 horizon 기반** ground truth — 실시간 예측 가능 라벨.
4. 기존 파이프라인과의 **하위 호환** — `config.yaml` 옵션으로 신/구 방식 전환 가능.
5. 결과는 `results/verification/`에 신규 리포트로 저장되어 **Rebuttal Letter**에 인용 가능.

---

## 3. 작업 항목 (8개 작업)

### W1. FinBERT 기반 감성 분석 모듈 신규 구현
- **파일:** `src/data/finbert_sentiment.py`
- **모델:** `ProsusAI/finbert` (2019.08 공개 — 2021~2023 백테스트 시점 이전)
- **출력 컬럼:** `sentiment_class`, `sentiment_polarity`, `sentiment_subjectivity`
- **변환 로직:**  
  - FinBERT logits → softmax → (negative, neutral, positive) 확률  
  - `polarity = p(positive) - p(negative)` ∈ [-1, +1]  
  - `subjectivity = 1 - p(neutral)`  
- **Fallback:** transformers 미설치 시 기존 CSV polarity 그대로 사용 + 경고 로그.

### W2. FinBERT 뉴스 재생성 스크립트
- **파일:** `scripts/regenerate_news_sentiment_finbert.py`
- **입력:** `data/cryptonews_2021-10-12_2023-12-19.csv`
- **출력:** `data/cryptonews_finbert_2021-10-12_2023-12-19.csv`
- **로그:** 변환 건수, 처리 시간, 분포 비교(원본 vs FinBERT).

### W3. Trend Scanning 라벨링 알고리즘 신규 구현
- **파일:** `src/regime/trend_scanning.py`
- **알고리즘:** López de Prado (2018) Trend Scanning  
  - 각 시점 t에 대해 horizon L ∈ [L_min, L_max] 범위에서 미래 가격에 OLS 적합  
  - 정규화 기울기의 t-value 가장 큰 horizon 선택  
  - **라벨:** t-value > +threshold → Bull(2), < -threshold → Bear(0), else Sideways(1)
- **하위 호환:** `RegimeGroundTruth`에 `method='sma'`(기존) 또는 `'trend_scanning'`(신규) 분기.
- **선행성 보장:** 라벨 시점 t는 t-1 정보로만 학습/예측 가능하도록 lookahead 제거.

### W4. Walk-Forward Expanding Window CV / Purged K-fold 신규 구현
- **파일:** `src/validation/walk_forward_cv.py`
- **클래스:**  
  - `WalkForwardExpandingCV(n_splits, min_train_size, test_size, gap)`  
  - `PurgedKFold(n_splits, embargo)`
- **API:** scikit-learn 호환 `split(X)` → `(train_idx, test_idx)` 제너레이터.
- **하이퍼파라미터 튜닝 루프:** `tune_regime_classifier()` — 5 fold 평균 F1로 best params 선택.

### W5. 설정 파일 옵션 추가
- **파일:** `config/config.yaml`
- **신규 키:**
  ```yaml
  features:
    sentiment:
      model: finbert            # finbert | csv (기존)
      device: cpu               # cpu | cuda
  regime:
    label_method: trend_scanning  # sma | trend_scanning
    trend_scanning:
      horizon_min: 5
      horizon_max: 20
      t_threshold: 1.5
  validation:
    cv_method: walk_forward    # walk_forward | purged_kfold | single
    n_splits: 5
    embargo_hours: 24
  ```

### W6. 파이프라인 통합
- **파일 수정:** `scripts/train_and_verify.py`
- **CLI 옵션:** `--reviewer3-mode` 플래그 추가 시 위의 신규 모듈 전부 사용.
- **출력 리포트:** `results/verification/reviewer3_compliance.md`
  - FinBERT 적용 여부 + 라벨링 방법 + CV 방법 명시
  - 신/구 방법론 성능 비교표

### W7. 검증 & 문서화
- **파일:** `doc/Reviewer3_대응_결과리포트.md`
- **포함:**  
  - Before/After 성능 비교 (Sharpe, Cum Return, CAGR, Win Rate, MDD)  
  - CV fold별 분산  
  - 라벨링 방법별 ground truth 분포 변화

### W8. README & Git 반영
- README.md: Reviewer #3 대응 섹션 추가.
- `git add`/`commit`/`push`로 `https://github.com/heosanghun/2_ESWARegime` 반영.

---

## 4. 산출 파일 목록

| 구분 | 경로 | 비고 |
|------|------|------|
| 신규 | `src/data/finbert_sentiment.py` | W1 |
| 신규 | `scripts/regenerate_news_sentiment_finbert.py` | W2 |
| 신규 | `src/regime/trend_scanning.py` | W3 |
| 신규 | `src/validation/__init__.py` | W4 |
| 신규 | `src/validation/walk_forward_cv.py` | W4 |
| 수정 | `src/regime/ground_truth.py` | W3 (method 분기) |
| 수정 | `src/data/news_sentiment.py` | W1 (model 분기) |
| 수정 | `config/config.yaml` | W5 |
| 수정 | `scripts/train_and_verify.py` | W6 |
| 신규 | `doc/Reviewer3_대응_결과리포트.md` | W7 |
| 수정 | `README.md` | W8 |

---

## 5. 일정

| 단계 | 작업 | 소요 |
|------|------|------|
| Day 1 | W1·W3·W4 (모듈 코드 작성) — **본 작업** | 즉시 |
| Day 1 | W5·W6 (설정/통합) — **본 작업** | 즉시 |
| Day 2 | W2 (FinBERT 뉴스 재생성) — 사용자 환경에서 GPU/CPU로 실행 | 1일 |
| Day 3 | 백테스트 재실행 + W7 결과 리포트 | 1일 |
| Day 3 | W8 README & Git Push | 1시간 |

---

## 6. 위험·제약

- **transformers 패키지 미설치 시:** FinBERT 로딩 불가 → fallback으로 기존 CSV polarity 사용 + 경고.
- **GPU 미사용:** CPU에서 31,014건 추론 시 약 1~2시간 예상.
- **Trend Scanning 파라미터 (horizon, threshold):** 민감도 분석 필요 — 후속 작업으로 sweep 가능.

---

## 7. 검증 기준

- [ ] `python -c "from src.data.finbert_sentiment import FinBERTSentimentExtractor"` 무오류
- [ ] `python -c "from src.regime.trend_scanning import TrendScanningLabeler"` 무오류
- [ ] `python -c "from src.validation.walk_forward_cv import WalkForwardExpandingCV, PurgedKFold"` 무오류
- [ ] `scripts/train_and_verify.py --reviewer3-mode --backtest-only` 정상 종료
- [ ] `results/verification/reviewer3_compliance.md` 생성
