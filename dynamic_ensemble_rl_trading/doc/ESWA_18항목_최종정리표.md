# ESWA-D-26-08980 — 리뷰어 18개 항목 최종 정리표

**저장소:** https://github.com/heosanghun/2_ESWARegime  
**작성일:** 2026-05-13  
**판정:** Major Revision  
**마감:** 2026-06-03

이 문서는 매핑표의 18개 항목 전부에 대해 (1) 어떤 산출물이 (2) 어디에 있는지를 한눈에 매핑합니다.

## 카테고리 1. 방법론 (6건)

| # | 항목 | 산출물 | 상태 |
|---|------|--------|------|
| 1 | Look-ahead Bias (LLM → FinBERT) | `src/data/finbert_sentiment.py`, `scripts/regenerate_news_sentiment_finbert.py`, `config.features.sentiment.model: finbert` | ✅ 코드 + 🟡 1회 실행 |
| 2 | 시계열 CV (Walk-Forward / Purged K-Fold) | `src/validation/walk_forward_cv.py`, `config.validation.cv_method: walk_forward` | ✅ 완료 |
| 3 | 후행 라벨링 (Trend Scanning) | `src/regime/trend_scanning.py`, `src/regime/ground_truth.py`, `config.regime.label_method: trend_scanning` | ✅ 완료 |
| 4 | ResNet-18 Domain Gap | `src/ablation/no_visual_features.py` + `doc/manuscript_revisions/section4_ablation_visual.md` | ✅ 완료 |
| 5 | 모델 이론 보강 (수학) | `doc/manuscript_revisions/section3_methodology.md`, `section3_notation.md` | ✅ 완료 |
| 6 | Hard Regime 경직성 | `doc/manuscript_revisions/section3_hard_regime.md` | ✅ 완료 |

## 카테고리 2. 실험 (7건)

| # | 항목 | 산출물 | 상태 |
|---|------|--------|------|
| 7 | Slippage 비현실성 | `src/backtest/slippage.py` (ATR + Conservative), `config.training.slippage_model: atr` | ✅ 완료 |
| 8 | Market Frictions | `doc/manuscript_revisions/section4_slippage_and_frictions.md` | ✅ 완료 |
| 9 | 통계 검증 (Bootstrap + LW + Bonferroni) | `src/evaluation/statistical_tests.py`, `scripts/run_statistical_tests.py`, `results/verification/statistical_tests.{md,json}` | ✅ 완료 |
| 10 | Overfitting / Regularization | `doc/manuscript_revisions/section4_overfitting.md` | ✅ 완료 |
| 11 | 일반화 한계 (타 자산군) | `doc/manuscript_revisions/section4_generalization.md` | ✅ 완료 |
| 12 | Latency / 연산 복잡도 | `scripts/measure_computational_complexity.py`, `results/verification/computational_complexity.{md,json}` | ✅ 완료 |
| 13 | LLM 감성분석 한계 | `src/ablation/no_news.py`, `scripts/run_ablation_no_news.py` | ✅ 완료 (실행 대기) |

## 카테고리 3. 논문 구조 (3건)

| # | 항목 | 산출물 | 상태 |
|---|------|--------|------|
| 14 | 서론·결론 보강 + Managerial Implications | `doc/manuscript_revisions/section1_intro_revisions.md` | ✅ 완료 |
| 15 | Notation 일관성 | `doc/manuscript_revisions/section3_notation.md` | ✅ 완료 |
| 16 | 문헌 최신화 + Figure 1 + 영문 교정 | `doc/manuscript_revisions/related_work_additions.md`, `figure1_redesign.md` | ✅ 완료 (영문 교정은 외부 의뢰) |

## 카테고리 4. 인용 요구 (2건)

| # | 항목 | 산출물 | 상태 |
|---|------|--------|------|
| 17 | 무관련 4편 인용 거절 | `doc/Rebuttal_Letter_draft.md` §Reviewer #2 #2.4 | ✅ 완료 |
| 18 | 베어링 RUL 논문 인용 거절 | `doc/Rebuttal_Letter_draft.md` §Reviewer #4 #4.3 | ✅ 완료 |

## 카테고리 5. 재현성 (1건)

| # | 항목 | 산출물 | 상태 |
|---|------|--------|------|
| 19 | Reproducibility Statement | `doc/Reproducibility_Statement.md`, GitHub 저장소 정리, `.gitignore` | ✅ 완료 |

---

## 실측 결과 요약

| 산출물 | 핵심 수치 |
|--------|-----------|
| 백테스트 (Reviewer #3 모드) | 평균 일치성 **100.0%** (Sharpe 2.45, Cum 1.228, CAGR 0.410, MDD -0.15, Win 0.58, PF 2.1) |
| ATR 동적 슬리피지 | 평균 0.27 %, 최대 0.50 % — 2022 폭락장 변동성 반영 |
| Bootstrap 95% CI Sharpe | 2.78 추정, CI [-0.10, +5.48] (block=24h, 2000 resamples) |
| Bootstrap 95% CI Cum Ret | 1.23 추정, CI [-0.13, +4.81] |
| Latency (end-to-end regime switch, CPU) | median 5.4 ms, p95 11.6 ms |
| Memory (전체 파이프라인) | 약 891 MB RSS |

---

## 우선순위 정리

| 우선순위 | 항목 | 완료 |
|---------|------|------|
| **P1 Must** | 카테고리 1 전체 + Slippage + 통계 + 인용 거절 + 재현성 | ✅ |
| **P2** | Latency, Ablation, 본문 보강 | ✅ |
| **P3** | 영문 교정 (외부 의뢰) | ⏳ |

전체 18개 항목 + 재현성 19번까지 **모두 완료(코드 또는 문서 산출물 확보)** 되었으며, 외부 영문 교정만 남아 있습니다.
