# 오늘 밤 자율 작업 1페이지 요약 (사용자 복귀용)

_작성: 2026-05-14 22:20 KST,  자율 작업 종료 예정: 2026-05-15 10:00 KST_

## 한 줄 요약

> Reviewer #3 의 모든 방법론 요구사항(FinBERT, Trend Scanning,
> Walk-Forward CV, ATR slippage 등)을 정직하게 적용한 결과, **현재 코드/
> 방법론으로는 논문 Table 2 의 수치를 재현할 수 없음**을 통계적으로 확인
> 했습니다. 원인은 (a) **후행 라벨(SMA-50)** 이 자기참조 정보 누설로
> 분류기 정확도를 인위적으로 올리고 있고, (b) **`paper_alignment`** 후
> 처리 계층이 측정치를 다시 쓰고 있던 두 가지 결합 때문입니다.

## 핵심 수치 (정직 측정, Walk-Forward 5-fold, raw)

| 지표 | 논문 | 정직 측정 | 95% CI |
|------|---:|---:|---|
| Sharpe Ratio | 1.89 | **−20.50** | [−23.98, −16.12] |
| Cumulative Return | 0.893 | **−0.737** | [−0.863, −0.586] |
| Win Rate | 0.678 | **0.101** | [0.045, 0.174] |

모든 지표가 Bonferroni 보정 후에도 통계적으로 유의한 차이.
**평균적으로 Buy & Hold 대비 −85.6 pp 손실.**

## 결정적 진단 (Fold 1, 같은 테스트 구간)

| 라벨 | 액션 | 분류기 정확도 | Cum Ret | vs B&H |
|---|---|---:|---:|---:|
| Trend Scanning | Long-Short | ~47 % | −84.9 % | −41 pp |
| SMA-50 | Long-Short | ~90 % | −52.7 % | −9 pp |
| SMA-50 | Long-Only | ~90 % | −76.8 % | −33 pp |
| (B&H) | — | — | −43.9 % | 0 pp |

**SMA 라벨로 분류기를 ~90% 끌어올려도 B&H 를 못 이깁니다.**
→ PPO 보상함수가 시간당 단위에서 의도한 regime-specialised 거동을
   학습시키지 못한다는 추가 증거.

## 5-fold 종합 비교 (Trend Scanning vs SMA-50)

| 라벨 방식 | Cum Ret 평균 | Sharpe 평균 | 일치도 평균 |
|---|---:|---:|---:|
| Trend Scanning (Reviewer #3 요구) | **−73.7 %** | **−20.50** | 4.7 % |
| **SMA-50 (논문 원본 방식)** | **−52.3 %** | **−13.07** | 10.1 % |
| 논문 Table 2 발표값 | +89.3 % | +1.89 | — |
| Buy & Hold | +12.0 % | — | — |

**논문의 원본 SMA 방식으로도 정직 측정시 Sharpe −13.07** (논문 +1.89 와의
격차 = 14.96). 즉, `paper_alignment` 후처리 외에는 논문의 +1.89 수치를
설명할 수 있는 수학적 메커니즘이 없습니다.

## 가장 중요한 산출물 3 개

1. **`doc/AUTONOMOUS_OVERNIGHT_REPORT.md`** — 전체 분석 보고서
2. **`doc/Rebuttal_Letter_draft.md`** (Honest Addendum 추가됨)
3. **`results/walk_forward/`** — 5-fold 모든 raw 결과 + 통계검증 + Table 1

## 결정 대기 사항 (10시 복귀 후)

1. **공개 옵션 A**: 정직한 수치로 논문을 수정하여 재제출.
   - Table 2 ⇒ Walk-Forward fold 평균 + 95% CI 로 교체
   - "Limitations" 섹션 신설
2. **공개 옵션 B**: 분류기 백본을 시퀀스 모델로 교체 후 재학습 시도
   (현재 분류기는 forward label 에서 학습 자체가 거의 안 됩니다).
3. **공개 옵션 C**: 논문 철회 + 재설계 후 재제출.

## 제가 *하지 않은* 것

- `paper_alignment` 재활성화 (액션 반전, B&H 혼합, 스케일링 등)
  → 측정치 위조에 해당. 절대 작동시키지 않음.
- Git push (사용자 확인 전 외부 상태 변경 금지).
- 라벨 정의 자체를 SMA 로 되돌리는 행위 (Reviewer #3 의 핵심 지적을
  무시하는 것이 됨).

## 즉시 가능한 빠른 액션

```powershell
# 1) 전체 정직 보고서 열기
notepad doc/AUTONOMOUS_OVERNIGHT_REPORT.md

# 2) Walk-Forward 5-fold 결과 보기
notepad results/walk_forward/summary.md
notepad results/walk_forward/diagnostics.md
notepad results/walk_forward/statistical_tests.md
notepad results/walk_forward/table1_classifier_per_fold.md
notepad results/walk_forward/clf_feature_ablation.md

# 3) Reviewer #3 Compliance (FinBERT, Trend Scanning, Walk-Forward 모두 활성)
notepad results/verification/reviewer3_compliance.md
```
