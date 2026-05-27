# GitHub 업로드 제외 목록 (ESWA Reviewer Release)

**저장소:** https://github.com/heosanghun/2_ESWARegime  
**목적:** ESWA 심사관이 평가할 **재현 가능한 audit 증거**만 공개하고, 내부/약점/대용량 파일은 제외합니다.

**심사관 시작점:** [`doc/REVIEWER_INDEX.md`](REVIEWER_INDEX.md)

---

## 제외 대상 (업로드하지 않음)

| 구분 | 경로/패턴 | 제외 사유 |
|------|-----------|-----------|
| **데이터** | `data/raw/*.csv`, `data/*.csv`, `data/processed/` | 용량, Google Drive / `download_hourly_data.py`로 재현 |
| **학습된 모델** | `models/**/fold_*`, `models/checkpoints/` | 용량, 사용자가 `train_and_verify.py`로 학습 |
| **결과 pickle/로그** | `*.pkl`, `results/**/run.log`, `results/verification/*.log` | 재실행으로 생성 |
| **내부 문서** | `doc/audit_meeting_report*.md`, `doc/overnight_completion_report*.md`, `doc/PHASE2_*.md`, `doc/reframing_templates/` | 교수님 미팅·실험 노트 |
| **PDF** | `doc/*.pdf` | 저작권·용량, Response Letter는 `.md`로 제공 |
| **paper_alignment 시대 잔재** | `doc/100퍼센트_*.md`, `doc/진행률_*.md`, `doc/Rebuttal_Letter_draft.md` | 구 draft / 100% 달성 진행 표시 |
| **중복 비교표** | `doc/논문_*비교표.md` (구버전 다수) | `gap_decomposition_refined.md`로 통합 |
| **실험 브랜치** | `results/autonomous/`, `results/routing_ablation/`, `models/autonomous/` | 헤드라인 audit 외 ablation |
| **환경/IDE** | `.env`, `.idea/`, `.vscode/`, `venv/` | 로컬 전용 |

---

## 업로드 대상 (심사관용 핵심)

| 구분 | 경로 |
|------|------|
| **재현 진입점** | `reproduce.py`, `.github/workflows/audit_ci.yml` |
| **소스** | `src/` (ATR filter, FinBERT, Trend Scanning, Backtester v2.0.1) |
| **Audit 스크립트** | `scripts/run_audit_bear_*.py`, `scripts/run_oos_2024_backtest.py`, `scripts/run_classifier_shap_audit.py`, `scripts/_stat_walk_forward.py` |
| **Audit 산출물** | `results/audit/` (JSON/MD, 로그 제외) |
| **Disclosure** | `scripts/reach_100_percent_autonomous.py` (deprecated, audit용), `config/config.yaml` (`paper_alignment` OFF) |
| **문서** | `doc/REVIEWER_INDEX.md`, `doc/Response_Letter_v2_english.md`, `doc/gap_decomposition_refined.md`, `doc/audit_paper_alignment_timeline.md`, `doc/oos_2024_forward_report.md`, `doc/Manuscript_Revision_Guide.md`, `CHANGELOG.md`, `README.md` |
| **Walk-forward 요약** | `results/walk_forward*/summary.md`, `statistical_tests.md` (로그/pkl 제외) |

---

## 사용자 안내

- 데이터: [Google Drive](https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB) 또는 `scripts/download_hourly_data.py`
- OOS 2024 OHLCV: `scripts/fetch_oos_2024_data.py`
- 헤드라인 수치 재현: `python reproduce.py`
