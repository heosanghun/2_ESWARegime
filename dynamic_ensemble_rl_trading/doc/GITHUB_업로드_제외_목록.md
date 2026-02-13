# GitHub 업로드 제외 목록

**저장소:** https://github.com/heosanghun/2_ESWARegime  
**목적:** 업로드 대상이 아닌 항목을 미리 검토·점검하여 **절대 업로드하지 않도록** `.gitignore`로 관리합니다.

---

## 제외 대상 (업로드하지 않음)

| 구분 | 경로/패턴 | 제외 사유 |
|------|-----------|-----------|
| **데이터** | `data/raw/*.csv`, `data/*.csv`, `data/processed/` | 용량 큼, 논문 Google Drive로 배포, 사용자가 다운로드·생성 |
| **학습된 모델** | `models/checkpoints/`, `models/ppo_agents/`, `models/regime_classifier/*.json` | 용량 큼, 사용자가 직접 학습 |
| **결과/로그** | `results/backtest/*.pkl`, `results/logs/*.log`, `results/verification/*.log`, `results/verification/*.json`, `results/plots/*.png` | 재실행으로 생성 가능 |
| **설정 백업** | `config/config_backup_iter_*.yaml` | 반복 실험용 로컬 백업, 공개 불필요 |
| **Python** | `__pycache__/`, `*.pyc`, `venv/`, `.venv/` | 실행 시 자동 생성 |
| **환경/IDE** | `.env`, `.idea/`, `.vscode/` | 로컬 환경·비밀 정보 |
| **기타** | `*.zip`, `*.pkl`, `*.log` | 대용량 또는 임시 파일 |

---

## 업로드 대상 (저장소에 포함)

- `src/` — 소스 코드 전부  
- `scripts/` — 실행 스크립트 (학습·백테스트·검증)  
- `config/config.yaml` — 메인 설정 (로컬 절대경로는 placeholder 사용)  
- `config/hyperparameters.yaml`, `config/paths.yaml` — 설정  
- `doc/` — 문서 (비교표, 검증 리포트 등)  
- `tests/` — 테스트 코드  
- `README.md`, `requirements.txt` — 루트 문서 및 의존성  

---

## 사용자 안내

- 데이터는 논문에서 안내한 [Google Drive](https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB)에서 다운로드하거나, `scripts/download_hourly_data.py`로 OHLCV를 수집합니다.  
- 학습된 모델은 `scripts/train_and_verify.py`로 직접 학습한 뒤 `models/`에 저장됩니다.
