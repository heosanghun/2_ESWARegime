# GitHub Upload Exclusions (ESWA Reviewer Release)

**Repository:** https://github.com/heosanghun/2_ESWARegime  
**Purpose:** Publish only **reproducible audit evidence** for ESWA reviewers; exclude internal notes, large binaries, and experimental branches.

**Reviewer entry point:** [`doc/REVIEWER_INDEX.md`](REVIEWER_INDEX.md)

---

## Excluded (never upload)

| Category | Path / pattern | Reason |
|----------|----------------|--------|
| Data | `data/raw/*.csv`, `data/*.csv`, `data/processed/` | Large; reproduce via Google Drive or `download_hourly_data.py` |
| Trained models | `models/**/fold_*`, `models/checkpoints/` | Large; train via `train_and_verify.py` |
| Pickles / logs | `*.pkl`, `results/**/run.log` | Regenerated on rerun |
| Internal docs | `doc/audit_meeting_report*.md`, `doc/overnight_completion_report*.md`, `doc/PHASE2_*.md` | Advisor meeting / experiment notes |
| PDFs | `doc/*.pdf` | Copyright; use `.md` response letters |
| Legacy drafts | `doc/Rebuttal_Letter_draft.md`, `doc/100percent_*.md` | Superseded |
| Duplicate tables | legacy `doc/paper_*comparison*.md` | Consolidated in `gap_decomposition_refined.md` |
| Experimental branches | `results/autonomous/`, `models/autonomous/` | Not headline audit |
| Local env | `.env`, `.idea/`, `.vscode/`, `venv/` | Machine-specific |

---

## Included (reviewer-facing)

| Category | Path |
|----------|------|
| Reproduction | `reproduce.py`, `.github/workflows/audit_ci.yml` |
| Source | `src/` (ATR filter, FinBERT, Trend Scanning, Backtester v2.0.1) |
| Audit scripts | `scripts/run_audit_bear_*.py`, `scripts/run_oos_2024_backtest.py`, `scripts/run_classifier_shap_audit.py` |
| Audit results | `results/audit/` (JSON/MD, no logs) |
| Disclosure | `scripts/reach_100_percent_autonomous.py` (deprecated), `config/config.yaml` |
| Docs | `doc/REVIEWER_INDEX.md`, `doc/Response_Letter_v2_english.md`, `doc/gap_decomposition_refined.md`, `README.md` |

---

## User instructions

- Data: [Google Drive](https://drive.google.com/drive/folders/14UvhfTAUGlqbL27kbP-Bn86KgPZ9OxpB) or `scripts/download_hourly_data.py`
- OOS 2024: `scripts/fetch_oos_2024_data.py`
- Headline numbers: `python reproduce.py`
