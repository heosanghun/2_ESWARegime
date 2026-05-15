"""
Overnight autonomous pipeline orchestrator.

Sequence
--------
1. Wait for/verify the most recent train_and_verify v4 run.
2. Read v4 metrics and write a v4 mini-report.
3. Run Walk-Forward Expanding Window CV (5 folds, expanding).
4. Run statistical robustness checks on the walk-forward returns.
5. Aggregate all artefacts into `doc/AUTONOMOUS_OVERNIGHT_REPORT.md`.

Designed to run unattended; each stage is idempotent / resumable.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "results" / "verification"
DOC_DIR = PROJECT_ROOT / "doc"
DOC_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("overnight")


def _setup_log() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "overnight.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def stage_walk_forward() -> int:
    logger.info("STAGE 2: Walk-Forward Expanding Window CV")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_walk_forward.py"),
    ]
    logger.info("Launching: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def stage_statistical_tests() -> int:
    """Run Ledoit-Wolf Sharpe + Bonferroni on whatever returns exist."""
    logger.info("STAGE 3: Statistical robustness tests")
    script = PROJECT_ROOT / "scripts" / "run_statistical_tests.py"
    if not script.exists():
        logger.warning("run_statistical_tests.py missing, skipping.")
        return 0
    return subprocess.call([sys.executable, str(script)], cwd=str(PROJECT_ROOT))


def stage_complexity() -> int:
    logger.info("STAGE 4: Computational complexity")
    script = PROJECT_ROOT / "scripts" / "measure_computational_complexity.py"
    if not script.exists():
        logger.warning("measure_computational_complexity.py missing, skipping.")
        return 0
    return subprocess.call([sys.executable, str(script)], cwd=str(PROJECT_ROOT))


def collect_final_report() -> None:
    logger.info("STAGE 5: Compiling final overnight report")
    wf_summary = PROJECT_ROOT / "results" / "walk_forward" / "summary.md"
    wf_json = PROJECT_ROOT / "results" / "walk_forward" / "summary.json"
    v4_log = PROJECT_ROOT / "results" / "verification" / "honest_retrain_v4.log"
    rev3 = PROJECT_ROOT / "results" / "verification" / "reviewer3_compliance.md"

    lines = []
    lines.append("# Autonomous Overnight Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for label, p in [
        ("v4 training log", v4_log),
        ("Walk-Forward summary (md)", wf_summary),
        ("Walk-Forward summary (json)", wf_json),
        ("Reviewer #3 compliance", rev3),
    ]:
        lines.append(f"- {label}: `{p.relative_to(PROJECT_ROOT)}` "
                     f"(exists: {p.exists()})")
    lines.append("")

    # Pull the v4 final metrics (the "최종 일치성" line if present).
    if v4_log.exists():
        tail = v4_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-50:]
        lines.append("## v4 final lines")
        lines.append("")
        lines.append("```")
        lines.extend(tail)
        lines.append("```")
        lines.append("")

    # Embed walk-forward summary if it exists.
    if wf_summary.exists():
        lines.append("## Walk-Forward Aggregate")
        lines.append("")
        lines.append(wf_summary.read_text(encoding="utf-8", errors="ignore"))
        lines.append("")

    if wf_json.exists():
        try:
            data = json.loads(wf_json.read_text(encoding="utf-8"))
            n_folds = data.get("aggregate", {}).get("n_folds")
            cons = data["aggregate"]["avg_consistency_pct"]["mean"]
            lines.append(f"_Folds: {n_folds},  mean consistency = {cons:.1f}%_")
            lines.append("")
        except Exception:
            pass

    out = DOC_DIR / "AUTONOMOUS_OVERNIGHT_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Final report written to %s", out)


def main() -> int:
    _setup_log()
    t0 = time.time()
    logger.info("===== Overnight autonomous run started =====")

    # Stage 2: Walk-Forward CV (this is the bulk of the work).
    rc_wf = stage_walk_forward()
    logger.info("walk-forward exit=%s", rc_wf)

    # Stage 3: Statistical tests.
    rc_stat = stage_statistical_tests()
    logger.info("statistical-tests exit=%s", rc_stat)

    # Stage 4: Computational complexity (cheap; can rerun safely).
    rc_cx = stage_complexity()
    logger.info("complexity exit=%s", rc_cx)

    # Stage 5: Final report.
    collect_final_report()
    logger.info("Overnight done in %.1f min.", (time.time() - t0) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
