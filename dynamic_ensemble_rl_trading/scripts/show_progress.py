"""
Display training/verification pipeline progress.

Parses train_and_verify.log or terminal logs and outputs progress.
"""

import re
from pathlib import Path
from datetime import datetime

LOG_PATH = Path(__file__).parent.parent / "results" / "verification" / "train_and_verify.log"
PROGRESS_PATH = Path(__file__).parent.parent / "results" / "verification" / "progress.md"

# Total work per step
STEP1_TASKS = 1          # Regime Classifier
STEP2_POOLS = 3          # Bull, Bear, Sideways
STEP2_AGENTS_PER_POOL = 5
STEP2_TOTAL = STEP2_POOLS * STEP2_AGENTS_PER_POOL  # 15
STEP3_TASKS = 1          # Backtest
STEP4_TASKS = 1          # Compare

TOTAL_AGENTS = 15
TIMESTEP_PER_AGENT = 1_000_000


def parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {"step": 0, "phase": "idle", "detail": "log none", "agents_done": 0}

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.strip().split("\n")

    # Start of current run = last line containing "1,000,000 timesteps" or "1000000 timesteps each"
    start_idx = 0
    for i, line in enumerate(lines):
        if "1,000,000 timesteps" in line or "1000000 timesteps each" in line:
            start_idx = i

    step = 0
    phase = "unknown"
    detail = ""
    agents_done = 0
    current_agent = ""

    for i, line in enumerate(lines):
        if "STEP 1:" in line or "Regime Classifier" in line:
            step = 1
            phase = "Regime Classifier"
        if "Regime classifier training completed" in line or "Regime Classifier saved" in line:
            step = 2
            phase = "PPO Agents"
        if "Training Bull pool" in line:
            detail = "Bull pool"
        if "Training Bear pool" in line:
            detail = "Bear pool"
        if "Training Sideways pool" in line:
            detail = "Sideways pool"
        if "Training agent " in line and i >= start_idx:
            m = re.search(r"Training agent (\d+)/5", line)
            if m:
                current_agent = f"{detail} Agent {m.group(1)}/5"
        if "PPO agent training completed" in line and i >= start_idx:
            agents_done += 1
        if "STEP 3:" in line or "STEP 3" in line:
            step = 3
            phase = "Backtest"
            detail = ""
        if "Backtest completed" in line:
            step = 4
            phase = "Paper comparison"
        if "avg_consistency" in line or "pipeline" in line.lower() and "complete" in line.lower():
            step = 5
            phase = "Done"

    if step == 2 and current_agent and agents_done < TOTAL_AGENTS:
        detail = current_agent

    return {
        "step": step,
        "phase": phase,
        "detail": detail or (f"Agent {agents_done + 1}/{TOTAL_AGENTS}" if step == 2 else ""),
        "agents_done": min(agents_done, TOTAL_AGENTS),
        "total_agents": TOTAL_AGENTS,
    }


def format_progress(info: dict) -> str:
    s = info
    step = s["step"]
    agents_done = s["agents_done"]
    total = s["total_agents"]

    lines = [
        "# Training/Verification Pipeline Progress",
        "",
        f"**Updated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall steps",
        "| Step | Description | Status |",
        "|------|-------------|--------|",
    ]

    total = s["total_agents"]
    steps_desc = [
        (1, "Regime Classifier training", step >= 1),
        (2, "PPO Agents training (15)", step >= 2 and agents_done >= total),
        (3, "Test period backtest", step >= 3),
        (4, "Paper Table 2 comparison", step >= 4),
        (5, "Complete", step >= 5),
    ]
    for n, desc, done in steps_desc:
        status = "Complete" if done else ("In progress" if step == n else "Pending")
        lines.append(f"| {n} | {desc} | {status} |")

    lines.extend([
        "",
        "## STEP 2 detail (PPO Agents)",
        "",
    ])

    if step >= 2:
        pct = (agents_done / total * 100) if total else 0
        bar_len = 30
        filled = int(bar_len * agents_done / total) if total else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        lines.append(f"- **Progress:** {agents_done}/{total} agents ({pct:.1f}%)")
        lines.append(f"- `[{bar}]`")
        lines.append(f"- **Current:** {s['phase']} — {s['detail']}")
        lines.append("")
        # Estimated remaining time (~32 min per agent)
        if agents_done < total:
            remaining = total - agents_done
            est_min = remaining * 32
            lines.append(f"- Estimated time remaining: ~{est_min} min ({est_min/60:.1f} hours)")
    else:
        lines.append("- Pending")

    lines.append("")
    lines.append("---")
    lines.append("*This file is updated when `scripts/show_progress.py` is run.*")
    return "\n".join(lines)


def main():
    info = parse_log(LOG_PATH)
    report = format_progress(info)
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(report, encoding="utf-8")
    # Console: ASCII-only summary to avoid Windows cp949
    pct = (info["agents_done"] / info["total_agents"] * 100) if info["total_agents"] else 0
    print(f"Step {info['step']} | Phase: {info['phase']} | Agents: {info['agents_done']}/{info['total_agents']} ({pct:.1f}%)")
    print(f"Progress file: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
