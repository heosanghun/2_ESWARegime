"""
Check in-progress/completed status and paper comparison results in one run.
Run this script alone to see status and results without manual checks.
"""

import json
import re
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
LOG_PATH = BASE / "results" / "verification" / "train_and_verify.log"
METRICS_PATH = BASE / "results" / "verification" / "metrics_vs_paper.json"
OUT_PATH = BASE / "results" / "verification" / "status_and_results.txt"


def get_training_status():
    """Parse log: in progress (1M run) vs complete (full pipeline) vs not started."""
    if not LOG_PATH.exists():
        return "Not started", 0, 15, ""

    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    lines = text.strip().split("\n")

    # 1M run start line
    start_idx = 0
    for i, line in enumerate(lines):
        if "1,000,000 timesteps" in line or "1000000 timesteps each" in line:
            start_idx = i

    agents_done = 0
    for i, line in enumerate(lines):
        if i >= start_idx and "PPO agent training completed" in line:
            agents_done += 1

    if "Backtest+compare done" in text or "Full pipeline complete" in text:
        # Distinguish backtest-only vs full pipeline completion
        if "Backtest-only mode" in text and "Backtest+compare done" in text:
            return "Complete (backtest only)", 15, 15, ""
        return "Complete (full pipeline)", 15, 15, ""

    if agents_done >= 15:
        return "In progress (backtest phase)", 15, 15, ""

    pool = "Bull"
    detail = ""
    for i in range(start_idx, len(lines)):
        if "Training Bear pool" in lines[i]:
            pool = "Bear"
        if "Training Sideways pool" in lines[i]:
            pool = "Sideways"
        if "Training agent " in lines[i]:
            m = re.search(r"Training agent (\d+)/5", lines[i])
            if m:
                detail = "{} {}/5".format(pool, m.group(1))
    return "In progress (training)", agents_done, 15, detail


def get_results():
    """Read metrics_vs_paper.json."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    status, done, total, detail = get_training_status()
    results = get_results()

    lines = []
    lines.append("=" * 60)
    lines.append("Status and results summary (updated: {})".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("=" * 60)
    lines.append("")
    lines.append("[Progress status]")
    lines.append("  Status: {}".format(status))
    if status.startswith("In progress") and total:
        lines.append("  PPO agents: {}/{}".format(done, total))
        if detail:
            lines.append("  Current: {}".format(detail))
    lines.append("")
    lines.append("[Results vs paper]")
    if results:
        act = results.get("actual_metrics", {})
        paper = results.get("paper_metrics", {})
        cons = results.get("consistency", {})
        avg = results.get("avg_consistency", 0)
        lines.append("  Average consistency: {:.1f}%".format(avg))
        lines.append("")
        lines.append("  Metric            Paper(target)  Actual       Consistency(%)")
        lines.append("  " + "-" * 50)
        for k in ["Sharpe Ratio", "Cumulative Return", "CAGR", "Maximum Drawdown", "Win Rate", "Profit Factor"]:
            p = paper.get(k, 0)
            a = act.get(k, 0)
            c = cons.get(k, 0)
            if isinstance(a, float) and abs(a) < 1e-6 and k != "Win Rate":
                a_str = "{:.4f}".format(a)
            elif isinstance(a, float):
                a_str = "{:.2f}".format(a) if abs(a) <= 10 else "{:.4f}".format(a)
            else:
                a_str = str(a)
            lines.append("  {:18s} {:>10}   {:>10}   {:>6.1f}%".format(k[:18], str(p), a_str, c))
        lines.append("")
        lines.append("  Goal: 100% match on all metrics (paper performance)")
    else:
        lines.append("  No results yet (generated after backtest run)")
    lines.append("")
    lines.append("=" * 60)

    out_text = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out_text, encoding="utf-8")

    # ASCII summary on console (avoid encoding issues)
    print("Status:", status)
    if results:
        print("Avg consistency:", "{:.1f}%".format(results.get("avg_consistency", 0)))
        print("Full report:", OUT_PATH)
    else:
        print("No metrics yet. Run backtest or full pipeline.")
    print("Report file:", OUT_PATH)


if __name__ == "__main__":
    main()
