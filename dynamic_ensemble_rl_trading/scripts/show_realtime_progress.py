"""
Real-time progress display (100% consistency autonomous self-verification).

Reads progress_reach_100.json and logs, refreshing progress output periodically.
"""

import json
import time
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROGRESS_FILE = BASE / "results" / "verification" / "progress_reach_100.json"
LOG_FILE = BASE / "results" / "verification" / "reach_100_autonomous.log"
REFRESH_SEC = 5
MAX_IDLE_SEC = 600  # Exit if no update for 10 minutes


def load_progress():
    if not PROGRESS_FILE.exists():
        return None
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def last_new_best_from_log():
    if not LOG_FILE.exists():
        return None
    try:
        text = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
        for line in reversed(text.strip().split("\n")):
            if "NEW BEST" in line:
                return line.strip()
        return None
    except Exception:
        return None


def progress_from_log():
    """Extract Try N from log when progress_reach_100.json is missing."""
    if not LOG_FILE.exists():
        return None
    try:
        text = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
        lines = text.strip().split("\n")
        current = 0
        best_avg = 0.0
        for line in lines:
            if "Try " in line:
                import re
                m = re.search(r"Try (\d+) blend=([\d.]+) scale=([\d.]+)", line)
                if m:
                    current = int(m.group(1))
            if "NEW BEST avg=" in line:
                m = re.search(r"NEW BEST avg=([\d.]+)%", line)
                if m:
                    best_avg = float(m.group(1))
        if current > 0:
            return {
                "current_try": current,
                "max_rounds": 20,
                "pct": round(100.0 * current / 20, 1),
                "best_avg_consistency": best_avg,
                "status": "running",
                "from_log": True,
            }
        return None
    except Exception:
        return None


def main():
    print("Real-time progress: 100% consistency autonomous run")
    print("Reading: " + str(PROGRESS_FILE))
    print("Refresh every " + str(REFRESH_SEC) + " sec. Ctrl+C to stop.")
    print("-" * 60)
    last_updated = 0
    last_status = None
    while True:
        p = load_progress() or progress_from_log()
        now = time.time()
        if p:
            last_updated = now
            status = p.get("status", "?")
            current = p.get("current_try", 0)
            total = p.get("max_rounds", 20)
            pct = p.get("pct", 0)
            best = p.get("best_avg_consistency", 0)
            blend = p.get("current_blend", 0)
            scale = p.get("current_scale", 0)
            phase = p.get("phase", "?")
            updated = p.get("updated", "")

            bar_len = 40
            filled = int(bar_len * current / total) if total else 0
            bar = "#" * filled + "-" * (bar_len - filled)

            line1 = "[%s] %s/%s (%.1f%%)" % (bar, current, total, pct)
            line2 = "Best consistency: %.1f%%  |  blend=%.2f scale=%.2f  |  %s" % (best, blend, scale, phase)
            line3 = "Updated: %s  |  status: %s" % (updated, status)

            sys.stdout.write("\r" + " " * 80 + "\r")
            print(line1)
            print(line2)
            print(line3)
            last_status = status

            new_best = last_new_best_from_log()
            if new_best:
                print("Last NEW BEST: " + new_best[:70])
            print("-" * 60)

            if status == "done":
                print("Run finished.")
                break
        else:
            if last_updated and (now - last_updated) > MAX_IDLE_SEC:
                print("No progress file or no update for 10 min. Exiting.")
                break
            print("\rWaiting for progress file... " + str(int(now) % 10) + "  ", end="", flush=True)

        time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
