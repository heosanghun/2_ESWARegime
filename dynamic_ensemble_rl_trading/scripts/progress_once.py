"""
진행률 1회 출력 (실시간 갱신 없음).
reach_100 자율 실행 또는 train_and_verify 로그/진행 파일 기준.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROGRESS_FILE = BASE / "results" / "verification" / "progress_reach_100.json"
LOG_100 = BASE / "results" / "verification" / "reach_100_autonomous.log"
METRICS_JSON = BASE / "results" / "verification" / "metrics_vs_paper.json"


def main():
    # 1) progress_reach_100.json
    if PROGRESS_FILE.exists():
        try:
            p = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            cur = p.get("current_try", 0)
            total = p.get("max_rounds", 20)
            pct = p.get("pct", 0)
            best = p.get("best_avg_consistency", 0)
            status = p.get("status", "?")
            print("=== Reach 100% autonomous progress ===")
            print("Try: %s / %s  (%.1f%%)" % (cur, total, pct))
            print("Best avg consistency: %.1f%%" % best)
            print("Status: %s" % status)
            print("File: %s" % PROGRESS_FILE)
            return
        except Exception:
            pass

    # 2) reach_100 로그 파싱
    if LOG_100.exists():
        try:
            text = LOG_100.read_text(encoding="utf-8", errors="ignore")
            lines = text.strip().split("\n")
            current = 0
            best_avg = 0.0
            best_blend = best_scale = None
            for line in lines:
                m = re.search(r"Try (\d+) blend=([\d.]+) scale=([\d.]+)", line)
                if m:
                    current = int(m.group(1))
                m = re.search(r"NEW BEST avg=([\d.]+)% blend=([\d.]+) scale=([\d.]+)", line)
                if m:
                    best_avg = float(m.group(1))
                    best_blend = m.group(2)
                    best_scale = m.group(3)
            if "Rounds done" in text or "Best avg consistency" in text:
                status = "done"
            else:
                status = "running"
            total = 20
            pct = round(100.0 * current / total, 1) if total else 0
            bar_len = 30
            filled = int(bar_len * current / total) if total else 0
            bar = "#" * filled + "-" * (bar_len - filled)
            print("=== Reach 100% autonomous progress (from log) ===")
            print("[%s] %s/%s  (%.1f%%)" % (bar, current, total, pct))
            print("Best avg consistency: %.1f%%  (blend=%s scale=%s)" % (best_avg, best_blend or "?", best_scale or "?"))
            print("Status: %s" % status)
            print("Log: %s" % LOG_100)
            return
        except Exception:
            pass

    # 3) metrics_vs_paper.json
    if METRICS_JSON.exists():
        try:
            d = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
            avg = d.get("avg_consistency", 0)
            print("=== Latest verification result ===")
            print("Avg consistency vs paper: %.1f%%" % avg)
            print("File: %s" % METRICS_JSON)
            return
        except Exception:
            pass

    print("No progress or result file found.")
    print("Run: python scripts/reach_100_percent_autonomous.py")
    print("Or:  python scripts/show_realtime_progress.py  (real-time)")


if __name__ == "__main__":
    main()
