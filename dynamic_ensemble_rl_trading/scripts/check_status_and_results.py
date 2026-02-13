"""
진행중/완료 여부 + 논문 대비 결과값을 한 번에 확인.
수동 확인 없이 이 스크립트만 실행하면 상태와 결과를 파악할 수 있음.
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
    """로그 파싱: 진행중(1M run) vs 완료(전체 파이프라인 완료) vs 미실행."""
    if not LOG_PATH.exists():
        return "미실행", 0, 15, ""

    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    lines = text.strip().split("\n")

    # 1M run 시작 라인
    start_idx = 0
    for i, line in enumerate(lines):
        if "1,000,000 timesteps" in line or "1000000 timesteps each" in line:
            start_idx = i

    agents_done = 0
    for i, line in enumerate(lines):
        if i >= start_idx and "PPO agent training completed" in line:
            agents_done += 1

    if "Backtest+compare 완료" in text or "전체 파이프라인 완료" in text:
        # 마지막 완료 메시지가 backtest-only인지 full인지
        if "Backtest-only mode" in text and "Backtest+compare 완료" in text:
            return "완료(백테스트만 실행됨)", 15, 15, ""
        return "완료(전체 파이프라인)", 15, 15, ""

    if agents_done >= 15:
        return "진행중(백테스트 단계)", 15, 15, ""

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
    return "진행중(학습)", agents_done, 15, detail


def get_results():
    """metrics_vs_paper.json 읽기."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    status, done, total, detail = get_training_status()
    results = get_results()

    lines = []
    lines.append("=" * 60)
    lines.append("상태 및 결과 요약 (갱신: {})".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("=" * 60)
    lines.append("")
    lines.append("[진행 상태]")
    lines.append("  상태: {}".format(status))
    if "진행중" in status and total:
        lines.append("  PPO 에이전트: {}/{}".format(done, total))
        if detail:
            lines.append("  현재: {}".format(detail))
    lines.append("")
    lines.append("[논문 대비 결과]")
    if results:
        act = results.get("actual_metrics", {})
        paper = results.get("paper_metrics", {})
        cons = results.get("consistency", {})
        avg = results.get("avg_consistency", 0)
        lines.append("  평균 일치성: {:.1f}%".format(avg))
        lines.append("")
        lines.append("  지표            논문(목표)    실제값      일치성(%)")
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
        lines.append("  목표: 모든 지표 100% 일치 (논문 성과지표)")
    else:
        lines.append("  결과 없음 (백테스트 실행 후 생성됨)")
    lines.append("")
    lines.append("=" * 60)

    out_text = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out_text, encoding="utf-8")

    # 콘솔에는 ASCII 요약만 (인코딩 이슈 방지)
    print("Status:", status)
    if results:
        print("Avg consistency:", "{:.1f}%".format(results.get("avg_consistency", 0)))
        print("Full report:", OUT_PATH)
    else:
        print("No metrics yet. Run backtest or full pipeline.")
    print("Report file:", OUT_PATH)


if __name__ == "__main__":
    main()
