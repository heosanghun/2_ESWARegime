"""
논문 제시 성과지표 vs 코드베이스 성과지표 비교표 생성.
각 항목별 일치성(퍼센트)을 계산하여 표와 파일로 출력.
"""

import json
from pathlib import Path
from datetime import datetime

# 논문 [최종완성본_260211] (4).pdf Table 2 - Proposed Method
PAPER_METRICS = {
    "Sharpe Ratio": 2.45,
    "Cumulative Return": 1.23,
    "CAGR": 0.41,
    "Maximum Drawdown": -0.15,
    "Win Rate": 0.58,
    "Profit Factor": 2.1,
}

# 일치성 계산: 100% when actual == paper; 감쇠는 scale 대비 차이로 계산
def consistency_percent(paper_val: float, actual_val: float) -> float:
    if actual_val is None:
        return 0.0
    scale = max(abs(paper_val), 0.01)
    rel_diff = min(1.0, abs(actual_val - paper_val) / scale)
    return round(100.0 * (1.0 - rel_diff), 1)


def main():
    base = Path(__file__).parent.parent
    json_path = base / "results" / "verification" / "metrics_vs_paper.json"
    out_dir = base / "results" / "verification"
    doc_dir = base.parent.parent / "doc"  # ESWARegime/doc
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"Not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actual = data.get("actual_metrics", {})
    rows = []
    for name, paper_val in PAPER_METRICS.items():
        actual_val = actual.get(name)
        pct = consistency_percent(paper_val, actual_val) if actual_val is not None else 0.0
        diff = (actual_val - paper_val) if actual_val is not None else None
        rows.append({
            "지표": name,
            "논문_제시": paper_val,
            "코드베이스": actual_val,
            "차이": diff,
            "일치성_퍼센트": pct,
        })

    # Markdown 테이블
    md_lines = [
        "# 논문 vs 코드베이스 성과지표 비교표",
        "",
        "출처: 논문 [최종완성본_260211] A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes (4).pdf, Table 2 (Proposed Method) vs 현재 코드베이스 백테스트 결과.",
        "",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 지표 | 논문 제시 값 | 코드베이스 값 | 차이 | 일치성 (%) |",
        "|------|-------------|---------------|------|------------|",
    ]
    for r in rows:
        paper_s = str(r["논문_제시"])
        code_s = f"{r['코드베이스']:.4f}" if r["코드베이스"] is not None else "N/A"
        diff_s = f"{r['차이']:.4f}" if r["차이"] is not None else "N/A"
        pct_s = f"{r['일치성_퍼센트']:.1f}%"
        md_lines.append(f"| {r['지표']} | {paper_s} | {code_s} | {diff_s} | {pct_s} |")

    avg_pct = sum(r["일치성_퍼센트"] for r in rows) / len(rows) if rows else 0
    md_lines.extend([
        "",
        "## 종합",
        "",
        f"- **항목별 일치성**: 위 표 참조.",
        f"- **평균 일치성**: **{avg_pct:.1f}%** (6개 지표 산술 평균).",
        "",
        "※ 일치성(%) = 논문 값과의 상대적 차이가 작을수록 100%에 가깝게 계산. 테스트 기간·데이터가 논문과 동일하지 않으면 수치가 달라질 수 있음.",
        "",
    ])

    md_path = out_dir / "paper_vs_code_comparison_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {md_path}")

    doc_md = doc_dir / "논문_코드베이스_성과지표_비교표.md"
    doc_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {doc_md}")

    # CSV
    csv_path = out_dir / "paper_vs_code_comparison_table.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("지표,논문 제시 값,코드베이스 값,차이,일치성(%)\n")
        for r in rows:
            code_s = "" if r["코드베이스"] is None else str(r["코드베이스"])
            diff_s = "" if r["차이"] is None else str(r["차이"])
            f.write(f"{r['지표']},{r['논문_제시']},{code_s},{diff_s},{r['일치성_퍼센트']:.1f}\n")
    print(f"Saved: {csv_path}")

    print("\n일치성 요약:")
    for r in rows:
        print(f"  {r['지표']}: {r['일치성_퍼센트']:.1f}%")
    print(f"  평균: {avg_pct:.1f}%")


if __name__ == "__main__":
    main()
