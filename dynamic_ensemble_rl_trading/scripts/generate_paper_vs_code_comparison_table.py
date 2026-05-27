"""
Generate paper vs codebase performance metrics comparison table.

Computes per-metric consistency (percent) and outputs tables and files.
"""

import json
from pathlib import Path
from datetime import datetime

# Paper [final_260211] (4).pdf Table 2 - Proposed Method
PAPER_METRICS = {
    "Sharpe Ratio": 2.45,
    "Cumulative Return": 1.23,
    "CAGR": 0.41,
    "Maximum Drawdown": -0.15,
    "Win Rate": 0.58,
    "Profit Factor": 2.1,
}

# Consistency: 100% when actual == paper; decay based on relative difference vs scale
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
            "metric": name,
            "paper_value": paper_val,
            "codebase_value": actual_val,
            "difference": diff,
            "consistency_pct": pct,
        })

    # Markdown table
    md_lines = [
        "# Paper vs Codebase Performance Metrics Comparison",
        "",
        "Source: Paper [final_260211] A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes (4).pdf, Table 2 (Proposed Method) vs current codebase backtest results.",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Metric | Paper value | Codebase value | Difference | Consistency (%) |",
        "|--------|-------------|----------------|------------|-----------------|",
    ]
    for r in rows:
        paper_s = str(r["paper_value"])
        code_s = f"{r['codebase_value']:.4f}" if r["codebase_value"] is not None else "N/A"
        diff_s = f"{r['difference']:.4f}" if r["difference"] is not None else "N/A"
        pct_s = f"{r['consistency_pct']:.1f}%"
        md_lines.append(f"| {r['metric']} | {paper_s} | {code_s} | {diff_s} | {pct_s} |")

    avg_pct = sum(r["consistency_pct"] for r in rows) / len(rows) if rows else 0
    md_lines.extend([
        "",
        "## Summary",
        "",
        f"- **Per-metric consistency**: see table above.",
        f"- **Average consistency**: **{avg_pct:.1f}%** (arithmetic mean of 6 metrics).",
        "",
        "Note: consistency (%) is closer to 100% when the relative difference from the paper value is small. Values may differ if the test period or data do not match the paper.",
        "",
    ])

    md_path = out_dir / "paper_vs_code_comparison_table.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {md_path}")

    doc_md = doc_dir / "paper_vs_codebase_performance_comparison_table.md"
    doc_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {doc_md}")

    # CSV
    csv_path = out_dir / "paper_vs_code_comparison_table.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("metric,paper_value,codebase_value,difference,consistency_pct\n")
        for r in rows:
            code_s = "" if r["codebase_value"] is None else str(r["codebase_value"])
            diff_s = "" if r["difference"] is None else str(r["difference"])
            f.write(f"{r['metric']},{r['paper_value']},{code_s},{diff_s},{r['consistency_pct']:.1f}\n")
    print(f"Saved: {csv_path}")

    print("\nConsistency summary:")
    for r in rows:
        print(f"  {r['metric']}: {r['consistency_pct']:.1f}%")
    print(f"  Average: {avg_pct:.1f}%")


if __name__ == "__main__":
    main()
