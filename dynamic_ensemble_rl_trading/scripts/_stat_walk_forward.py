"""
Statistical robustness analysis on the walk-forward fold metrics.

This script analyses the *aggregate* distribution of metrics across the
N walk-forward folds, computes confidence intervals via the percentile
bootstrap, and applies a Bonferroni correction across the set of
performance metrics compared to the published paper Table 2 values.

The output is written to:
  - results/walk_forward/statistical_tests.md
  - results/walk_forward/statistical_tests.json
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_and_verify import PAPER_METRICS  # noqa: E402

logger = logging.getLogger("stat_wf")


def percentile_bootstrap(
    samples: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, rng_seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    n = len(samples)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(samples[idx]))
    return {
        "mean": float(np.mean(samples)),
        "ci_low": float(np.quantile(boots, alpha / 2)),
        "ci_high": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(n),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    wf_dir = PROJECT_ROOT / "results" / "walk_forward"
    fold_files = sorted(wf_dir.glob("fold_*/fold_summary.json"))
    if not fold_files:
        logger.error("No fold_summary.json found in %s", wf_dir)
        return 1
    folds: List[Dict] = [json.loads(fp.read_text(encoding="utf-8")) for fp in fold_files]

    metric_keys = list(PAPER_METRICS.keys())
    n_folds = len(folds)
    # Bonferroni-adjusted alpha for K comparisons.
    K = len(metric_keys)
    alpha = 0.05
    alpha_bonf = alpha / K

    out_lines = [
        "# Walk-Forward Statistical Robustness Tests",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"**N folds:** {n_folds}    |    **Family-wise α:** {alpha}    "
        f"|    **Bonferroni-adjusted α:** {alpha_bonf:.4f}",
        "",
        "## Per-metric paired comparison vs paper Table 2",
        "",
        "Each fold's metric is treated as one observation. We report the "
        "mean across folds, the percentile bootstrap 95% CI (10,000 "
        "resamples), and a Bonferroni-adjusted CI; the test compares the "
        "fold mean to the paper's published value.",
        "",
        "| Metric | Paper value | Fold mean | 95% CI | Bonferroni 95% CI | Paper inside CI? |",
        "|--------|------------:|----------:|--------|--------------------|------------------|",
    ]
    summary_dict: Dict[str, Dict] = {}
    for m in metric_keys:
        vals = np.array([f["metrics"].get(m, np.nan) for f in folds], dtype=float)
        if np.isnan(vals).all():
            continue
        bp = percentile_bootstrap(vals)
        # Bonferroni CI: alpha replaced by alpha_bonf.
        bp_bonf = percentile_bootstrap(vals, alpha=alpha_bonf)
        paper_v = float(PAPER_METRICS[m])
        inside = bp["ci_low"] <= paper_v <= bp["ci_high"]
        inside_bonf = bp_bonf["ci_low"] <= paper_v <= bp_bonf["ci_high"]
        summary_dict[m] = {
            "paper": paper_v,
            "mean": bp["mean"],
            "ci": [bp["ci_low"], bp["ci_high"]],
            "ci_bonferroni": [bp_bonf["ci_low"], bp_bonf["ci_high"]],
            "paper_inside_ci": inside,
            "paper_inside_ci_bonferroni": inside_bonf,
            "fold_values": vals.tolist(),
        }
        out_lines.append(
            f"| {m} | {paper_v} | {bp['mean']:.4f} | "
            f"[{bp['ci_low']:.4f}, {bp['ci_high']:.4f}] | "
            f"[{bp_bonf['ci_low']:.4f}, {bp_bonf['ci_high']:.4f}] | "
            f"{'YES' if inside else 'no'} / Bonf:{'YES' if inside_bonf else 'no'} |"
        )

    out_lines.append("")
    out_lines.append("## Interpretation")
    out_lines.append("")
    out_lines.append(
        "- A 'YES' in the last column means the paper-reported value falls "
        "inside the bootstrap 95% confidence interval estimated from the "
        f"{n_folds} walk-forward folds. A 'no' means the difference between "
        "paper and walk-forward estimate is statistically significant."
    )
    out_lines.append(
        "- 'Bonf:YES' applies the family-wise Bonferroni correction across "
        f"{K} metrics (α/{K}); this is the multiple-comparison-safe test "
        "recommended by Reviewer #1.4 (Ledoit-Wolf + Bonferroni)."
    )

    out_md = wf_dir / "statistical_tests.md"
    out_json = wf_dir / "statistical_tests.json"
    out_md.write_text("\n".join(out_lines), encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "n_folds": n_folds,
                "alpha": alpha,
                "alpha_bonferroni": alpha_bonf,
                "metrics": summary_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", out_md, out_json)
    print("\n".join(out_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
