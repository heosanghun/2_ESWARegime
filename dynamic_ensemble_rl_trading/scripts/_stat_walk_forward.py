"""
Statistical robustness analysis on the walk-forward fold metrics.

This script analyses the *aggregate* distribution of metrics across the
N walk-forward folds, computes confidence intervals via the percentile
bootstrap, and applies a Bonferroni correction across the set of
performance metrics compared to the published paper Table 2 values.

Audit extension (2026-05-26):
  - --subdir         : choose any results/<subdir>/ that contains
                       fold_*/fold_summary.json (default: walk_forward).
  - --out-dir        : write outputs under results/<out-dir>/
                       (default: same as --subdir).
  - --output-prefix  : filename prefix (e.g. "s1_" produces
                       s1_statistical_tests.{md,json}).
  - --label          : human-readable config label for the report header.

Examples
--------

    # Legacy (v1 baseline, original location):
    python scripts/_stat_walk_forward.py

    # Audit P1.1-A — S1 soft routing + EMA12:
    python scripts/_stat_walk_forward.py \\
        --subdir routing_ablation/phase2_soft/S1_soft_ema12 \\
        --out-dir audit \\
        --output-prefix s1_ \\
        --label "S1 soft routing + EMA12 (primary honest)"

    # Audit P1.1-B — ATR 1.8% sideways filter screen:
    python scripts/_stat_walk_forward.py \\
        --subdir autonomous/screen_extra_18 \\
        --out-dir audit \\
        --output-prefix atr_screen_ \\
        --label "ATR 1.8% sideways filter screen (no retrain)"
"""
from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subdir",
        default="walk_forward",
        help="Subdir under results/ containing fold_*/fold_summary.json (default: walk_forward).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output subdir under results/ (default: same as --subdir).",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Filename prefix; '<prefix>statistical_tests.{md,json}'. Empty = legacy name.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable label for the config (e.g. 'S1' or 'ATR 1.8%%') for the report header.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 10000).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    target_dir = PROJECT_ROOT / "results" / args.subdir
    fold_files = sorted(target_dir.glob("fold_*/fold_summary.json"))
    if not fold_files:
        logger.error("No fold_summary.json found in %s", target_dir)
        return 1
    folds: List[Dict] = [json.loads(fp.read_text(encoding="utf-8")) for fp in fold_files]

    metric_keys = list(PAPER_METRICS.keys())
    n_folds = len(folds)
    K = len(metric_keys)
    alpha = 0.05
    alpha_bonf = alpha / K

    label = args.label or args.subdir

    out_lines = [
        f"# Walk-Forward Statistical Robustness Tests — {label}",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"**Source:** `results/{args.subdir}/`    |    **N folds:** {n_folds}    "
        f"|    **Family-wise α:** {alpha}    |    **Bonferroni α:** {alpha_bonf:.4f}    "
        f"|    **Bootstrap resamples:** {args.n_boot:,}",
        "",
        "## Per-metric paired comparison vs paper Table 2",
        "",
        "Each fold's metric is treated as one observation. We report the "
        "mean across folds, the percentile bootstrap 95% CI, and a "
        "Bonferroni-adjusted CI; the test compares the fold mean to the "
        "paper's published value (PAPER_METRICS in train_and_verify.py).",
        "",
        "| Metric | Paper value | Fold mean | 95% CI | Bonferroni 95% CI | Paper inside CI? |",
        "|--------|------------:|----------:|--------|--------------------|------------------|",
    ]
    summary_dict: Dict[str, Dict] = {}
    for m in metric_keys:
        vals = np.array([f["metrics"].get(m, np.nan) for f in folds], dtype=float)
        if np.isnan(vals).all():
            continue
        bp = percentile_bootstrap(vals, n_boot=args.n_boot)
        bp_bonf = percentile_bootstrap(vals, n_boot=args.n_boot, alpha=alpha_bonf)
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

    out_lines.extend([
        "",
        "## Interpretation",
        "",
        "- A 'YES' in the last column means the paper-reported value falls "
        "inside the bootstrap 95% confidence interval estimated from the "
        f"{n_folds} walk-forward folds. A 'no' means the difference between "
        "paper and walk-forward estimate is statistically significant.",
        "- 'Bonf:YES' applies the family-wise Bonferroni correction across "
        f"{K} metrics (α/{K}); this is the multiple-comparison-safe test "
        "recommended by Reviewer #1.4 (Ledoit-Wolf + Bonferroni).",
    ])

    out_dir = PROJECT_ROOT / "results" / (args.out_dir or args.subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or ""
    md_name = f"{prefix}statistical_tests.md"
    json_name = f"{prefix}statistical_tests.json"
    out_md = out_dir / md_name
    out_json = out_dir / json_name
    out_md.write_text("\n".join(out_lines), encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "label": label,
                "source_subdir": args.subdir,
                "generated": datetime.now().isoformat(timespec="seconds"),
                "n_folds": n_folds,
                "n_boot": args.n_boot,
                "alpha": alpha,
                "alpha_bonferroni": alpha_bonf,
                "metrics": summary_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", out_md, out_json)
    # Console echo with cp949-safe fallback (Windows default codepage).
    try:
        print("\n".join(out_lines))
    except UnicodeEncodeError:
        import sys as _sys
        safe = "\n".join(out_lines).encode("ascii", errors="replace").decode("ascii")
        _sys.stdout.write(safe + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
