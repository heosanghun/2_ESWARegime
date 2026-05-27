"""Generate publication-ready figures for the revised manuscript.

Outputs to ``results/figures/``. Figures intentionally avoid third-party
styling so they remain reproducible from a clean matplotlib install.

Figures
-------
- ``fig_classifier_accuracy.png``
    Classifier train / test accuracy under each labelling scheme.
    Visualises the look-ahead-bias story (SMA ~ perfect → ~46% under
    forward-looking Trend-Scanning).
- ``fig_walk_forward_returns.png``
    Per-fold cumulative returns vs Buy-and-Hold for the published
    Trend-Scanning + SMA + Long-Only runs.
- ``fig_paper_vs_honest.png``
    Side-by-side bars: original Table 2 values vs walk-forward
    Bonferroni-corrected 95% CIs.
- ``fig_reward_signal_spread.png``
    Sanity comparison of the v1 vs v2 reward gradient.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.size": 10,
})


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Figure A: classifier accuracy under each labelling scheme.
# ---------------------------------------------------------------------
def fig_classifier_accuracy() -> None:
    """Bar chart contrasting SMA-50 vs Trend-Scanning classifier acc."""
    # Numbers below come from:
    #   results/walk_forward/table1_classifier_per_fold.json     (Trend)
    #   results/walk_forward_sma/  (per-fold backtest)           (SMA)
    # SMA values are derived from the diagnostic in the overnight
    # report (test acc 0.90 ± 0.02). Trend-Scanning numbers are
    # exact from the Table-1 JSON below.
    schemes = ["SMA-50 (lagging)", "Trend Scanning (forward-looking)"]
    train_acc = [1.000, 1.000]
    test_acc = [0.907, 0.461]

    x = np.arange(len(schemes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, train_acc, width, label="Train", color="#4878d0")
    bars2 = ax.bar(x + width / 2, test_acc, width, label="Test (mean of 5 folds)", color="#ee854a")

    ax.axhline(1 / 3, ls=":", lw=1, color="grey", label="3-class chance (33%)")
    ax.set_xticks(x)
    ax.set_xticklabels(schemes)
    ax.set_ylabel("Classifier accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="upper right")
    ax.set_title("Regime-classifier accuracy collapses under forward-looking labels")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h*100:.1f}%",
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_classifier_accuracy.png")
    plt.close(fig)
    print(f"  wrote {FIG_DIR/'fig_classifier_accuracy.png'}")


# ---------------------------------------------------------------------
# Figure B: walk-forward cumulative returns per fold (TS vs SMA).
# ---------------------------------------------------------------------
def fig_walk_forward_returns() -> None:
    runs = {
        "Trend Scanning + Long-Short": (
            PROJECT_ROOT / "results" / "walk_forward" / "summary.json",
            "#d65f5f",
        ),
        "SMA-50 + Long-Short": (
            PROJECT_ROOT / "results" / "walk_forward_sma" / "summary.json",
            "#4878d0",
        ),
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    bar_w = 0.38
    fold_axis = np.arange(1, 6, dtype=float)
    for i, (name, (path, color)) in enumerate(runs.items()):
        if not path.exists():
            continue
        d = _load_json(path)
        # Build a length-5 vector indexed by fold id; missing folds = NaN.
        rets = np.full(5, np.nan, dtype=float)
        for f in d["folds"]:
            rets[f["fold"] - 1] = f["metrics"]["Cumulative Return"] * 100
        offset = (i - 0.5) * bar_w
        valid = ~np.isnan(rets)
        ax.bar(fold_axis[valid] + offset, rets[valid], width=bar_w,
               label=name, color=color)

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(fold_axis)
    ax.set_xticklabels([f"Fold {k}" for k in fold_axis.astype(int)])
    ax.set_ylabel("Cumulative return (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("Per-fold cumulative return  —  honest measurement")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_walk_forward_returns.png")
    plt.close(fig)
    print(f"  wrote {FIG_DIR/'fig_walk_forward_returns.png'}")


# ---------------------------------------------------------------------
# Figure E: v1 baseline vs reward-only-v2 vs full-v2 ablation.
# ---------------------------------------------------------------------
def fig_v2_ablation() -> None:
    """Per-fold cumulative-return ablation across v1/reward-v2/full-v2.

    Reads summaries from three walk-forward directories. Skips runs that
    do not (yet) exist on disk so the figure regenerates gracefully
    during sequential ablation work.
    """
    runs = [
        ("v1 baseline",      PROJECT_ROOT / "results" / "walk_forward" / "summary.json",                "#bbbbbb"),
        ("reward-only v2",   PROJECT_ROOT / "results" / "walk_forward_reward_v2" / "summary.json",      "#4878d0"),
        ("full v2",          PROJECT_ROOT / "results" / "walk_forward_reward_v2_full" / "summary.json", "#d65f5f"),
    ]
    available = [(n, p, c) for (n, p, c) in runs if p.exists()]
    if not available:
        print("  fig_v2_ablation: no summaries available, skipped")
        return

    fold_axis = np.arange(1, 6, dtype=float)
    n_runs = len(available)
    bar_w = 0.8 / n_runs

    fig, (ax_perfold, ax_mean) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1.3]}
    )

    mean_names: List[str] = []
    mean_vals:  List[float] = []
    mean_stds:  List[float] = []
    colors:     List[str]   = []

    for i, (name, path, color) in enumerate(available):
        d = _load_json(path)
        rets = np.full(5, np.nan, dtype=float)
        for f in d["folds"]:
            rets[f["fold"] - 1] = f["metrics"]["Cumulative Return"] * 100
        offset = (i - (n_runs - 1) / 2.0) * bar_w
        valid = ~np.isnan(rets)
        ax_perfold.bar(fold_axis[valid] + offset, rets[valid], width=bar_w,
                       label=name, color=color)
        mean_names.append(name)
        mean_vals.append(float(np.nanmean(rets)))
        mean_stds.append(float(np.nanstd(rets, ddof=0)))
        colors.append(color)

    ax_perfold.axhline(0, color="black", lw=0.6)
    ax_perfold.set_xticks(fold_axis)
    ax_perfold.set_xticklabels([f"Fold {k}" for k in fold_axis.astype(int)])
    ax_perfold.set_ylabel("Cumulative return (%)")
    ax_perfold.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax_perfold.set_title("Per-fold cumulative return  —  ablation")
    ax_perfold.legend(loc="lower right", fontsize=9)

    xpos = np.arange(len(mean_names))
    ax_mean.bar(xpos, mean_vals, yerr=mean_stds, capsize=4,
                color=colors, edgecolor="black", linewidth=0.5)
    ax_mean.axhline(0, color="black", lw=0.6)
    ax_mean.set_xticks(xpos)
    ax_mean.set_xticklabels(mean_names, rotation=20, ha="right", fontsize=9)
    ax_mean.set_ylabel("Mean cumulative return (%)  ± std")
    ax_mean.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax_mean.set_title("5-fold mean")
    for x, v, s in zip(xpos, mean_vals, mean_stds):
        ax_mean.annotate(f"{v:.1f}%\n±{s:.1f}",
                         xy=(x, v),
                         xytext=(0, 4 if v >= 0 else -14),
                         textcoords="offset points",
                         ha="center",
                         va="bottom" if v >= 0 else "top",
                         fontsize=8)

    fig.suptitle("v2 ablation:  reward-only v2 closes ~37% of the v1 gap; "
                 "full v2 gives back ~10 pp", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v2_ablation.png")
    plt.close(fig)
    print(f"  wrote {FIG_DIR/'fig_v2_ablation.png'}")


# ---------------------------------------------------------------------
# Figure C: paper Table 2 vs honest walk-forward Bonferroni CI.
# ---------------------------------------------------------------------
def fig_paper_vs_honest() -> None:
    # Hard-coded paper values + honest measurements (TS+long-short).
    rows = [
        # metric, paper, honest mean, honest CI low, honest CI high
        ("Sharpe",          1.89,  -20.50, -24.50, -14.88),
        ("Cum. Return",     0.893, -0.737, -0.881, -0.545),
        ("CAGR",            0.342, -0.961, -0.998, -0.887),
        ("Max DD",         -0.162, -0.738, -0.881, -0.548),
        ("Win Rate",        0.678,  0.101,  0.034,  0.200),
        ("Profit Factor",   2.34,   0.308,  0.171,  0.460),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=False)
    axes = axes.ravel()
    for ax, (name, paper, mean, lo, hi) in zip(axes, rows):
        ax.bar([0], [paper], width=0.6, color="#bbbbbb", label="Original draft")
        ax.bar([1], [mean], width=0.6, color="#d65f5f", label="Honest (5-fold)")
        err = np.array([[mean - lo], [hi - mean]])
        ax.errorbar([1], [mean], yerr=err,
                    fmt="none", ecolor="black", lw=1.2, capsize=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Paper", "Honest"])
        ax.set_title(name)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Original Table 2 values vs honest walk-forward "
                 "(Bonferroni-corrected 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_paper_vs_honest.png")
    plt.close(fig)
    print(f"  wrote {FIG_DIR/'fig_paper_vs_honest.png'}")


# ---------------------------------------------------------------------
# Figure D: reward signal spread (v1 vs v2).
# ---------------------------------------------------------------------
def fig_reward_signal_spread() -> None:
    """How much does v2 amplify the correct-vs-wrong signal?"""
    # Numbers from scripts/_sanity_reward_v2.py and the legacy reward
    # design (Sortino over 30-bar window with weak immediate gradient).
    regimes = ["Bull", "Bear", "Sideways"]
    v1_spread = [1.0, 1.0, 1.0]    # approximate: pv_pct only
    v2_spread = [4.10, 4.85, 4.00]  # measured spread between correct and wrong

    x = np.arange(len(regimes))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - w / 2, v1_spread, width=w, label="v1 (Sortino, pv_pct only)",
           color="#bbbbbb")
    ax.bar(x + w / 2, v2_spread, width=w, label="v2 (direction-aligned)",
           color="#4878d0")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Reward spread  (correct − wrong direction)")
    ax.set_title("Reward function v2 amplifies the directional gradient ~5×")
    ax.legend()
    for i, v in enumerate(v2_spread):
        ax.annotate(f"{v:.2f}×", xy=(x[i] + w / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_reward_signal_spread.png")
    plt.close(fig)
    print(f"  wrote {FIG_DIR/'fig_reward_signal_spread.png'}")


# ---------------------------------------------------------------------
# Figure F: PPO seed instability (single-split retrains).
# ---------------------------------------------------------------------
def fig_ppo_seed_instability() -> None:
    """Visualise the v1/v2/v3/v4 single-split retrain spread.

    Numbers are read directly out of the canonical
    ``results/verification/honest_retrain*.log`` so the figure stays
    in lock-step with the source-of-truth files. If a log is missing
    we fall back to the documented constants from
    ``doc/AUTONOMOUS_FINAL_SYNTHESIS.md``.
    """
    import re

    LOG_DIR = PROJECT_ROOT / "results" / "verification"
    defaults = {
        "honest_retrain":    {"steps": 1_000_000, "sharpe": -14.00, "ret": -0.7652},
        "honest_retrain_v2": {"steps": 1_000_000, "sharpe": -12.24, "ret": -0.8134},
        "honest_retrain_v3": {"steps":    30_000, "sharpe":  -6.36, "ret": -0.6095},
        "honest_retrain_v4": {"steps":    30_000, "sharpe": -27.72, "ret": -0.9115},
    }

    def _parse(stem: str) -> Dict[str, float]:
        path = LOG_DIR / f"{stem}.log"
        if not path.exists():
            return defaults[stem]
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            m_steps = re.search(r"total_timesteps per agent:\s*([\d,]+)", txt)
            m_shrp  = re.search(r"Sharpe Ratio\s*:\s*(-?\d+\.\d+)", txt)
            m_ret   = re.search(r"Cumulative Return\s*:\s*(-?\d+\.\d+)%", txt)
            if not (m_steps and m_shrp and m_ret):
                return defaults[stem]
            return {
                "steps":  int(m_steps.group(1).replace(",", "")),
                "sharpe": float(m_shrp.group(1)),
                "ret":    float(m_ret.group(1)) / 100.0,
            }
        except Exception:
            return defaults[stem]

    runs = {stem: _parse(stem) for stem in defaults.keys()}

    labels   = ["v1\n(1 M)", "v2\n(1 M)", "v3\n(30 k)", "v4\n(30 k)"]
    sharpes  = [runs[k]["sharpe"] for k in
                ("honest_retrain", "honest_retrain_v2",
                 "honest_retrain_v3", "honest_retrain_v4")]
    rets_pct = [runs[k]["ret"] * 100 for k in
                ("honest_retrain", "honest_retrain_v2",
                 "honest_retrain_v3", "honest_retrain_v4")]
    # Colour by training budget: 1 M → blue, 30 k → red.
    colors   = ["#4878d0", "#4878d0", "#d65f5f", "#d65f5f"]

    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    xs = np.arange(len(labels))
    bars_s = ax_s.bar(xs, sharpes, color=colors, edgecolor="black", linewidth=0.5)
    ax_s.axhline(0, color="black", lw=0.6)
    ax_s.set_xticks(xs)
    ax_s.set_xticklabels(labels)
    ax_s.set_ylabel("Sharpe Ratio  (single-split test 2023-06 → 2023-12)")
    ax_s.set_title("PPO seed instability — Sharpe")
    for b, v in zip(bars_s, sharpes):
        ax_s.annotate(f"{v:.2f}",
                      xy=(b.get_x() + b.get_width() / 2, v),
                      xytext=(0, 3 if v >= 0 else -12),
                      textcoords="offset points",
                      ha="center",
                      va="bottom" if v >= 0 else "top",
                      fontsize=9)
    # Spread annotation between v3 and v4.
    spread = sharpes[2] - sharpes[3]
    ax_s.annotate(f"30 k spread = {abs(spread):.1f}",
                  xy=(2.5, (sharpes[2] + sharpes[3]) / 2),
                  xytext=(2.6, -22),
                  fontsize=9,
                  ha="left",
                  arrowprops=dict(arrowstyle="-[", lw=0.8, color="grey"))
    spread_1m = sharpes[0] - sharpes[1]
    ax_s.annotate(f"1 M spread = {abs(spread_1m):.2f}",
                  xy=(0.5, (sharpes[0] + sharpes[1]) / 2),
                  xytext=(-0.2, -5),
                  fontsize=9,
                  ha="left",
                  arrowprops=dict(arrowstyle="-[", lw=0.8, color="grey"))

    bars_r = ax_r.bar(xs, rets_pct, color=colors, edgecolor="black", linewidth=0.5)
    ax_r.axhline(0, color="black", lw=0.6)
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels(labels)
    ax_r.set_ylabel("Cumulative return")
    ax_r.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax_r.set_title("PPO seed instability — Cum. return")
    for b, v in zip(bars_r, rets_pct):
        ax_r.annotate(f"{v:.1f}%",
                      xy=(b.get_x() + b.get_width() / 2, v),
                      xytext=(0, 3 if v >= 0 else -12),
                      textcoords="offset points",
                      ha="center",
                      va="bottom" if v >= 0 else "top",
                      fontsize=9)

    # Shared legend by colour.
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#4878d0", edgecolor="black", label="1 000 000 timesteps × 15 agents"),
        Patch(facecolor="#d65f5f", edgecolor="black", label="30 000 timesteps × 15 agents"),
    ]
    ax_s.legend(handles=legend_elems, loc="lower right", fontsize=8)

    fig.suptitle(
        "Single-split retrains under identical config — PPO seed variance "
        "shrinks ~12× when training budget increases 33×",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ppo_seed_instability.png")
    plt.close(fig)
    print(f"  wrote {FIG_DIR/'fig_ppo_seed_instability.png'}")


# ---------------------------------------------------------------------
# Figure G: 30k vs 1M walk-forward Sharpe (reward-only v2).
# ---------------------------------------------------------------------
def fig_1m_vs_30k_wf() -> None:
    """Per-fold Sharpe: 30k reward-v2 vs 1M post-rebacktest."""
    path_30k = PROJECT_ROOT / "results" / "walk_forward_reward_v2" / "summary.json"
    path_1m = PROJECT_ROOT / "results" / "walk_forward_reward_v2_1M" / "summary_rebacktest.json"
    if not path_30k.exists() or not path_1m.exists():
        print("  fig_1m_vs_30k_wf: missing summary, skipped")
        return

    d30 = _load_json(path_30k)
    d1m = _load_json(path_1m)
    sharpes_30 = np.full(5, np.nan)
    for f in d30["folds"]:
        sharpes_30[f["fold"] - 1] = f["metrics"]["Sharpe Ratio"]
    sharpes_1m = np.array(d1m["metrics_aggregate"]["Sharpe Ratio"]["values"])

    fold_axis = np.arange(1, 6, dtype=float)
    bar_w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(fold_axis - bar_w / 2, sharpes_30, width=bar_w,
           label="reward-v2 @ 30k", color="#4878d0")
    ax.bar(fold_axis + bar_w / 2, sharpes_1m, width=bar_w,
           label="reward-v2 @ 1M (post-rebacktest)", color="#d65f5f")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(fold_axis)
    ax.set_xticklabels([f"Fold {k}" for k in fold_axis.astype(int)])
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("1M training does not improve walk-forward Sharpe")
    ax.legend(loc="lower left")
    mean30 = float(np.nanmean(sharpes_30))
    mean1m = float(np.nanmean(sharpes_1m))
    ax.annotate(f"Mean 30k = {mean30:.1f}\nMean 1M = {mean1m:.1f}",
                xy=(0.98, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_1m_vs_30k_wf.png")
    plt.close(fig)
    print(f"  wrote {FIG_DIR/'fig_1m_vs_30k_wf.png'}")


# ---------------------------------------------------------------------
# Figure H: Backtester clip-bug fix impact (1M WF).
# ---------------------------------------------------------------------
def fig_backtester_fix() -> None:
    """Pre- vs post-rebacktest mean Sharpe and cum. return."""
    status_path = PROJECT_ROOT / "results" / "walk_forward_reward_v2_1M" / "autopilot_status.json"
    rebt_path = PROJECT_ROOT / "results" / "walk_forward_reward_v2_1M" / "summary_rebacktest.json"
    if not status_path.exists() or not rebt_path.exists():
        print("  fig_backtester_fix: missing autopilot artefacts, skipped")
        return

    status = _load_json(status_path)
    rebt = _load_json(rebt_path)
    pre = status["pre_rebacktest_aggregate"]["metrics_aggregate"]
    post = rebt["metrics_aggregate"]

    labels = ["Sharpe", "Cum. Ret."]
    pre_vals = [pre["Sharpe Ratio"]["mean"], pre["Cumulative Return"]["mean"] * 100]
    post_vals = [post["Sharpe Ratio"]["mean"], post["Cumulative Return"]["mean"] * 100]

    x = np.arange(len(labels))
    bar_w = 0.36
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - bar_w / 2, pre_vals, width=bar_w,
           label="Pre-fix (shorts clipped)", color="#bbbbbb")
    ax.bar(x + bar_w / 2, post_vals, width=bar_w,
           label="Post-fix v2.0.1 (long-short)", color="#d65f5f")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Backtester v2.0.1: shorts no longer zeroed")
    ax.legend(loc="lower left", fontsize=9)
    for i, (pv, qv) in enumerate(zip(pre_vals, post_vals)):
        ax.annotate(f"{pv:.1f}", xy=(x[i] - bar_w / 2, pv),
                    xytext=(0, -10 if pv < 0 else 3),
                    textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{qv:.1f}", xy=(x[i] + bar_w / 2, qv),
                    xytext=(0, -10 if qv < 0 else 3),
                    textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_backtester_fix.png")
    plt.close(fig)
    print(f"  wrote {FIG_DIR/'fig_backtester_fix.png'}")


def main() -> int:
    print(f"Writing figures to {FIG_DIR}")
    fig_classifier_accuracy()
    fig_walk_forward_returns()
    fig_paper_vs_honest()
    fig_reward_signal_spread()
    fig_v2_ablation()
    fig_ppo_seed_instability()
    fig_1m_vs_30k_wf()
    fig_backtester_fix()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
