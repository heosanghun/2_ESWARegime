# One-Page Overnight Autonomous Work Summary (User Return Brief)

_Written: 2026-05-14 22:20 KST; autonomous run scheduled to end 2026-05-15 10:00 KST_

## One-line summary

> After applying Reviewer #3 methodology requirements honestly (FinBERT, Trend Scanning,
> Walk-Forward CV, ATR slippage, etc.), we **statistically confirmed that the current code /
> methodology cannot reproduce the manuscript Table 2 figures**. The root causes are (a)
> **lagged SMA-50 labels** inflating classifier accuracy via self-referential leakage, and
> (b) the **`paper_alignment`** post-processing layer rewriting measured metrics.

## Headline numbers (honest measurement, 5-fold walk-forward, raw)

| Metric | Manuscript | Honest measurement | 95% CI |
|--------|----------:|-------------------:|--------|
| Sharpe Ratio | 1.89 | **−20.50** | [−23.98, −16.12] |
| Cumulative Return | 0.893 | **−0.737** | [−0.863, −0.586] |
| Win Rate | 0.678 | **0.101** | [0.045, 0.174] |

All metrics remain statistically different after Bonferroni correction.  
**Average underperformance vs Buy & Hold: −85.6 pp.**

## Decisive diagnosis (Fold 1, same test window)

| Label | Action | Classifier accuracy | Cum Ret | vs B&H |
|-------|--------|--------------------:|--------:|-------:|
| Trend Scanning | Long-Short | ~47% | −84.9% | −41 pp |
| SMA-50 | Long-Short | ~90% | −52.7% | −9 pp |
| SMA-50 | Long-Only | ~90% | −76.8% | −33 pp |
| (B&H) | — | — | −43.9% | 0 pp |

**Even ~90% classifier accuracy with SMA labels fails to beat B&H.**  
→ Additional evidence that the hourly PPO reward does not learn the intended regime-specialised behaviour.

## 5-fold comparison (Trend Scanning vs SMA-50)

| Label method | Mean Cum Ret | Mean Sharpe | Mean consistency |
|--------------|-------------:|------------:|-----------------:|
| Trend Scanning (Reviewer #3) | **−73.7%** | **−20.50** | 4.7% |
| **SMA-50 (original manuscript)** | **−52.3%** | **−13.07** | 10.1% |
| Manuscript Table 2 reported | +89.3% | +1.89 | — |
| Buy & Hold | +12.0% | — | — |

**Even with the original SMA labelling, honest measurement yields Sharpe −13.07** (gap vs manuscript +1.89 = 14.96).  
No mathematical mechanism explains +1.89 without `paper_alignment` post-processing.

## Top three deliverables

1. **`doc/AUTONOMOUS_OVERNIGHT_REPORT.md`** — full analysis report
2. **`doc/Rebuttal_Letter_draft.md`** (Honest Addendum appended)
3. **`results/walk_forward/`** — all 5-fold raw outputs + statistical tests + Table 1

## Decisions pending (after 10:00 return)

1. **Option A — Transparent revision**: revise manuscript with honest numbers.
   - Replace Table 2 with walk-forward fold means + 95% CI
   - Add a Limitations section
2. **Option B — Model rebuild**: replace classifier backbone with a sequence model and retrain
   (current classifier barely learns on forward labels).
3. **Option C — Withdraw and redesign** before resubmission.

## What we did *not* do

- Re-enable `paper_alignment` (action inversion, B&H blending, scaling, etc.)
  → would fabricate metrics; never activated.
- Git push (no external state change before user confirmation).
- Revert labels to SMA-only (would ignore Reviewer #3's core critique).

## Quick actions on return

```powershell
# 1) Open full honest report
notepad doc/AUTONOMOUS_OVERNIGHT_REPORT.md

# 2) Walk-forward 5-fold results
notepad results/walk_forward/summary.md
notepad results/walk_forward/diagnostics.md
notepad results/walk_forward/statistical_tests.md
notepad results/walk_forward/table1_classifier_per_fold.md
notepad results/walk_forward/clf_feature_ablation.md

# 3) Reviewer #3 compliance (FinBERT, Trend Scanning, Walk-Forward enabled)
notepad results/verification/reviewer3_compliance.md
```
