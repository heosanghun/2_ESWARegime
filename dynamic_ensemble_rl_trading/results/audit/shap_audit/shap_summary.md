# Classifier SHAP Audit (Reviewer #3 item #4)

_Generated: 2026-05-27T00:17:49_  
**Source model:** `models/walk_forward_reward_v2/fold_5/regime_classifier/model.json`
**Sample slice:** 2023-08-19 → 2023-12-19 (500 subsampled bars)
**Feature dim:** 539  (technical 19 + visual 512 + sentiment 8 by default)

## Feature-group contributions (mean |SHAP|)

| Group | Size | Sum mean |SHAP| | % of total | Mean / feature |
|---|---:|---:|---:|---:|
| technical | 19 | 0.0854 | 3.01% | 4.4965e-03 |
| visual | 512 | 2.6720 | 94.25% | 5.2188e-03 |
| sentiment | 8 | 0.0775 | 2.73% | 9.6815e-03 |

## Feature-group contributions (XGBoost gain)

| Group | Sum gain | % of total | Mean / feature |
|---|---:|---:|---:|
| technical | 91.97 | 2.65% | 4.8406e+00 |
| visual | 3336.35 | 96.08% | 6.5163e+00 |
| sentiment | 44.18 | 1.27% | 5.5222e+00 |

## Interpretation

Reviewer #3 raised concerns about the ResNet-18 candlestick image branch contributing noise rather than predictive signal. The table above quantifies how the trained XGBoost classifier uses each feature group:

* If the **visual** share is large (e.g. > 50%) AND the classifier still has only marginal-above-chance accuracy (≈ 46% on Trend-Scanning labels, 33% chance), the classifier is consuming a large amount of *capacity* on the 512-D ResNet embedding without converting that capacity into discriminative power. This is consistent with Reviewer #3's intuition that the visual branch contributes mostly noise: the model fits to it heavily during training but the resulting decision boundary does not generalise.

* If the **technical** share is small but the model ablation in Section 4 of the manuscript shows that *removing* the visual branch does not materially change test accuracy, this is direct evidence that the 19-D technical features carry the same information more compactly. Recommended action: report the SHAP share as a complement to the ablation accuracy table, framing the visual branch as **redundant capacity rather than zero contribution**.
