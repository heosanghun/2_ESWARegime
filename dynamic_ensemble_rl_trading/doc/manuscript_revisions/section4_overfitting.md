# Overfitting & Regularization (Reviewer #2, item #10)

We explicitly characterise the regularisation pipeline that prevents
the 15-agent system from memorising historical patterns:

1. **Walk-Forward Expanding Window CV.** Hyper-parameter selection is
   performed with `WalkForwardExpandingCV(n_splits=5)`
   (`src/validation/walk_forward_cv.py`). No future bar is ever used
   to score a candidate.
2. **Purged K-Fold with embargo.** For label-spanning estimators (e.g.
   the regime classifier with Trend-Scanning labels) we use
   `PurgedKFold(embargo=0.01)` so that test folds are insulated from
   training samples whose forward label window overlaps the test
   horizon.
3. **PPO entropy bonus & clipping.** PPO is configured with
   \(c_2 = 0.01\) and clip range \(\epsilon = 0.2\); the entropy term
   prevents premature policy collapse to historical action patterns.
4. **Weight decay & policy network width.** All PPO MLPs use modest
   width (256-256) and a per-layer weight decay of \(10^{-4}\) at the
   optimiser level — a value chosen to match the small effective
   sample size (\(\sim\)19 k hourly bars).
5. **Agent diversity by seed.** Within each regime pool the five
   agents share the architecture but use different random seeds,
   inducing exploration diversity and reducing single-agent overfit.
6. **Dynamic weighting acts as model averaging.** Re-weighting agents
   by rolling Sharpe with a softmax temperature 10 down-weights
   over-fit experts whenever their out-of-sample performance degrades.

Together these mechanisms ensure that the strong out-of-sample
Sharpe Ratio is not an artefact of in-sample tuning.
