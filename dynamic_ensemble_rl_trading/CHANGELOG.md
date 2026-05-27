# Changelog

All notable changes to this codebase for the ESWA-D-26-08980
revision are recorded here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project loosely follows semantic versioning with the
caveat that the public API at this stage of the manuscript is the
``scripts/train_and_verify.py`` /
``scripts/run_walk_forward.py`` command-line interface.

## [v2.0.1] — 2026-05-19 (Backtester long-short fix; apply metrics after 1M WF completes)

### Fixed

- **`src/backtest/backtester.py`** — Walk-forward Sharpe/CumRet
  previously clipped all negative ``effective_weight`` values to
  zero via ``np.clip(weights, 0.0, 3.0)``, silently disabling
  short positions in every reported metric while PPO trained on
  long-short actions. The backtester now uses the same
  ``LONG_SHORT_WEIGHT_MAP`` as ``MultiRegimeTradingEnv`` and
  symmetric clipping ``[-max_position, +max_position]``.
- **`scripts/train_and_verify.py`**, **`scripts/evaluate.py`** —
  forward ``allow_short`` and ``max_position`` from config into
  ``Backtester``.

### Added

- **`scripts/_sanity_backtester_long_short.py`** — constant-weight
  sanity checks (long/short/flat × up/down drift).
- **`scripts/_rebacktest_walk_forward_folds.py`** — re-run STEP 3
  backtest only on existing fold models (no PPO retraining).

### Added (autopilot)

- **`scripts/_post_wf_autopilot.py`** — watches for walk-forward
  completion, then automatically runs sanity → rebacktest → scenario
  → Bonferroni → comparison report (T1–T5). State in
  ``results/<subdir>/autopilot_status.json``.
- **`doc/POST_WF_AUTOPILOT.md`** — operator guide.
- **`run_walk_forward.py --autopilot`** — chain autopilot after all
  folds on future runs.

### Notes

- Folds already backtested **before this fix** (including the
  in-flight 1M WF run started 2026-05-17) used the old clip
  behaviour. After Fold 5 training finishes, refresh metrics with:
  ``python scripts/_rebacktest_walk_forward_folds.py --subdir walk_forward_reward_v2_1M --raw-metrics``

## [v2.0.0] — 2026-05-15

### Added

- **`src/env/rewards.py`** — Reward function v2: direction-aligned
  shaping (`α·w·r`) on top of the realised PnL signal, with
  regime-specific bonuses (Bull long-bias, Bear short-bias with
  asymmetric wrong-side penalty, Sideways position-magnitude
  penalty). Sanity harness in `scripts/_sanity_reward_v2.py`
  reports a ~5× correct-vs-wrong-side reward spread under v2 vs
  v1. The legacy positional-only API (no `target_weight` /
  `bar_return`) is preserved as a back-compat fallback.
- **`src/env/trading_env.py`** — environment now passes the signed
  `target_weight` and bar return to the reward calculator so v2
  shaping can be applied.
- **`src/regime/regime_classifier.py`** — constructor now exposes
  the full XGBoost regularisation surface (`reg_lambda`,
  `reg_alpha`, `colsample_bytree`, `subsample`,
  `min_child_weight`, `early_stopping_rounds`). The pipeline
  driver forwards all of them from `config.yaml`.
- **`src/data/feature_fusion.py`** — `use_visual` flag added.
  When `False`, the 512-D ResNet-18 candlestick branch is
  dropped end-to-end. Default flipped to `False` in
  `config.yaml.features.use_visual`.
- **`scripts/_sanity_reward_v2.py`** — verifies the v2 reward
  monotonically rewards correct-direction positions in each
  regime, with a back-compat smoke test for v1 callers.
- **`scripts/_generate_figures.py`** — produces four
  publication-ready figures: classifier accuracy under each
  labelling scheme, walk-forward per-fold returns, paper-vs-honest
  comparison with Bonferroni CIs, and the v1-vs-v2 reward spread.
- **`scripts/_compare_v1_v2_walk_forward.py`** — aggregates the
  three walk-forward runs into a single ablation table and
  comparison figure.
- **`scripts/_run_full_v2_walk_forward.py`** — orchestrator that
  waits for the reward-only walk-forward to land and then
  automatically launches the combined-v2 walk-forward.
- **`doc/Rebuttal_Letter_v2_honest.md`** — fully rewritten
  rebuttal letter (honest reframing); §5 now lists delivered v2
  architectural improvements; §6 elaborates citation responses.
- **`doc/Manuscript_Revision_Guide.md`** — Edit 6.5 (architectural
  fixes) added.
- **`doc/AUTONOMOUS_OVERNIGHT_REPORT.md`** — Day-2 section added
  documenting the v2 architectural improvements and the
  reward-only ablation finding.
- **`doc/AUTONOMOUS_FINAL_SYNTHESIS.md`** — consolidated single
  source-of-truth for every performance number cited in the
  rebuttal and the manuscript. Includes the four single-split
  retrain logs (`results/verification/honest_retrain*.log`), the
  three walk-forward summaries, and the seed-instability finding.
- **`results/verification/honest_retrain*.log`** — four independent
  single-split retrains of the full pipeline at two training-budget
  levels. The (`_v3`, `_v4`) pair at 30 k timesteps differ by
  Sharpe 21.4 under identical configuration; the (`_v1`, `_v2`)
  pair at 1 M timesteps differ by Sharpe 1.76. Establishes the
  PPO seed-instability finding now documented in Manuscript §4.8
  and §5.L6.
- **`doc/Editor_Extension_Request.md`** — drafted (deferred to the
  author to finalise).

### Changed

- **`config/config.yaml`** — `regime_classifier` block now exposes
  v2 regularisation knobs (top-level *and* `hyperparameters`
  block). `features.use_visual` defaulted to `false`.
- **`scripts/train_and_verify.py`** — forwards every classifier
  regularisation parameter from `config.yaml` to the
  `RegimeClassifier` constructor (v1 silently ignored all of
  them). Also forwards `cfg.features.use_visual` to `FeatureFusion`.

### Removed

- Nothing was deleted in v2.0.0. The `config.paper_alignment`
  block from v1 is *retained but disabled by default* (see
  `--raw-metrics` switch and the Honesty Statement in `README.md`)
  so that the post-processing artefact remains visible as a
  teaching example, per the rebuttal letter §0 and §4.

### Reproduction targets

* v1 baseline (Trend-Scanning labels, reward-v1, classifier-v1,
  visual-on): `results/walk_forward/`.
* Reward-only v2 ablation (Trend-Scanning labels, reward-v2,
  classifier-v1, visual-on): `results/walk_forward_reward_v2/`.
* Combined-v2 (Trend-Scanning labels, reward-v2, classifier-v2,
  visual-off): `results/walk_forward_reward_v2_full/`.

All three are reproduced by

```bash
python scripts/run_walk_forward.py --label-method trend_scanning --subdir <name>
```

with `config/config.yaml` toggled accordingly.

## [v1.x] — pre-2026-05-14

* Original submission (`ESWA-D-26-08980`).
* `config.paper_alignment` post-processing layer **silently active**
  (action inversion, B&H blending, position scaling ×1.76, Sharpe
  capping). Documented in the rebuttal letter §0 as the mechanism
  by which the original Table 2 was generated.
* SMA-50 lagging labels in `src/regime/ground_truth.py`.
* DeepSeek-R1 (post-2020 LLM) for news sentiment scoring.
* No time-series-safe cross-validation.
