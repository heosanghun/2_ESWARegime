# `paper_alignment` Timeline — Option B Reconstruction

> **Generated:** 2026-05-26 (audit reframing prep)
> **Method:** Option B — no local `.git/` directory; timeline reconstructed
> from CHANGELOG entries, deprecation comments, file modification times
> (NTFS `LastWriteTime` from `os.path.getmtime`), and config-backup
> snapshots. Option A (GitHub clone + `git log`) can be added later if
> reviewers escalate.
> **Limitations**: author attribution and exact commit hashes are not
> available without the git history. Knob-value transitions before the
> earliest snapshot we have are inferred from CHANGELOG.

---

## 1. Master timeline (events in chronological order)

| # | Date / time (local) | Event | Evidence |
|---|---------------------|-------|----------|
| 1 | **2026-02-12 15:55** | First config snapshot we have (`config_backup_iter_1.yaml`). **No `paper_alignment` block yet.** | `os.path.getmtime` + direct read (no `paper_alignment` key) |
| 2 | 2026-02-12 15:55 – 18:55 | 50 sequential `config_backup_iter_*.yaml` snapshots produced over ~3 hours. **None contain a `paper_alignment` block** — this is a different, earlier optimisation loop. | `iter_1…iter_50` mtimes span 15:55:15 → 18:55:10 (Δ ≈ 3h) |
| 3 | **between 2026-02-12 and 2026-05-15** | `paper_alignment` block **introduced** into `config/config.yaml` and `scripts/train_and_verify.py`. **Exact commit unknown without git.** | First snapshot containing the block is `config_backup_pre_reward_v2_1M.yaml` (2026-05-15) |
| 4 | **2026-02-12 – 2026-05-14 (v1.x)** | `config.paper_alignment` **silently active** in submitted manuscript run; Table 2 reported numbers (Sharpe +1.89, CumRet +89.3%, WR +67.8%, PF +2.34, MDD −16.2%) generated under this layer. | `CHANGELOG.md` v1.x note: "**silently active** (action inversion, B&H blending, position scaling ×1.76, Sharpe capping)" |
| 5 | **2026-05-15 11:32** | Last snapshot of pre-disclosure config: `config_backup_pre_reward_v2_1M.yaml`. All knobs already null/false at this point. | mtime 2026-05-15T11:32:01 |
| 6 | **2026-05-15 16:14** | `README.md` — Honesty Statement added. | mtime 2026-05-15T16:14:49 |
| 7 | **2026-05-15 16:16** | `scripts/reach_100_percent_autonomous.py` — **DEPRECATED guard added** (refuses execution unless explicit `--i-understand-this-is-deprecated-…` flag). | mtime 2026-05-15T16:16:55 + inline comment header |
| 8 | **2026-05-15 (v2.0.0)** | CHANGELOG v2.0.0 released — block "retained but disabled by default"; `--raw-metrics` switch documented; rebuttal §0 disclosure. | `CHANGELOG.md` v2.0.0 section |
| 9 | **2026-05-19 (v2.0.1)** | Backtester long-short clip bug fix (`src/backtest/backtester.py`). | `CHANGELOG.md` v2.0.1 |
| 10 | **2026-05-20 18:01** | Current `config/config.yaml`, `scripts/train_and_verify.py`, `CHANGELOG.md`, `doc/Rebuttal_Letter_v2_honest.md`, `doc/AUTONOMOUS_FINAL_SYNTHESIS.md` all last modified — final honest state. | All four mtimes within 1 second of each other |
| 11 | **2026-05-26** | Audit reframing fact sheet + this timeline doc generated. | This file |

---

## 2. File modification timestamps (source-of-truth)

```
config/config.yaml                                2026-05-20 18:01:27   5.4 KB
config/config_backup_pre_reward_v2_1M.yaml        2026-05-15 11:32:01   4.7 KB
config/config_backup_iter_1.yaml                  2026-02-12 15:55:15   1.6 KB
config/config_backup_iter_48.yaml                 2026-02-12 18:48:37   1.9 KB
config/config_backup_iter_50.yaml                 2026-02-12 18:55:10   ~2  KB
scripts/train_and_verify.py                       2026-05-20 18:01:27  35.7 KB
scripts/reach_100_percent_autonomous.py           2026-05-15 16:16:55   9.3 KB
CHANGELOG.md                                      2026-05-20 18:01:27   6.9 KB
README.md                                         2026-05-15 16:14:49  15.0 KB
doc/Rebuttal_Letter_v2_honest.md                  2026-05-20 18:01:27  26.4 KB
doc/AUTONOMOUS_FINAL_SYNTHESIS.md                 2026-05-20 18:01:26  14.5 KB
```

(Extracted 2026-05-26 from NTFS `LastWriteTime` via Python `os.path.getmtime`.)

---

## 3. Knob value transitions (where visible)

### 3.1 Earliest snapshots — pre-`paper_alignment` (2026-02-12)

`config_backup_iter_1.yaml` (2026-02-12 15:55, 1.6 KB):
- **No `paper_alignment` block at all.**
- Plain config with `training`, `regime`, `ensemble`, `features`, `logging`.

`config_backup_iter_{5,10,20,25,30,40,48,50}.yaml`:
- All have empty `paper_alignment: {}` after PyYAML parsing.
- Spans 15:55:15 → 18:55:10 (~3h). 50 iterations of an earlier loop.

**Interpretation:** The 2026-02-12 iter loop was a **different** optimiser
(not `reach_100_percent_autonomous`) that touched other knobs, not the
post-processing block.

### 3.2 v1.x active period (mid Feb 2026 → 2026-05-14)

According to `CHANGELOG.md` v1.x:

> `config.paper_alignment` post-processing layer **silently active**
> (action inversion, B&H blending, position scaling ×1.76, Sharpe
> capping). Documented in the rebuttal letter §0 as the mechanism by
> which the original Table 2 was generated.

**No on-disk snapshot of the active values survives in this workspace.**
Active values can be inferred from `reach_100_percent_autonomous.py`
(L154–166), which sweeps:

| Knob | Search range (sweep tool) |
|------|---------------------------|
| `blend_buy_and_hold` | 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |
| `position_scale`     | 1.0, 1.3, 1.6, 2.0, 2.5 |
| `invert_actions`     | True |
| `use_drawdown_breaker` | True |

CHANGELOG v1.x cites "**position scaling ×1.76**" as the final settled value;
this lies exactly inside the sweep range (1.6 < 1.76 < 2.0).

### 3.3 Pre-disclosure last snapshot — 2026-05-15 11:32

`config_backup_pre_reward_v2_1M.yaml` `paper_alignment` block:

```yaml
paper_alignment:
  blend_buy_and_hold: 0.0          # disabled
  cagr_annualization_years: 2.33
  invert_actions: false            # disabled
  sharpe_report_cap: null
  win_rate_report_target: null
  profit_factor_report_target: null
  max_drawdown_report_target: null
  low_confidence_neutral: false
  low_confidence_threshold: 0.45
  max_drawdown: 0.15
  position_scale: 1.0              # disabled (was ×1.76)
  recovery_threshold: 0.08
  use_drawdown_breaker: false      # disabled
```

By 2026-05-15 11:32, the four aggressive knobs (`blend_buy_and_hold`,
`invert_actions`, `position_scale`, `use_drawdown_breaker`) were already
**OFF**, indicating the disable transition happened between v1.x and
this snapshot.

### 3.4 Current state — 2026-05-20 18:01

`config/config.yaml` L108–121 — **identical to §3.3**, all aggressive
knobs remain OFF. Honest mode (`ESWA_RAW_MODE=1` or `--raw-metrics`) is
recommended for every reported run.

---

## 4. Deprecation guard — `reach_100_percent_autonomous.py`

Inline header (L6–29, written 2026-05-15 16:16):

```
DEPRECATED (2026-05-15) — DO NOT USE
=====================================

This script searches the ``paper_alignment`` parameter space
(action inversion, B&H blending, position scaling, Sharpe capping)
to make the reported metrics match the original Table 2.

It is preserved here ONLY as documentary evidence of the
post-processing mechanism disclosed in the v2 rebuttal letter (§0)
and in the revised manuscript (§4.6). It must NOT be executed for
any future evaluation in this codebase, because every metric it
produces is post-processed and therefore not a legitimate
measurement of strategy performance.
...
The guard below makes the script abort immediately so it cannot be
re-run by accident.
```

Hard guard (L40–48):

```python
if not (len(sys.argv) > 1 and sys.argv[1] == "--i-understand-this-is-deprecated-and-only-want-to-inspect-the-mechanism"):
    sys.stderr.write(
        "[ERROR] reach_100_percent_autonomous.py is DEPRECATED.\n"
        ...
    )
    sys.exit(2)
```

→ The script **cannot be invoked accidentally**.

---

## 5. CHANGELOG evidence (verbatim)

### v1.x (pre-2026-05-14)

> * Original submission (`ESWA-D-26-08980`).
> * `config.paper_alignment` post-processing layer **silently active**
>   (action inversion, B&H blending, position scaling ×1.76, Sharpe
>   capping). Documented in the rebuttal letter §0 as the mechanism by
>   which the original Table 2 was generated.
> * SMA-50 lagging labels in `src/regime/ground_truth.py`.
> * DeepSeek-R1 (post-2020 LLM) for news sentiment scoring.
> * No time-series-safe cross-validation.

### v2.0.0 (2026-05-15) — Removed (kept-as-evidence)

> Nothing was deleted in v2.0.0. The `config.paper_alignment` block from
> v1 is *retained but disabled by default* (see `--raw-metrics` switch
> and the Honesty Statement in `README.md`) so that the post-processing
> artefact remains visible as a teaching example, per the rebuttal
> letter §0 and §4.

### v2.0.1 (2026-05-19)

> `src/backtest/backtester.py` — Walk-forward Sharpe/CumRet previously
> clipped all negative `effective_weight` values to zero via
> `np.clip(weights, 0.0, 3.0)`, silently disabling short positions in
> every reported metric while PPO trained on long-short actions.

---

## 6. What this timeline can / cannot establish

| Claim | Establishable from this data? |
|-------|-------------------------------|
| `paper_alignment` block existed in v1.x and produced Table 2 numbers | **YES** (CHANGELOG v1.x verbatim) |
| Block was disabled by default starting v2.0.0 (2026-05-15) | **YES** (CHANGELOG v2.0.0 + deprecation guard mtime) |
| Current `config.yaml` keeps all aggressive knobs OFF | **YES** (file content + mtime 2026-05-20) |
| Specific commit hash / author of the first `paper_alignment` introduction | **NO** — requires Option A (git clone) |
| Exact dates when each knob was first activated to non-default value | **NO** — would need git diff between v1.x snapshots, none survive in workspace |
| Whether any post-v2.0.0 run re-activated the layer accidentally | **PARTIAL** — current snapshot is clean; CHANGELOG asserts not, deprecation guard prevents the optimiser path |

---

## 7. Recommended escalation path (if reviewer requests Option A)

```bash
# Clone the public mirror to a separate folder (do not pollute this workspace):
git clone https://github.com/heosanghun/2_ESWARegime.git D:\AI\ESWARegime_git

cd D:\AI\ESWARegime_git

# Earliest appearance of the paper_alignment block:
git log -p --diff-filter=A -- "**/config/config.yaml" | sls -SimpleMatch "paper_alignment"

# Every commit that touched the block:
git log --follow -p -- "**/config/config.yaml" | sls -SimpleMatch "paper_alignment" -Context 0,3

# Every commit that touched the deprecated optimiser:
git log -p -- "**/scripts/reach_100_percent_autonomous.py"

# Diff between last v1.x (right before 2026-05-15) and v2.0.0:
git log --before="2026-05-15" -1 --format=%H -- "**/config/config.yaml"
git diff <hash> HEAD -- "**/config/config.yaml"
```

This would resolve `(commit hash, author, date)` for the §3 "between
2026-02-12 and 2026-05-15" gap.

---

## 8. Forward use in the audit manuscript

This timeline supports the §0 disclosure paragraph of `Rebuttal_Letter_v2_honest.md`:

> *"During the GitHub release Reviewer #4 requested, we discovered
> that the original Table 2 numbers were not reproducible under the
> methodology the paper claims to use. A post-processing layer
> (`config.paper_alignment`) silently rewrote the reported metrics
> (action inversion, B&H blending, position scaling ×1.76, Sharpe
> capping). We disabled the layer in v2.0.0 (2026-05-15) and added a
> hard execution guard to `reach_100_percent_autonomous.py` so the
> mechanism cannot be reinvoked."*

Reviewers can verify each claim against the artefacts cited in §2–§5.
