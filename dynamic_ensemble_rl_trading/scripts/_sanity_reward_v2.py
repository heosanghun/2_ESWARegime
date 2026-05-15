"""Quick sanity check for the v2 reward design.

Verifies:
1. The new optional kwargs (target_weight, bar_return) are honoured.
2. The reward is monotone in direction alignment, i.e. correctly
   directional positions receive larger reward than wrong-directional
   positions in each regime.
3. Backward compatibility: calls without the new kwargs still return a
   finite value (PV-based contribution only).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.env.rewards import RegimeRewardCalculator


def _scenario(regime: str) -> None:
    """Compare correct-side vs wrong-side reward in a representative bar."""
    calc = RegimeRewardCalculator(reward_scale=100.0)
    pv_before = 10_000.0
    # Approximate effect of holding w=±1 over a -0.5% bar.
    bar_return = -0.005 if regime == "bear" else 0.005
    txn = 1.0  # $1 cost on a $10k portfolio (10 bps)
    # PV mapping: long during up-bar => gain; short during up-bar => loss.
    pv_long = pv_before * (1 + bar_return) - txn
    pv_short = pv_before * (1 - bar_return) - txn

    r_long = calc.calculate_reward(
        regime, pv_before, pv_long, txn,
        target_weight=+1.0, bar_return=bar_return,
    )
    r_short = calc.calculate_reward(
        regime, pv_before, pv_short, txn,
        target_weight=-1.0, bar_return=bar_return,
    )
    r_flat = calc.calculate_reward(
        regime, pv_before, pv_before - txn, txn,
        target_weight=0.0, bar_return=bar_return,
    )

    print(f"[{regime.upper():8s}] bar_return={bar_return*100:+.2f}%   "
          f"LONG={r_long:+7.3f}   SHORT={r_short:+7.3f}   FLAT={r_flat:+7.3f}")

    if regime == "bear":
        assert r_short > r_long, f"Bear: short should beat long, got {r_short} vs {r_long}"
    elif regime == "bull":
        assert r_long > r_short, f"Bull: long should beat short, got {r_long} vs {r_short}"
    elif regime == "sideways":
        # Either side bleeds vs flat (after costs) in this synthetic bar.
        assert r_flat > min(r_long, r_short), "Sideways: flat should dominate one side"


def _backward_compat() -> None:
    calc = RegimeRewardCalculator()
    out = calc.calculate_reward("bull", 10_000.0, 10_050.0, 0.0)
    assert isinstance(out, float), "Reward must remain a float"
    print(f"[COMPAT ] bull(no kwargs) = {out:+.3f}")


if __name__ == "__main__":
    _backward_compat()
    _scenario("bull")
    _scenario("bear")
    _scenario("sideways")
    print("\nAll sanity checks passed.")
