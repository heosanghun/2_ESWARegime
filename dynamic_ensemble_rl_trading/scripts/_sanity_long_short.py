"""Sanity test: verify the long-short env produces correct PnL for trivial
strategies on the +62% test window.

Pure-long always-on   → should approx Buy & Hold (~+62%)
Pure-short always-on  → should approx -62%
All-flat (action=2)   → should stay ~$10,000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.data_processor import MarketDataHandler
from src.env.trading_env import MultiRegimeTradingEnv


def run_constant_action(action: int, allow_short: bool) -> float:
    dh = MarketDataHandler("data/raw/btcusdt_1h.csv")
    ohlcv = dh.load_data(start_date="2023-06-19", end_date="2023-12-19")
    state_data = pd.DataFrame(
        np.zeros((len(ohlcv), 1)),
        index=ohlcv.index,
        columns=["dummy"],
    )
    env = MultiRegimeTradingEnv(
        ohlcv_data=ohlcv,
        state_data=state_data,
        regime_type="Bull",
        initial_balance=10_000.0,
        transaction_fee=0.0005,
        slippage=0.0002,
        max_position=1.0,
        allow_short=allow_short,
    )
    obs, _ = env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(action)
    pv = env.portfolio_value
    ret = (pv - 10_000.0) / 10_000.0
    return ret


def main() -> None:
    for label, action, allow_short in [
        ("Strong Buy   (a=4, long-short)",  4, True),
        ("Strong Sell  (a=0, long-short)",  0, True),
        ("Hold         (a=2, long-short)",  2, True),
        ("Strong Buy   (a=4, long-only)",   4, False),
        ("Strong Sell  (a=0, long-only)",   0, False),  # should == flat
    ]:
        r = run_constant_action(action, allow_short)
        print(f"{label}: total_return = {r*100:+.2f}%")


if __name__ == "__main__":
    main()
