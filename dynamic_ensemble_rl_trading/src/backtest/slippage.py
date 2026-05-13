"""
Volatility-aware (ATR-based) slippage models.

Reviewer #3 (ESWA-D-26-08980) flagged that a fixed 0.02% slippage is
unrealistic for the 2022 crypto crash. This module provides two more
honest slippage specifications:

* ``ATRSlippageModel`` — slippage scales with the current Average True
  Range (volatility), capped between min/max. Realistic for venues like
  Binance Spot during 2021-2023.
* ``ConservativeFixedSlippage`` — fixed 0.10% used as a sanity-check
  upper bound recommended by the reviewer.

Both expose ``rate(t, ohlcv) -> float`` so they can plug into the
backtester without changing the public API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ATRSlippageModel:
    """
    Slippage as a function of ATR-normalized volatility.

        slip_t = clip(base + k * ATR_t / Close_t, min_slip, max_slip)

    Parameters
    ----------
    window : int
        ATR window length (bars).
    base : float
        Slippage floor (e.g. exchange minimum).
    k : float
        Scale factor multiplying ATR/Close.
    min_slip, max_slip : float
        Hard clipping bounds (typical: 0.0002 .. 0.005).
    """

    window: int = 14
    base: float = 0.0002
    k: float = 0.5
    min_slip: float = 0.0002
    max_slip: float = 0.005

    def precompute(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Compute ATR-based slippage rates for every row in `ohlcv`."""
        high = ohlcv['high'].astype(float)
        low = ohlcv['low'].astype(float)
        close = ohlcv['close'].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self.window, min_periods=1).mean()
        rate = self.base + self.k * (atr / close.replace(0, np.nan))
        rate = rate.clip(lower=self.min_slip, upper=self.max_slip)
        rate = rate.fillna(self.base)
        return rate.rename("slippage_rate")


@dataclass
class ConservativeFixedSlippage:
    rate_value: float = 0.001  # 0.10%

    def precompute(self, ohlcv: pd.DataFrame) -> pd.Series:
        return pd.Series(self.rate_value, index=ohlcv.index, name="slippage_rate")


def build_slippage_model(cfg: dict) -> Optional[object]:
    """Factory: read `training.slippage_model` from config."""
    if cfg is None:
        return None
    name = str(cfg.get('slippage_model', 'fixed')).lower()
    if name == 'atr':
        ats = cfg.get('atr_slippage', {}) or {}
        return ATRSlippageModel(
            window=int(ats.get('window', 14)),
            base=float(ats.get('base', 0.0002)),
            k=float(ats.get('k', 0.5)),
            min_slip=float(ats.get('min_slip', 0.0002)),
            max_slip=float(ats.get('max_slip', 0.005)),
        )
    if name == 'conservative':
        cf = cfg.get('conservative_slippage', {}) or {}
        return ConservativeFixedSlippage(
            rate_value=float(cf.get('rate', 0.001))
        )
    return None


__all__ = [
    "ATRSlippageModel",
    "ConservativeFixedSlippage",
    "build_slippage_model",
]
