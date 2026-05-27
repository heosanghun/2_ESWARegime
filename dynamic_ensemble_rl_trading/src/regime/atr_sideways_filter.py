"""ATR-based sideways market filter — force flat when volatility is too low."""

from __future__ import annotations

import pandas as pd


def compute_atr_pct_series(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    """Return ATR/Close ratio (fraction, e.g. 0.005 = 0.5%)."""
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=1).mean()
    return (atr / close.replace(0, float("nan"))).fillna(0.0)


def is_sideways_bar(atr_pct: float, threshold: float) -> bool:
    """True when normalized ATR is below threshold → chop / low-vol sideways."""
    return atr_pct < threshold
