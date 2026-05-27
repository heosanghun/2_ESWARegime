"""
Trend Scanning labels (López de Prado, 2018, Ch. 5).

Reviewer #3 (ESWA-D-26-08980) pointed out that SMA-50 based ground truth
is a **lagging indicator** — the ground truth reflects a regime that has
already occurred, so the system cannot, in principle, predict it in real
time.

This module implements the **Trend Scanning** algorithm which assigns a
forward-looking label to every observation by scanning multiple future
horizons and selecting the one with the most statistically significant
trend (largest |t-value| of the OLS slope).

References
----------
M. López de Prado, "Advances in Financial Machine Learning",
Wiley, 2018, Chapter 5.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrendScanningLabeler:
    """
    Forward-looking trend labels via scanned OLS regressions.

    Parameters
    ----------
    horizon_min : int
        Minimum forward horizon (in bars) to scan.
    horizon_max : int
        Maximum forward horizon (in bars) to scan.
    t_threshold : float
        |t-value| threshold for classifying a trend as Bull/Bear.
        Below this magnitude the regime is labelled `Sideways`.
    use_log_price : bool
        If True, regress on log(price). Recommended for compounding assets
        such as BTC.

    Output labels
    -------------
    0 = Bear, 1 = Sideways, 2 = Bull (consistent with `RegimeGroundTruth`).
    """

    def __init__(
        self,
        horizon_min: int = 5,
        horizon_max: int = 20,
        t_threshold: float = 1.5,
        use_log_price: bool = True,
    ) -> None:
        if horizon_min < 3:
            raise ValueError("horizon_min must be >= 3 for stable OLS")
        if horizon_max < horizon_min:
            raise ValueError("horizon_max must be >= horizon_min")
        self.horizon_min = horizon_min
        self.horizon_max = horizon_max
        self.t_threshold = float(t_threshold)
        self.use_log_price = use_log_price

    @staticmethod
    def _tvalue(y: np.ndarray) -> Tuple[float, float]:
        """
        OLS y = a + b * t. Returns (slope_b, t-stat of slope).

        t-stat = b / SE(b), where
            SE(b) = sigma / sqrt(sum((t - t_mean)^2))
            sigma^2 = SSR / (n - 2)
        """
        n = len(y)
        if n < 3:
            return 0.0, 0.0
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        xc = x - x_mean
        yc = y - y_mean
        sxx = (xc ** 2).sum()
        if sxx <= 0:
            return 0.0, 0.0
        b = (xc * yc).sum() / sxx
        a = y_mean - b * x_mean
        resid = y - (a + b * x)
        ssr = (resid ** 2).sum()
        dof = n - 2
        if dof <= 0:
            return float(b), 0.0
        sigma2 = ssr / dof
        if sigma2 <= 0:
            return float(b), 0.0
        se_b = np.sqrt(sigma2 / sxx)
        if se_b == 0:
            return float(b), 0.0
        return float(b), float(b / se_b)

    def generate_labels(
        self,
        price_data: pd.Series,
        direction: str = "forward",
    ) -> pd.Series:
        """
        Generate regime labels for every timestamp.

        direction='forward' (default)
            Scan FUTURE windows (t..t+L) — López de Prado Trend Scanning.
        direction='backward'
            Scan PAST windows (t-L..t) — causal / no lookahead. Usable at
            inference without future price data.
        """
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")
        if direction == "backward":
            return self._generate_labels_backward(price_data)

        if not isinstance(price_data, pd.Series):
            raise TypeError("price_data must be a pandas Series")
        prices = price_data.astype(float).values
        if self.use_log_price:
            prices = np.log(np.clip(prices, 1e-12, None))

        n = len(prices)
        labels = np.full(n, 1, dtype=int)
        tvals_out = np.zeros(n, dtype=np.float64)
        horizons_out = np.zeros(n, dtype=np.int64)

        horizons: List[int] = list(
            range(self.horizon_min, self.horizon_max + 1)
        )

        for t in range(n):
            best_t = 0.0
            best_h = 0
            for h in horizons:
                end = t + h
                if end >= n:
                    break
                window = prices[t : end + 1]
                _, tval = self._tvalue(window)
                if abs(tval) > abs(best_t):
                    best_t = tval
                    best_h = h
            tvals_out[t] = best_t
            horizons_out[t] = best_h
            if best_t > self.t_threshold:
                labels[t] = 2  # Bull
            elif best_t < -self.t_threshold:
                labels[t] = 0  # Bear
            else:
                labels[t] = 1  # Sideways

        result = pd.Series(labels, index=price_data.index, dtype=int)
        # Stash diagnostics on the series for downstream logging
        result.attrs["t_values"] = pd.Series(tvals_out, index=price_data.index)
        result.attrs["selected_horizon"] = pd.Series(
            horizons_out, index=price_data.index
        )

        counts = result.value_counts().sort_index().to_dict()
        logger.info(
            "TrendScanning labels (forward): Bear=%d Sideways=%d Bull=%d "
            "(horizon=%d..%d, |t|>%.2f)",
            counts.get(0, 0),
            counts.get(1, 0),
            counts.get(2, 0),
            self.horizon_min,
            self.horizon_max,
            self.t_threshold,
        )
        return result

    def _generate_labels_backward(self, price_data: pd.Series) -> pd.Series:
        """Causal labels: scan past windows ending at t (no future data)."""
        if not isinstance(price_data, pd.Series):
            raise TypeError("price_data must be a pandas Series")
        prices = price_data.astype(float).values
        if self.use_log_price:
            prices = np.log(np.clip(prices, 1e-12, None))

        n = len(prices)
        labels = np.full(n, 1, dtype=int)
        tvals_out = np.zeros(n, dtype=np.float64)
        horizons_out = np.zeros(n, dtype=np.int64)
        horizons: List[int] = list(range(self.horizon_min, self.horizon_max + 1))

        for t in range(n):
            best_t = 0.0
            best_h = 0
            for h in horizons:
                start = t - h
                if start < 0:
                    continue
                window = prices[start : t + 1]
                _, tval = self._tvalue(window)
                if abs(tval) > abs(best_t):
                    best_t = tval
                    best_h = h
            tvals_out[t] = best_t
            horizons_out[t] = best_h
            if best_t > self.t_threshold:
                labels[t] = 2
            elif best_t < -self.t_threshold:
                labels[t] = 0
            else:
                labels[t] = 1

        result = pd.Series(labels, index=price_data.index, dtype=int)
        result.attrs["t_values"] = pd.Series(tvals_out, index=price_data.index)
        result.attrs["selected_horizon"] = pd.Series(
            horizons_out, index=price_data.index
        )
        counts = result.value_counts().sort_index().to_dict()
        logger.info(
            "TrendScanning labels (backward/causal): Bear=%d Sideways=%d Bull=%d "
            "(horizon=%d..%d, |t|>%.2f)",
            counts.get(0, 0),
            counts.get(1, 0),
            counts.get(2, 0),
            self.horizon_min,
            self.horizon_max,
            self.t_threshold,
        )
        return result

    @staticmethod
    def get_regime_name(label: int) -> str:
        return {0: "Bear", 1: "Sideways", 2: "Bull"}.get(int(label), "Unknown")


__all__ = ["TrendScanningLabeler"]
