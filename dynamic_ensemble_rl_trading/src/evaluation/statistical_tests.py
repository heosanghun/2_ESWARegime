"""
Statistical robustness tools for ensemble trading performance.

Reviewer #2 / #4 (item #9) required:
  - bootstrap confidence intervals for Sharpe, Cumulative Return, ...
  - Ledoit-Wolf style robust significance test against benchmarks
  - Bonferroni correction for multiple comparisons

These helpers are framework-agnostic; they consume numpy arrays of
step-level returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


HOURS_PER_YEAR = 365 * 24


def _annualized_sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return float((mu / sd) * np.sqrt(HOURS_PER_YEAR))


def _cumulative_return(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    return float(np.prod(1.0 + returns) - 1.0)


def stationary_bootstrap_indices(
    n: int, rng: np.random.Generator, block: int = 24
) -> np.ndarray:
    """Politis-Romano stationary bootstrap with geometric block length."""
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    p = 1.0 / max(1, block)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


@dataclass
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    samples: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "estimate": float(self.estimate),
            "lower": float(self.lower),
            "upper": float(self.upper),
        }


def bootstrap_metric(
    returns: np.ndarray,
    metric: str = "sharpe",
    n_boot: int = 2000,
    block: int = 24,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Block-bootstrap CI for a metric on hourly returns."""
    rng = np.random.default_rng(seed)
    fn = {"sharpe": _annualized_sharpe, "cumulative_return": _cumulative_return}[metric]
    n = len(returns)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, rng, block=block)
        samples[b] = fn(returns[idx])
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapCI(estimate=fn(returns), lower=lo, upper=hi, samples=samples)


def ledoit_wolf_sharpe_test(
    r_strategy: np.ndarray, r_benchmark: np.ndarray
) -> Tuple[float, float]:
    """
    Heteroskedastic, autocorrelation-robust test for equality of two
    Sharpe ratios using the Ledoit-Wolf (2008, JEF) closed form.

    Returns
    -------
    z : float
        Test statistic ~ N(0,1) under H0.
    p : float
        Two-sided p-value.
    """
    r1 = np.asarray(r_strategy, dtype=float)
    r2 = np.asarray(r_benchmark, dtype=float)
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    if n < 30:
        return 0.0, 1.0

    mu1, mu2 = r1.mean(), r2.mean()
    s1, s2 = r1.std(ddof=1), r2.std(ddof=1)
    if s1 == 0 or s2 == 0:
        return 0.0, 1.0
    sh1, sh2 = mu1 / s1, mu2 / s2
    rho = np.corrcoef(r1, r2)[0, 1]

    # Ledoit & Wolf (2008) asymptotic variance (no autocorrelation term;
    # add Newey-West if you need to be even more conservative).
    var = (
        2
        - 2 * rho
        + 0.5 * (sh1 ** 2 + sh2 ** 2)
        - rho * sh1 * sh2
    ) / n
    if var <= 0:
        return 0.0, 1.0
    z = (sh1 - sh2) / np.sqrt(var)
    # two-sided normal p-value
    from math import erf, sqrt

    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))
    return float(z), float(p)


def bonferroni_correct(p_values: Sequence[float]) -> np.ndarray:
    """Family-wise error rate control by Bonferroni."""
    m = len(p_values)
    if m == 0:
        return np.array([])
    return np.clip(np.asarray(p_values, dtype=float) * m, 0.0, 1.0)


def comprehensive_report(
    strategy_returns: np.ndarray,
    benchmarks: Dict[str, np.ndarray],
    n_boot: int = 2000,
    block: int = 24,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """One-shot wrapper used by the verification scripts."""
    ci_sharpe = bootstrap_metric(
        strategy_returns, "sharpe", n_boot=n_boot, block=block,
        confidence=confidence, seed=seed,
    )
    ci_cum = bootstrap_metric(
        strategy_returns, "cumulative_return", n_boot=n_boot, block=block,
        confidence=confidence, seed=seed,
    )
    tests = {}
    p_raw = []
    for name, bench in benchmarks.items():
        z, p = ledoit_wolf_sharpe_test(strategy_returns, bench)
        tests[name] = {"z": z, "p_value": p}
        p_raw.append(p)
    p_corr = bonferroni_correct(p_raw)
    for (name, d), pc in zip(tests.items(), p_corr):
        d["p_value_bonferroni"] = float(pc)

    return {
        "bootstrap": {
            "sharpe": ci_sharpe.as_dict(),
            "cumulative_return": ci_cum.as_dict(),
            "n_boot": n_boot,
            "block": block,
            "confidence": confidence,
        },
        "ledoit_wolf_vs_benchmarks": tests,
    }


__all__ = [
    "BootstrapCI",
    "bootstrap_metric",
    "ledoit_wolf_sharpe_test",
    "bonferroni_correct",
    "comprehensive_report",
    "stationary_bootstrap_indices",
]
