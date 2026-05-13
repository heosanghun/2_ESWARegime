"""
Time-series safe cross-validation splitters.

Reviewer #3 (ESWA-D-26-08980) flagged that the original hyper-parameter
tuning used standard K-fold cross-validation, which references future
data and therefore contaminates the training procedure.

This module provides two sklearn-compatible splitters that respect the
chronological ordering of financial data:

1. ``WalkForwardExpandingCV`` — anchored, growing training window with a
   rolling test window. This is what the ESWA paper actually describes
   in Section 4.1 ("Walk-Forward Expanding Window").

2. ``PurgedKFold`` — López de Prado's purged K-fold with an *embargo*
   period between train and test folds, eliminating leakage from labels
   that overlap multiple horizons (relevant for Trend Scanning labels).

A small helper, ``tune_regime_classifier``, demonstrates how to use the
splitters with the project's XGBoost regime classifier and returns the
best hyper-parameter combination plus per-fold metrics.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-Forward Expanding Window
# ---------------------------------------------------------------------------


class WalkForwardExpandingCV:
    """
    Anchored expanding-window walk-forward splitter.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    test_size : int or float
        Size of each test fold. If float in (0, 1), interpreted as a
        fraction of the total number of samples.
    min_train_size : int or float, optional
        Minimum size of the *first* training fold. Same units as
        ``test_size``. Defaults to total - n_splits * test_size.
    gap : int
        Number of samples to skip between train end and test start
        (helps when labels span multiple bars).

    Yields
    ------
    train_idx, test_idx : np.ndarray
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.1,
        min_train_size: Optional[float] = None,
        gap: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = int(gap)

    @staticmethod
    def _resolve(size: float, n: int) -> int:
        if isinstance(size, float) and 0.0 < size < 1.0:
            return max(1, int(round(size * n)))
        return int(size)

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        test_size = self._resolve(self.test_size, n)
        if self.min_train_size is not None:
            min_train = self._resolve(self.min_train_size, n)
        else:
            min_train = max(test_size, n - self.n_splits * test_size)
        if min_train + test_size + self.gap > n:
            raise ValueError(
                "Not enough samples for the requested splits "
                f"(n={n}, min_train={min_train}, test_size={test_size}, "
                f"gap={self.gap})."
            )

        remaining = n - (min_train + self.gap)
        max_splits = max(1, remaining // max(1, test_size))
        if max_splits < self.n_splits:
            logger.warning(
                "Requested %d splits but only %d fit. Reducing.",
                self.n_splits,
                max_splits,
            )
        actual_splits = min(self.n_splits, max_splits)

        for k in range(actual_splits):
            train_end = min_train + k * test_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n)
            if test_start >= n:
                break
            train_idx = np.arange(0, train_end, dtype=np.int64)
            test_idx = np.arange(test_start, test_end, dtype=np.int64)
            if len(test_idx) == 0:
                break
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Purged K-Fold with Embargo (López de Prado, AFML 2018, Ch.7)
# ---------------------------------------------------------------------------


class PurgedKFold:
    """
    K-Fold splitter with a *purge* and *embargo* period to prevent
    information leakage when labels span multiple bars.

    Parameters
    ----------
    n_splits : int
        Number of folds (>=2).
    embargo : int or float
        Number of samples to drop immediately *after* each test fold.
        Float in (0,1) is treated as a fraction of n.
    label_horizon : int
        Length (in bars) of the look-ahead window used to generate
        labels. The training fold is purged of any sample whose
        label window overlaps the test fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo: float = 0.01,
        label_horizon: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if label_horizon < 0:
            raise ValueError("label_horizon must be >= 0")
        self.n_splits = int(n_splits)
        self.embargo = embargo
        self.label_horizon = int(label_horizon)

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        indices = np.arange(n)
        fold_size = n // self.n_splits
        if isinstance(self.embargo, float) and 0.0 < self.embargo < 1.0:
            embargo = max(0, int(round(self.embargo * n)))
        else:
            embargo = int(self.embargo)

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            purge_start = max(0, test_start - self.label_horizon)
            purge_end = min(n, test_end + embargo)
            mask = np.ones(n, dtype=bool)
            mask[purge_start:purge_end] = False
            train_idx = indices[mask]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Hyper-parameter tuner for the XGBoost regime classifier
# ---------------------------------------------------------------------------


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    fold_scores: List[float] = field(default_factory=list)
    all_scores: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 without scikit-learn dependency."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s: List[float] = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def tune_regime_classifier(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, Sequence[Any]],
    splitter: Optional[Any] = None,
    base_params: Optional[Dict[str, Any]] = None,
) -> TuningResult:
    """
    Grid-search the XGBoost regime classifier under a time-series safe
    splitter. Returns the best parameter set by mean macro-F1 across
    folds.

    Parameters
    ----------
    X, y : np.ndarray
        Feature matrix and integer labels (0/1/2).
    param_grid : dict
        Mapping of hyper-parameter name -> list of candidate values.
    splitter : sklearn-compatible splitter, optional
        Defaults to :class:`WalkForwardExpandingCV(n_splits=5)`.
    base_params : dict, optional
        Common XGBoost parameters merged into every candidate.
    """
    try:
        from src.regime.regime_classifier import RegimeClassifier
    except ImportError:  # pragma: no cover - import path fallback
        from regime.regime_classifier import RegimeClassifier  # type: ignore

    splitter = splitter or WalkForwardExpandingCV(n_splits=5, test_size=0.1)
    base_params = dict(base_params or {})

    keys = list(param_grid.keys())
    grids = list(itertools.product(*[param_grid[k] for k in keys]))

    all_scores: List[Tuple[Dict[str, Any], float]] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    best_fold_scores: List[float] = []

    for combo in grids:
        params = {**base_params, **dict(zip(keys, combo))}
        fold_scores: List[float] = []
        for train_idx, test_idx in splitter.split(X):
            clf = RegimeClassifier(**params)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            fold_scores.append(_f1_macro(y[test_idx], y_pred))
        mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
        all_scores.append((params, mean_score))
        logger.info("Params=%s -> macro-F1=%.4f", params, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_fold_scores = fold_scores

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        fold_scores=best_fold_scores,
        all_scores=all_scores,
    )


__all__ = [
    "WalkForwardExpandingCV",
    "PurgedKFold",
    "TuningResult",
    "tune_regime_classifier",
]
