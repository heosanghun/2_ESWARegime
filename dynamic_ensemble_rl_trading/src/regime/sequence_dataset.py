"""Sliding-window dataset builder for sequential regime classifiers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def build_sequence_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Build (N, window, F) sequences aligned to label at the final timestep.

    Returns
    -------
    X : (n_samples, window, n_features)
    y : (n_samples,)
    index : timestamps for each sample (end of window)
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    idx = features.index.intersection(labels.index)
    feat = features.loc[idx].astype(np.float64)
    y_s = labels.loc[idx].astype(int)

    n = len(feat)
    if n < window:
        raise ValueError(f"Need at least {window} rows, got {n}")

    feat_vals = feat.values
    y_vals = y_s.values
    out_index = idx[window - 1 :]

    n_samples = n - window + 1
    n_features = feat_vals.shape[1]
    X = np.zeros((n_samples, window, n_features), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        X[i] = feat_vals[i : i + window]
        y[i] = int(y_vals[i + window - 1])

    return X, y, out_index
