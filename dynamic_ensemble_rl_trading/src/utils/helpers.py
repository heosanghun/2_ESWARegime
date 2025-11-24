"""
Helper utility functions.
"""

import numpy as np
import pandas as pd
from typing import Union


def ensure_numpy_array(data: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """
    Convert data to numpy array if needed.
    
    Parameters
    ----------
    data : array-like
        Input data.
    
    Returns
    -------
    np.ndarray
        Numpy array.
    """
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    """
    Safe division avoiding division by zero.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array.
    denominator : np.ndarray
        Denominator array.
    default : float, default=0.0
        Default value when denominator is zero.
    
    Returns
    -------
    np.ndarray
        Result of division.
    """
    result = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, default, dtype=float),
        where=(denominator != 0)
    )
    return result

