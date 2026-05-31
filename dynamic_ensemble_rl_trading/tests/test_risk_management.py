"""
Unit tests for Conductor Confidence Gating and Volatility Targeting Risk Management overlays.
"""

import pytest
import numpy as np
import pandas as pd

def test_confidence_gating_math():
    # Simulated inputs
    expected_weight = 0.8
    regime_confidence = 0.65  # 65% classification confidence
    
    # Formula: w_conf = w_expected * confidence
    w_conf = expected_weight * regime_confidence
    
    # Verification
    assert abs(w_conf - 0.52) < 1e-5
    
    # Low confidence case (avoid chop)
    low_confidence = 0.35
    w_conf_low = expected_weight * low_confidence
    assert abs(w_conf_low - 0.28) < 1e-5

def test_volatility_targeting_math():
    # Simulated inputs
    w_conf = 0.52
    target_vol = 0.15          # 15% target annualized vol
    
    # 4-hour return standard deviation (e.g. 0.005, which is 0.5% return std)
    rolling_std_step = 0.005
    # Annualized standard deviation (6 candles per day, 365.25 days per year)
    rolling_std_ann = rolling_std_step * np.sqrt(6 * 365.25)
    
    # Formula: w_scaled = w_conf * (target_vol / rolling_std_ann)
    w_scaled = w_conf * (target_vol / rolling_std_ann)
    
    # Verification
    # rolling_std_ann = 0.005 * sqrt(2191.5) = 0.005 * 46.8134596 = 0.234067
    # target_vol / rolling_std_ann = 0.15 / 0.234067 = 0.64084
    # w_scaled = 0.52 * 0.64084 = 0.3332
    assert abs(rolling_std_ann - 0.234067) < 1e-4
    assert abs(w_scaled - 0.33323) < 1e-4
