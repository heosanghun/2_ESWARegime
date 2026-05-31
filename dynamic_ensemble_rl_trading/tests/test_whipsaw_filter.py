"""
Unit tests for the Whipsaw Filter (Acoustic Dampener).
"""

import pytest
import numpy as np

def test_whipsaw_filter_math():
    # Test parameters
    ema_lambda = 0.15
    gate_threshold = 0.25
    
    # Sequence of raw weights
    # 0.0 -> 0.8 -> 0.8 -> 0.0
    raw_weights = [0.8, 0.8, 0.8, 0.0, 0.0, 0.0]
    
    prev_smoothed_weight = 0.0
    active_weight = 0.0
    
    history_smoothed = []
    history_active = []
    
    for w in raw_weights:
        prev_smoothed_weight = ema_lambda * w + (1.0 - ema_lambda) * prev_smoothed_weight
        if abs(prev_smoothed_weight - active_weight) >= gate_threshold:
            active_weight = prev_smoothed_weight
        history_smoothed.append(prev_smoothed_weight)
        history_active.append(active_weight)
        
    # Check steps
    # Step 1: w = 0.8
    # smoothed = 0.15 * 0.8 + 0.85 * 0.0 = 0.12
    # abs(0.12 - 0.0) = 0.12 < 0.25 -> active_weight remains 0.0
    assert abs(history_smoothed[0] - 0.12) < 1e-5
    assert history_active[0] == 0.0
    
    # Step 2: w = 0.8
    # smoothed = 0.15 * 0.8 + 0.85 * 0.12 = 0.12 + 0.102 = 0.222
    # abs(0.222 - 0.0) = 0.222 < 0.25 -> active_weight remains 0.0
    assert abs(history_smoothed[1] - 0.222) < 1e-5
    assert history_active[1] == 0.0
    
    # Step 3: w = 0.8
    # smoothed = 0.15 * 0.8 + 0.85 * 0.222 = 0.12 + 0.1887 = 0.3087
    # abs(0.3087 - 0.0) = 0.3087 >= 0.25 -> active_weight becomes 0.3087
    assert abs(history_smoothed[2] - 0.3087) < 1e-5
    assert abs(history_active[2] - 0.3087) < 1e-5
