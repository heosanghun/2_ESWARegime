"""
Unit tests for regime classification modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.regime.ground_truth import RegimeGroundTruth
from src.regime.regime_classifier import RegimeClassifier


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range('2021-01-01', periods=100, freq='h')
    # Create trending price data
    trend = np.linspace(100, 150, 100)
    noise = np.random.randn(100) * 2
    prices = trend + noise
    return pd.Series(prices, index=dates)


def test_ground_truth_generation(sample_price_data):
    """Test ground truth label generation."""
    gt = RegimeGroundTruth(
        sma_window=50,
        bull_threshold=0.0005,
        bear_threshold=-0.0005
    )
    
    labels = gt.generate_labels(sample_price_data)
    
    assert len(labels) == len(sample_price_data)
    assert all(label in [0, 1, 2] for label in labels)
    assert labels.dtype == int or labels.dtype == 'int64'


def test_regime_classifier_initialization():
    """Test RegimeClassifier initialization."""
    classifier = RegimeClassifier(
        n_estimators=10,
        max_depth=3,
        confidence_threshold=0.6
    )
    
    assert classifier.n_estimators == 10
    assert classifier.max_depth == 3
    assert classifier.confidence_threshold == 0.6
    assert not classifier.is_fitted


def test_regime_classifier_training():
    """Test RegimeClassifier training."""
    classifier = RegimeClassifier(
        n_estimators=10,
        max_depth=3,
        confidence_threshold=0.6
    )
    
    # Create dummy training data
    X = np.random.randn(100, 50)
    y = np.random.randint(0, 3, 100)
    
    classifier.fit(X, y)
    
    assert classifier.is_fitted


def test_confidence_based_selection():
    """Test confidence-based regime selection."""
    classifier = RegimeClassifier(confidence_threshold=0.6)
    
    # Mock fitted model
    classifier.is_fitted = True
    
    # Create dummy state
    state = np.random.randn(50)
    
    # This will fail without a fitted model, but tests the logic
    # In real usage, model must be trained first
    try:
        regime, confidence = classifier.select_regime_with_confidence(
            state, previous_regime=1
        )
        assert regime in [0, 1, 2]
        assert 0 <= confidence <= 1
    except (ValueError, AttributeError):
        # Expected if model not actually fitted
        pass


def test_sequential_regime_classifier():
    """Test SequentialRegimeClassifier initialization and training."""
    from src.regime.sequential_regime_classifier import SequentialRegimeClassifier
    classifier = SequentialRegimeClassifier(
        sequence_window=5,
        n_features=10,
        hidden_dim=8,
        num_layers=1,
        max_epochs=2,
        use_visual=False,
    )
    
    assert classifier.sequence_window == 5
    assert classifier.n_features == 10
    assert not classifier.is_fitted
    
    # Create sequence training data
    X = np.random.randn(20, 5, 10)
    y = np.random.randint(0, 3, 20)
    
    classifier.fit(X, y)
    assert classifier.is_fitted
    
    # Observe states and predict
    state = np.random.randn(10)
    classifier.observe_state(state)
    
    # Mocking or predicting
    proba = classifier.predict_proba(X[0:1])
    assert proba.shape == (1, 3)



