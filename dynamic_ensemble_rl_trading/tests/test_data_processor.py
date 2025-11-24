"""
Unit tests for data processing modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_processor import MarketDataHandler
from src.data.feature_extractor import TechnicalFeatureExtractor


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2021-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 101 + np.random.randn(100).cumsum(),
        'low': 99 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low, high >= open, high >= close, etc.
    data['high'] = data[['open', 'high', 'close']].max(axis=1) + 1
    data['low'] = data[['open', 'low', 'close']].min(axis=1) - 1
    
    return data


def test_market_data_handler(sample_ohlcv_data, tmp_path):
    """Test MarketDataHandler."""
    # Save sample data
    data_path = tmp_path / 'test_data.csv'
    sample_ohlcv_data.to_csv(data_path)
    
    # Test loading
    handler = MarketDataHandler(str(data_path))
    loaded_data = handler.load_data()
    
    assert len(loaded_data) > 0
    assert 'open' in loaded_data.columns
    assert 'close' in loaded_data.columns


def test_technical_feature_extractor(sample_ohlcv_data):
    """Test TechnicalFeatureExtractor."""
    extractor = TechnicalFeatureExtractor(normalization_window=30)
    features = extractor.extract_features(sample_ohlcv_data)
    
    assert len(features) > 0
    assert len(features.columns) > 0
    assert not features.isna().all().all()


def test_walk_forward_split(sample_ohlcv_data, tmp_path):
    """Test Walk-Forward splitting."""
    data_path = tmp_path / 'test_data.csv'
    sample_ohlcv_data.to_csv(data_path)
    
    handler = MarketDataHandler(str(data_path))
    handler.load_data()
    
    train, val, test = handler.get_walk_forward_splits(
        train_start='2021-01-01',
        train_end='2021-01-20',
        test_start='2021-01-21',
        test_end='2021-01-25',
        validation_ratio=0.2
    )
    
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

