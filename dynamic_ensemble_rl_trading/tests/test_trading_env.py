"""
Unit tests for trading environment.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.env.trading_env import MultiRegimeTradingEnv
from src.env.rewards import RegimeRewardCalculator


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data."""
    dates = pd.date_range('2021-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 101 + np.random.randn(100).cumsum(),
        'low': 99 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1) + 1
    data['low'] = data[['open', 'low', 'close']].min(axis=1) - 1
    
    return data


@pytest.fixture
def sample_state_data():
    """Create sample state data."""
    dates = pd.date_range('2021-01-01', periods=100, freq='H')
    state_dim = 100
    states = np.random.randn(100, state_dim)
    return pd.DataFrame(states, index=dates)


def test_trading_env_initialization(sample_ohlcv_data, sample_state_data):
    """Test trading environment initialization."""
    env = MultiRegimeTradingEnv(
        ohlcv_data=sample_ohlcv_data,
        state_data=sample_state_data,
        regime_type='Bull',
        initial_balance=10000.0
    )
    
    assert env.initial_balance == 10000.0
    assert env.regime_type == 'Bull'
    assert env.action_space.n == 5
    assert env.observation_space.shape[0] == sample_state_data.shape[1]


def test_trading_env_reset(sample_ohlcv_data, sample_state_data):
    """Test environment reset."""
    env = MultiRegimeTradingEnv(
        ohlcv_data=sample_ohlcv_data,
        state_data=sample_state_data,
        regime_type='Bull'
    )
    
    obs, info = env.reset()
    
    assert obs is not None
    assert isinstance(info, dict)
    assert env.current_step == 0
    assert env.portfolio_value == env.initial_balance


def test_reward_calculator_bull(sample_ohlcv_data, sample_state_data):
    """Test Bull market reward calculation."""
    calculator = RegimeRewardCalculator()
    
    reward = calculator.calculate_bull_reward(
        portfolio_value_before=10000.0,
        portfolio_value_after=10500.0
    )
    
    assert reward > 0
    assert abs(reward - 0.05) < 0.01  # 5% return


def test_reward_calculator_bear(sample_ohlcv_data, sample_state_data):
    """Test Bear market reward calculation."""
    calculator = RegimeRewardCalculator()
    
    reward = calculator.calculate_bear_reward(
        portfolio_value_before=10000.0,
        portfolio_value_after=9800.0,
        transaction_cost_incurred=10.0
    )
    
    assert isinstance(reward, float)


def test_reward_calculator_sideways(sample_ohlcv_data, sample_state_data):
    """Test Sideways market reward calculation."""
    calculator = RegimeRewardCalculator()
    
    reward = calculator.calculate_sideways_reward(
        portfolio_value_before=10000.0,
        portfolio_value_after=10050.0,
        transaction_cost_incurred=5.0
    )
    
    assert isinstance(reward, float)
    # Sideways should have penalty, so reward should be lower than Bull


