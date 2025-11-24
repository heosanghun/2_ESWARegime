"""
Unit tests for ensemble modules.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.ensemble.weighting import DynamicWeightCalculator
from src.ensemble.ensemble_trader import EnsembleTrader


def test_weight_calculator_initialization():
    """Test DynamicWeightCalculator initialization."""
    calculator = DynamicWeightCalculator(
        performance_window=30,
        temperature=10.0
    )
    
    assert calculator.performance_window == 30
    assert calculator.temperature == 10.0


def test_weight_calculator_agent_initialization():
    """Test agent initialization in weight calculator."""
    calculator = DynamicWeightCalculator()
    calculator.initialize_agents(num_agents=5)
    
    assert calculator.num_agents == 5
    assert len(calculator.agent_returns) == 5


def test_weight_calculator_update_returns():
    """Test updating returns for agents."""
    calculator = DynamicWeightCalculator()
    calculator.initialize_agents(num_agents=3)
    
    calculator.update_returns(0, 0.01)
    calculator.update_returns(1, 0.02)
    calculator.update_returns(2, -0.01)
    
    assert len(calculator.agent_returns[0]) == 1
    assert len(calculator.agent_returns[1]) == 1
    assert len(calculator.agent_returns[2]) == 1


def test_weight_calculation():
    """Test dynamic weight calculation."""
    calculator = DynamicWeightCalculator(temperature=10.0)
    calculator.initialize_agents(num_agents=3)
    
    # Add some returns
    for i in range(10):
        calculator.update_returns(0, 0.01 + np.random.randn() * 0.001)
        calculator.update_returns(1, 0.02 + np.random.randn() * 0.001)
        calculator.update_returns(2, -0.01 + np.random.randn() * 0.001)
    
    weights = calculator.calculate_weights()
    
    assert len(weights) == 3
    assert abs(np.sum(weights) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)
    assert all(w <= 1 for w in weights)
    # Agent 1 should have higher weight (better performance)
    assert weights[1] >= weights[2]


def test_ensemble_trader_initialization():
    """Test EnsembleTrader initialization."""
    weight_calc = DynamicWeightCalculator()
    trader = EnsembleTrader(weight_calc)
    
    assert trader.weight_calculator == weight_calc


