"""
Baseline trading methods for comparison.

This module implements baseline methods for Table 2 comparison:
- Buy & Hold
- Single PPO Agent
- XGBoost Trader
- CNN Trader
- Simple Ensemble
"""

from .single_ppo_agent import SinglePPOAgent
from .xgboost_trader import XGBoostTrader
from .cnn_trader import CNNTrader
from .simple_ensemble import SimpleEnsemble

__all__ = [
    'SinglePPOAgent',
    'XGBoostTrader',
    'CNNTrader',
    'SimpleEnsemble'
]
