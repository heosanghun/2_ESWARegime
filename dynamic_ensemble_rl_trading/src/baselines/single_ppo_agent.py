"""
Single PPO Agent baseline implementation.

This baseline uses a single PPO agent without regime classification
or ensemble mechanisms. It uses the same policy for all market conditions.
"""

import numpy as np
from typing import Dict, Optional
import logging

from ..agents.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


class SinglePPOAgent:
    """
    Single PPO agent baseline for comparison.
    
    This baseline removes:
    - Regime classification
    - Ensemble mechanisms
    - Dynamic weighting
    
    Uses a single PPO agent trained on all market conditions.
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        gamma: float = 0.99,
        seed: int = 42
    ):
        """
        Initialize single PPO agent.
        
        Parameters
        ----------
        env : gym.Env
            Trading environment (can be any regime-specific env).
        learning_rate : float, default=3e-4
            Learning rate for PPO.
        batch_size : int, default=64
            Batch size for PPO.
        gamma : float, default=0.99
            Discount factor.
        seed : int, default=42
            Random seed.
        """
        self.agent = PPOAgent(
            env=env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            seed=seed
        )
        logger.info("Initialized Single PPO Agent baseline")
    
    def train(self, total_timesteps: int, callback=None) -> None:
        """
        Train the single PPO agent.
        
        Parameters
        ----------
        total_timesteps : int
            Total training timesteps.
        callback : callable, optional
            Training callback.
        """
        logger.info(f"Training Single PPO Agent for {total_timesteps} timesteps")
        self.agent.train(total_timesteps, callback=callback)
    
    def predict(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Predict action from state.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        deterministic : bool, default=False
            Whether to use deterministic policy.
        
        Returns
        -------
        int
            Action index.
        """
        return self.agent.predict(state, deterministic=deterministic)
    
    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        
        Returns
        -------
        np.ndarray
            Action probability distribution.
        """
        return self.agent.predict_proba(state)
    
    def save(self, path: str) -> None:
        """Save the agent."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent."""
        self.agent.load(path)
