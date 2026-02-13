"""
Simple Ensemble baseline implementation.

This baseline uses ensemble of agents with equal weighting (no dynamic weighting).
Regime classification is still used to select the active pool.
"""

import numpy as np
from typing import Dict, Optional
import logging

from ..agents.pool import PPOAgentPool
from ..ensemble.ensemble_trader import EnsembleTrader

logger = logging.getLogger(__name__)


class SimpleEnsemble:
    """
    Simple ensemble baseline with equal weighting.
    
    This baseline:
    - Uses regime classification (like proposed method)
    - Uses ensemble of agents per regime
    - BUT uses equal weighting instead of dynamic weighting
    - No performance-based weight adjustment
    """
    
    def __init__(
        self,
        active_pool: PPOAgentPool,
        num_agents: int = 5
    ):
        """
        Initialize simple ensemble.
        
        Parameters
        ----------
        active_pool : PPOAgentPool
            Active agent pool for current regime.
        num_agents : int, default=5
            Number of agents in the pool.
        """
        self.active_pool = active_pool
        self.num_agents = num_agents
        
        # Equal weights: w_i = 1/N
        self.equal_weights = np.ones(num_agents) / num_agents
        
        logger.info(f"Initialized Simple Ensemble with {num_agents} agents (equal weighting)")
    
    def get_ensemble_action(
        self,
        state: np.ndarray
    ) -> Dict:
        """
        Get ensemble action with equal weighting.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector S_t.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'action': Final ensemble action
            - 'probabilities': Ensemble probability distribution
            - 'weights': Equal weights (1/N for each agent)
            - 'individual_probs': Individual agent probabilities
        """
        # Get actions and probabilities from all agents in pool
        pool_output = self.active_pool.get_pool_actions(state, return_probs=True)
        
        individual_probs = pool_output['probabilities']
        
        # Use equal weights: w_i = 1/N
        weights = self.equal_weights.copy()
        
        # Aggregate policies: pi_ensemble = sum(w_i * pi_i)
        ensemble_probs = np.zeros_like(individual_probs[0])
        
        for agent_probs, weight in zip(individual_probs, weights):
            ensemble_probs += weight * agent_probs
        
        # Normalize to ensure valid probability distribution
        ensemble_probs = ensemble_probs / (np.sum(ensemble_probs) + 1e-10)
        
        # Select action: a_t = argmax(pi_ensemble)
        final_action = np.argmax(ensemble_probs)
        
        return {
            'action': int(final_action),
            'probabilities': ensemble_probs,
            'weights': weights,
            'individual_probs': individual_probs
        }
    
    def reset(self) -> None:
        """Reset ensemble state."""
        # Nothing to reset for simple ensemble
        pass
