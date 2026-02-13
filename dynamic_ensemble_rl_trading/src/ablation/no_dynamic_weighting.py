"""
Model 1: Dynamic Weighting Removed

This ablation model removes the dynamic weighting mechanism and uses
equal weighting instead. Regime classification and ensemble are still used.
Expected Sharpe Ratio: ~1.58
"""

import numpy as np
from typing import Dict
import logging

from ..baselines.simple_ensemble import SimpleEnsemble
from ..agents.pool import PPOAgentPool

logger = logging.getLogger(__name__)


class NoDynamicWeighting:
    """
    Ablation Model 1: Dynamic Weighting Removed.
    
    This model:
    - Keeps regime classification
    - Keeps ensemble mechanism
    - Removes dynamic weighting (uses equal weights)
    """
    
    def __init__(self, active_pool: PPOAgentPool):
        """
        Initialize Model 1.
        
        Parameters
        ----------
        active_pool : PPOAgentPool
            Active agent pool for current regime.
        """
        self.ensemble = SimpleEnsemble(active_pool, num_agents=len(active_pool.agents))
        logger.info("Initialized Model 1: Dynamic Weighting Removed")
    
    def get_ensemble_action(self, state: np.ndarray) -> Dict:
        """
        Get ensemble action with equal weighting.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        
        Returns
        -------
        dict
            Ensemble action result.
        """
        return self.ensemble.get_ensemble_action(state)
    
    def reset(self) -> None:
        """Reset model state."""
        self.ensemble.reset()
