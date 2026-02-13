"""
Model 3: Ensemble Mechanism Removed

This ablation model removes the ensemble mechanism and uses only the
best-performing agent from each regime pool (selected on validation set).
Expected Sharpe Ratio: ~1.41
"""

import numpy as np
from typing import Dict, Optional
import logging

from ..agents.pool import PPOAgentPool
from ..agents.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


class NoEnsemble:
    """
    Ablation Model 3: Ensemble Mechanism Removed.
    
    This model:
    - Keeps regime classification
    - Removes ensemble mechanism
    - Uses only the best-performing agent from validation set
    """
    
    def __init__(
        self,
        active_pool: PPOAgentPool,
        best_agent_index: Optional[int] = None
    ):
        """
        Initialize Model 3.
        
        Parameters
        ----------
        active_pool : PPOAgentPool
            Active agent pool for current regime.
        best_agent_index : int, optional
            Index of best agent. If None, will select based on validation performance.
        """
        self.active_pool = active_pool
        
        if best_agent_index is None:
            # Select best agent based on validation performance
            # For now, use agent 0 as default (should be selected based on validation)
            self.best_agent_index = 0
            logger.warning("best_agent_index not provided. Using agent 0. Should select based on validation.")
        else:
            self.best_agent_index = best_agent_index
        
        self.best_agent = active_pool.agents[self.best_agent_index]
        logger.info(f"Initialized Model 3: Ensemble Removed (using agent {self.best_agent_index})")
    
    def select_best_agent(self, validation_states: np.ndarray, validation_returns: np.ndarray) -> None:
        """
        Select best agent based on validation performance.
        
        Parameters
        ----------
        validation_states : np.ndarray
            Validation states.
        validation_returns : np.ndarray
            Validation returns for each agent.
        """
        # Calculate Sharpe ratio for each agent
        best_sharpe = -np.inf
        best_index = 0
        
        for i, agent_returns in enumerate(validation_returns):
            if len(agent_returns) > 0:
                mean_return = np.mean(agent_returns)
                std_return = np.std(agent_returns)
                if std_return > 0:
                    sharpe = mean_return / std_return
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_index = i
        
        self.best_agent_index = best_index
        self.best_agent = self.active_pool.agents[self.best_agent_index]
        logger.info(f"Selected best agent: {self.best_agent_index} (Sharpe: {best_sharpe:.4f})")
    
    def get_action(self, state: np.ndarray) -> Dict:
        """
        Get action from best agent only (no ensemble).
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        
        Returns
        -------
        dict
            Action result from best agent.
        """
        proba = self.best_agent.predict_proba(state)
        action = np.argmax(proba)
        
        return {
            'action': int(action),
            'probabilities': proba,
            'agent_index': self.best_agent_index
        }
    
    def reset(self) -> None:
        """Reset model state."""
        pass
