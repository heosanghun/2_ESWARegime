"""
Model 4: Regime Classification Removed

This ablation model removes regime classification completely and uses
all 15 agents (5 from each regime pool) in a single ensemble with dynamic weighting.
Expected Sharpe Ratio: ~1.35
"""

import numpy as np
from typing import Dict, List
import logging

from ..agents.agent_manager import HierarchicalAgentManager
from ..ensemble.weighting import DynamicWeightCalculator
from ..ensemble.ensemble_trader import EnsembleTrader

logger = logging.getLogger(__name__)


class NoRegimeClassification:
    """
    Ablation Model 4: Regime Classification Removed.
    
    This model:
    - Removes regime classification completely
    - Uses all 15 agents (5 from each regime pool) in single ensemble
    - Keeps dynamic weighting across all 15 agents
    """
    
    def __init__(
        self,
        agent_manager: HierarchicalAgentManager,
        performance_window: int = 30,
        temperature: float = 10.0
    ):
        """
        Initialize Model 4.
        
        Parameters
        ----------
        agent_manager : HierarchicalAgentManager
            Agent manager with all three pools.
        performance_window : int, default=30
            Window size for performance tracking.
        temperature : float, default=10.0
            Temperature parameter for Softmax.
        """
        self.agent_manager = agent_manager
        
        # Collect all 15 agents from all pools
        self.all_agents = []
        self.all_agents.extend(agent_manager.bull_pool.agents)
        self.all_agents.extend(agent_manager.bear_pool.agents)
        self.all_agents.extend(agent_manager.sideways_pool.agents)
        
        total_agents = len(self.all_agents)
        logger.info(f"Collected {total_agents} agents from all pools")
        
        # Initialize dynamic weight calculator for all agents
        self.weight_calculator = DynamicWeightCalculator(
            performance_window=performance_window,
            temperature=temperature
        )
        self.weight_calculator.initialize_agents(total_agents)
        
        logger.info("Initialized Model 4: Regime Classification Removed")
    
    def get_ensemble_action(self, state: np.ndarray) -> Dict:
        """
        Get ensemble action from all 15 agents with dynamic weighting.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        
        Returns
        -------
        dict
            Ensemble action result.
        """
        # Get probabilities from all agents
        individual_probs = []
        for agent in self.all_agents:
            proba = agent.predict_proba(state)
            individual_probs.append(proba)
        
        individual_probs = np.array(individual_probs)
        
        # Calculate dynamic weights for all agents
        weights = self.weight_calculator.calculate_weights()
        
        # Ensure weights match number of agents
        if len(weights) != len(self.all_agents):
            self.weight_calculator.initialize_agents(len(self.all_agents))
            weights = self.weight_calculator.calculate_weights()
        
        # Aggregate policies: pi_ensemble = sum(w_i * pi_i)
        ensemble_probs = np.zeros_like(individual_probs[0])
        
        for agent_probs, weight in zip(individual_probs, weights):
            ensemble_probs += weight * agent_probs
        
        # Normalize
        ensemble_probs = ensemble_probs / (np.sum(ensemble_probs) + 1e-10)
        
        # Select action: a_t = argmax(pi_ensemble)
        final_action = np.argmax(ensemble_probs)
        
        return {
            'action': int(final_action),
            'probabilities': ensemble_probs,
            'weights': weights,
            'individual_probs': individual_probs
        }
    
    def update_agent_performance(
        self,
        agent_index: int,
        portfolio_value: float
    ) -> None:
        """
        Update performance tracking for an agent.
        
        Parameters
        ----------
        agent_index : int
            Index of agent (0-14).
        portfolio_value : float
            Current portfolio value.
        """
        if agent_index < len(self.all_agents):
            prev_value = (
                self.weight_calculator.agent_portfolio_values[agent_index][-1]
                if agent_index in self.weight_calculator.agent_portfolio_values
                and len(self.weight_calculator.agent_portfolio_values[agent_index]) > 0
                else None
            )
            
            self.weight_calculator.update_portfolio_value(agent_index, portfolio_value)
            
            if prev_value is not None and prev_value > 0:
                return_value = (portfolio_value - prev_value) / prev_value
                self.weight_calculator.update_returns(agent_index, return_value)
    
    def reset(self) -> None:
        """Reset model state."""
        self.weight_calculator.reset()
