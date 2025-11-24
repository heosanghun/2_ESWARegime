"""
Ensemble decision making module.

This module implements the Ensemble Decision Layer that aggregates
policies from multiple agents using dynamic weighting.
"""

import numpy as np
from typing import Dict, Optional
import logging

from .weighting import DynamicWeightCalculator
from ..agents.pool import PPOAgentPool

logger = logging.getLogger(__name__)


class EnsembleTrader:
    """
    Ensemble trader that aggregates policies from agent pool.
    
    Implements the Ensemble Decision Layer:
    - Calculates dynamic weights based on agent performance
    - Aggregates policies using weighted sum (Eq. 7)
    - Selects final action using argmax (Eq. 8)
    """
    
    def __init__(
        self,
        weight_calculator: DynamicWeightCalculator,
        performance_window: int = 30,
        temperature: float = 10.0
    ):
        """
        Initialize the ensemble trader.
        
        Parameters
        ----------
        weight_calculator : DynamicWeightCalculator
            Calculator for dynamic weights.
        performance_window : int, default=30
            Window size for performance tracking.
        temperature : float, default=10.0
            Temperature parameter for Softmax.
        """
        self.weight_calculator = weight_calculator
        self.performance_window = performance_window
        self.temperature = temperature
        
        # Track portfolio values for each agent
        self.agent_portfolio_values: Dict[int, list] = {}
        self.agent_returns: Dict[int, list] = {}
    
    def initialize_agents(self, num_agents: int) -> None:
        """
        Initialize tracking for agents.
        
        Parameters
        ----------
        num_agents : int
            Number of agents in the pool.
        """
        self.weight_calculator.initialize_agents(num_agents)
        
        for i in range(num_agents):
            self.agent_portfolio_values[i] = []
            self.agent_returns[i] = []
        
        logger.info(f"Initialized ensemble trader for {num_agents} agents")
    
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
            Index of the agent.
        portfolio_value : float
            Current portfolio value.
        """
        if agent_index not in self.agent_portfolio_values:
            self.agent_portfolio_values[agent_index] = []
            self.agent_returns[agent_index] = []
        
        prev_value = (
            self.agent_portfolio_values[agent_index][-1]
            if self.agent_portfolio_values[agent_index]
            else None
        )
        
        self.agent_portfolio_values[agent_index].append(portfolio_value)
        
        # Calculate return
        if prev_value is not None and prev_value > 0:
            return_value = (portfolio_value - prev_value) / prev_value
            self.agent_returns[agent_index].append(return_value)
            self.weight_calculator.update_returns(agent_index, return_value)
    
    def get_ensemble_action(
        self,
        state: np.ndarray,
        active_pool: PPOAgentPool
    ) -> Dict:
        """
        Get ensemble action from active agent pool.
        
        Implements Eq. 7 and Eq. 8:
        - Get probability distributions from all agents
        - Calculate dynamic weights
        - Aggregate policies: pi_ensemble = sum(w_i * pi_i)
        - Select action: a_t = argmax(pi_ensemble)
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector S_t.
        active_pool : PPOAgentPool
            Active agent pool for current regime.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'action': Final ensemble action
            - 'probabilities': Ensemble probability distribution
            - 'weights': Dynamic weights for each agent
            - 'individual_probs': Individual agent probabilities
        """
        # Get actions and probabilities from all agents in pool
        pool_output = active_pool.get_pool_actions(state, return_probs=True)
        
        individual_probs = pool_output['probabilities']
        num_agents = len(individual_probs)
        
        # Calculate dynamic weights
        weights = self.weight_calculator.calculate_weights()
        
        # Ensure weights match number of agents
        if len(weights) != num_agents:
            # Initialize if needed
            self.weight_calculator.initialize_agents(num_agents)
            weights = self.weight_calculator.calculate_weights()
        
        # Aggregate policies: pi_ensemble = sum(w_i * pi_i) (Eq. 7)
        ensemble_probs = np.zeros_like(individual_probs[0])
        
        for i, (agent_probs, weight) in enumerate(zip(individual_probs, weights)):
            ensemble_probs += weight * agent_probs
        
        # Normalize to ensure valid probability distribution
        ensemble_probs = ensemble_probs / (np.sum(ensemble_probs) + 1e-10)
        
        # Select action: a_t = argmax(pi_ensemble) (Eq. 8)
        final_action = np.argmax(ensemble_probs)
        
        return {
            'action': int(final_action),
            'probabilities': ensemble_probs,
            'weights': weights,
            'individual_probs': individual_probs
        }
    
    def reset(self) -> None:
        """Reset performance tracking."""
        self.agent_portfolio_values = {}
        self.agent_returns = {}
        self.weight_calculator.reset()
        logger.info("Ensemble trader reset")

