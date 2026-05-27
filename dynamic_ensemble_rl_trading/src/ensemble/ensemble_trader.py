"""
Ensemble decision making module.

Tracks per-agent performance so dynamic weighting operates correctly.
"""

import numpy as np
from typing import Dict, Optional, List
import logging

from .weighting import DynamicWeightCalculator
from ..agents.pool import PPOAgentPool

logger = logging.getLogger(__name__)


class EnsembleTrader:
    """
    Ensemble trader that aggregates policies from agent pool.
    
    Tracks virtual portfolios per agent so dynamic weighting can operate.
    """

    def __init__(
        self,
        weight_calculator: DynamicWeightCalculator,
        performance_window: int = 30,
        temperature: float = 10.0,
        initial_capital: float = 10000.0,
    ):
        self.weight_calculator = weight_calculator
        self.performance_window = performance_window
        self.temperature = temperature
        self.initial_capital = initial_capital
        
        # Virtual portfolio tracking per agent
        self.agent_portfolio_values: Dict[int, List[float]] = {}
        self.agent_returns: Dict[int, List[float]] = {}
        self.agent_actions: Dict[int, List[int]] = {}  # Record each agent's proposed action
    
    def initialize_agents(self, num_agents: int) -> None:
        """Initialize tracking for agents."""
        self.weight_calculator.initialize_agents(num_agents)
        
        for i in range(num_agents):
            self.agent_portfolio_values[i] = [self.initial_capital]
            self.agent_returns[i] = []
            self.agent_actions[i] = []
        
        logger.info(f"Initialized ensemble trader for {num_agents} agents")
    
    def update_agent_performance(
        self,
        agent_index: int,
        portfolio_value: float
    ) -> None:
        """Update performance tracking for an agent."""
        if agent_index not in self.agent_portfolio_values:
            self.agent_portfolio_values[agent_index] = [self.initial_capital]
            self.agent_returns[agent_index] = []
            self.agent_actions[agent_index] = []
        
        prev_value = (
            self.agent_portfolio_values[agent_index][-1]
            if len(self.agent_portfolio_values[agent_index]) > 0
            else self.initial_capital
        )
        
        self.agent_portfolio_values[agent_index].append(portfolio_value)
        
        # Calculate return
        if prev_value > 0:
            return_value = (portfolio_value - prev_value) / prev_value
            self.agent_returns[agent_index].append(return_value)
            self.weight_calculator.update_returns(agent_index, return_value)
    
    def update_all_agents_with_price_change(
        self,
        price_change: float,
        transaction_cost: float = 0.0007
    ) -> None:
        """
        Update every agent's virtual portfolio from a price move.

        Uses the previous step's action (the current step's action is not recorded yet).
        """
        # Action → weight mapping (Long-Short, matches paper Section 3.1).
        # If the surrounding environment is configured as long-only via
        # `allow_short=False`, the per-agent virtual PV will still be
        # conservative — these weights are only used for *tracking*
        # ensemble dispersion, not for actually placing trades.
        WEIGHT_MAP = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}

        for agent_idx in self.agent_portfolio_values:
            if len(self.agent_actions[agent_idx]) < 1:
                continue
            
            prev_pv = self.agent_portfolio_values[agent_idx][-1]
            if prev_pv <= 0:
                continue
            
            # Use the previous step's action (most recent recorded action)
            prev_action = self.agent_actions[agent_idx][-1]
            prev_weight = WEIGHT_MAP.get(prev_action, 0.5)
            
            # Prior-step weight (for transaction cost)
            prev_prev_weight = 0.0
            if len(self.agent_actions[agent_idx]) > 1:
                prev_prev_action = self.agent_actions[agent_idx][-2]
                prev_prev_weight = WEIGHT_MAP.get(prev_prev_action, 0.5)
            
            # Transaction cost from weight change in the previous step
            weight_change = abs(prev_weight - prev_prev_weight)
            txn_cost = transaction_cost * weight_change
            
            # Portfolio return = prev_weight * price_change - txn_cost
            portfolio_return = prev_weight * price_change - txn_cost
            
            new_pv = prev_pv * (1 + portfolio_return)
            new_pv = max(new_pv, 1.0)  # floor
            
            self.agent_portfolio_values[agent_idx].append(new_pv)
            
            # Record return
            if prev_pv > 0:
                return_value = (new_pv - prev_pv) / prev_pv
                self.agent_returns[agent_idx].append(return_value)
                self.weight_calculator.update_returns(agent_idx, return_value)
    
    def get_ensemble_action(
        self,
        state: np.ndarray,
        active_pool: PPOAgentPool
    ) -> Dict:
        """
        Get ensemble action from active pool.
        
        Record each agent's action for later performance tracking.
        """
        pool_output = active_pool.get_pool_actions(state, return_probs=True)
        
        individual_probs = pool_output['probabilities']
        individual_actions = pool_output['actions']
        num_agents = len(individual_probs)
        
        # Record per-agent actions
        for i, action in enumerate(individual_actions):
            if i not in self.agent_actions:
                self.agent_actions[i] = []
            self.agent_actions[i].append(int(action))
        
        # Compute dynamic weights
        weights = self.weight_calculator.calculate_weights()
        
        if len(weights) != num_agents:
            self.weight_calculator.initialize_agents(num_agents)
            weights = self.weight_calculator.calculate_weights()
        
        # Equal weights when insufficient history
        if all(w == weights[0] for w in weights) and len(self.agent_returns.get(0, [])) < 2:
            weights = np.ones(num_agents) / num_agents
        
        # Aggregate policies: pi_ensemble = sum(w_i * pi_i)
        ensemble_probs = np.zeros_like(individual_probs[0])
        
        for i, (agent_probs, weight) in enumerate(zip(individual_probs, weights)):
            ensemble_probs += weight * agent_probs
        
        ensemble_probs = ensemble_probs / (np.sum(ensemble_probs) + 1e-10)
        
        # Select action: a_t = argmax(pi_ensemble)
        final_action = np.argmax(ensemble_probs)
        
        return {
            'action': int(final_action),
            'probabilities': ensemble_probs,
            'weights': weights,
            'individual_probs': individual_probs,
            'individual_actions': individual_actions,
        }
    
    def reset(self) -> None:
        """Reset performance tracking."""
        self.agent_portfolio_values = {}
        self.agent_returns = {}
        self.agent_actions = {}
        self.weight_calculator.reset()
        logger.info("Ensemble trader reset")
