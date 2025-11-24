"""
Dynamic weight allocation for ensemble agents.

This module implements the dynamic weighting mechanism based on
rolling 30-day Sharpe Ratio using Softmax with temperature T=10.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DynamicWeightCalculator:
    """
    Calculates dynamic weights for ensemble agents.
    
    Implements Eq. 6 from the paper:
    w_{i,t} = exp(SR_{i,30}/T) / sum(exp(SR_{j,30}/T))
    
    where SR_{i,30} is the rolling 30-day Sharpe Ratio for agent i,
    and T is the temperature parameter (default: 10).
    """
    
    def __init__(
        self,
        performance_window: int = 30,
        temperature: float = 10.0,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the dynamic weight calculator.
        
        Parameters
        ----------
        performance_window : int, default=30
            Rolling window size in days for Sharpe Ratio calculation.
        temperature : float, default=10.0
            Temperature parameter T for Softmax.
        risk_free_rate : float, default=0.0
            Risk-free rate for Sharpe Ratio calculation.
        """
        self.performance_window = performance_window
        self.temperature = temperature
        self.risk_free_rate = risk_free_rate
        
        # Track returns for each agent
        self.agent_returns: List[deque] = []
        self.num_agents = 0
    
    def initialize_agents(self, num_agents: int) -> None:
        """
        Initialize tracking for multiple agents.
        
        Parameters
        ----------
        num_agents : int
            Number of agents to track.
        """
        self.num_agents = num_agents
        self.agent_returns = [deque(maxlen=self.performance_window) for _ in range(num_agents)]
        logger.info(f"Initialized weight calculator for {num_agents} agents")
    
    def update_returns(
        self,
        agent_index: int,
        return_value: float
    ) -> None:
        """
        Update returns for a specific agent.
        
        Parameters
        ----------
        agent_index : int
            Index of the agent (0 to num_agents-1).
        return_value : float
            Return value for the current period.
        """
        if agent_index >= self.num_agents:
            raise IndexError(f"Agent index {agent_index} out of range")
        
        self.agent_returns[agent_index].append(return_value)
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Calculate Sharpe Ratio from returns.
        
        Sharpe Ratio = (R_p - R_f) / sigma_p
        
        where:
        - R_p: Mean return
        - R_f: Risk-free rate
        - sigma_p: Standard deviation of returns
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns.
        
        Returns
        -------
        float
            Sharpe Ratio.
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            if mean_return > self.risk_free_rate:
                return 10.0  # Arbitrary high value
            else:
                return 0.0
        
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
        
        return sharpe_ratio
    
    def calculate_weights(self) -> np.ndarray:
        """
        Calculate dynamic weights using Softmax with temperature.
        
        Implements Eq. 6:
        w_{i,t} = exp(SR_{i,30}/T) / sum(exp(SR_{j,30}/T))
        
        Returns
        -------
        np.ndarray
            Array of weights for each agent, shape (num_agents,).
        """
        if self.num_agents == 0:
            raise ValueError("No agents initialized. Call initialize_agents() first.")
        
        sharpe_ratios = []
        
        for i in range(self.num_agents):
            returns = np.array(list(self.agent_returns[i]))
            
            if len(returns) < 2:
                # Insufficient data: use default weight
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = self.calculate_sharpe_ratio(returns)
            
            sharpe_ratios.append(sharpe_ratio)
        
        sharpe_ratios = np.array(sharpe_ratios)
        
        # Apply Softmax with temperature
        # w_i = exp(SR_i / T) / sum(exp(SR_j / T))
        exp_scores = np.exp(sharpe_ratios / self.temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def reset(self) -> None:
        """Reset all agent returns."""
        for returns_deque in self.agent_returns:
            returns_deque.clear()
        logger.info("Weight calculator reset")

