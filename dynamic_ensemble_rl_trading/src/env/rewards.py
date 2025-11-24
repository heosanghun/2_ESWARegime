"""
Regime-specific reward functions.

This module implements the three reward functions for different market regimes:
- Bull: Maximize profit (simple return)
- Bear: Maximize Sortino Ratio and minimize drawdown
- Sideways: Penalty for frequent trading (whipsaw avoidance)
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RegimeRewardCalculator:
    """
    Calculates regime-specific rewards for RL agents.
    
    Each market regime has a specialized reward function designed to
    elicit optimal behavior for that specific market condition.
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.0005,  # 0.05%
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the reward calculator.
        
        Parameters
        ----------
        transaction_cost : float, default=0.0005
            Transaction cost as fraction (0.05%).
        risk_free_rate : float, default=0.0
            Risk-free rate for Sortino Ratio calculation.
        """
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        # Track portfolio history for Sortino Ratio calculation
        self.portfolio_returns: list = []
        self.portfolio_values: list = []
    
    def calculate_bull_reward(
        self,
        portfolio_value_before: float,
        portfolio_value_after: float,
        transaction_cost_incurred: float = 0.0
    ) -> float:
        """
        Calculate reward for Bull market regime.
        
        Reward: R_bull = (V_{t+1} - V_t) / V_t
        This directly incentivizes profit maximization in rising markets.
        Transaction costs are intentionally omitted to encourage
        momentum capture.
        
        Parameters
        ----------
        portfolio_value_before : float
            Portfolio value at time t (V_t).
        portfolio_value_after : float
            Portfolio value at time t+1 (V_{t+1}).
        transaction_cost_incurred : float, default=0.0
            Transaction cost incurred (not used in Bull reward).
        
        Returns
        -------
        float
            Bull market reward.
        """
        if portfolio_value_before <= 0:
            return 0.0
        
        # Simple return: R = (V_{t+1} - V_t) / V_t
        reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        
        return reward
    
    def calculate_bear_reward(
        self,
        portfolio_value_before: float,
        portfolio_value_after: float,
        transaction_cost_incurred: float
    ) -> float:
        """
        Calculate reward for Bear market regime.
        
        Reward: R_bear = SortinoRatio - C * TransactionCost
        
        The Sortino Ratio focuses on downside risk, which is critical
        for capital preservation during bear markets.
        
        Parameters
        ----------
        portfolio_value_before : float
            Portfolio value at time t.
        portfolio_value_after : float
            Portfolio value at time t+1.
        transaction_cost_incurred : float
            Transaction cost incurred.
        
        Returns
        -------
        float
            Bear market reward.
        """
        if portfolio_value_before <= 0:
            return 0.0
        
        # Calculate return
        portfolio_return = (
            (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        )
        
        # Update portfolio history
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_values.append(portfolio_value_after)
        
        # Calculate Sortino Ratio
        sortino_ratio = self._calculate_sortino_ratio()
        
        # Bear reward: Sortino Ratio - C * TransactionCost
        reward = sortino_ratio - self.transaction_cost * transaction_cost_incurred
        
        return reward
    
    def calculate_sideways_reward(
        self,
        portfolio_value_before: float,
        portfolio_value_after: float,
        transaction_cost_incurred: float
    ) -> float:
        """
        Calculate reward for Sideways market regime.
        
        Reward: R_sideways = R_bull - 5 * TransactionCost
        
        This heavily penalizes frequent trading to avoid whipsaw losses
        in directionless markets. The factor of 5 is a design choice to
        strongly discourage unnecessary trades.
        
        Parameters
        ----------
        portfolio_value_before : float
            Portfolio value at time t.
        portfolio_value_after : float
            Portfolio value at time t+1.
        transaction_cost_incurred : float
            Transaction cost incurred.
        
        Returns
        -------
        float
            Sideways market reward.
        """
        # Calculate Bull reward (base return)
        bull_reward = self.calculate_bull_reward(
            portfolio_value_before,
            portfolio_value_after,
            transaction_cost_incurred
        )
        
        # Sideways reward: R_bull - 5 * TransactionCost
        penalty = 5.0 * self.transaction_cost * transaction_cost_incurred
        reward = bull_reward - penalty
        
        return reward
    
    def _calculate_sortino_ratio(
        self,
        window: int = 30
    ) -> float:
        """
        Calculate Sortino Ratio from portfolio returns.
        
        Sortino Ratio = (R_p - R_f) / DR
        
        where:
        - R_p: Average portfolio return
        - R_f: Risk-free rate
        - DR: Downside deviation (standard deviation of negative returns)
        
        Parameters
        ----------
        window : int, default=30
            Rolling window size for calculation.
        
        Returns
        -------
        float
            Sortino Ratio.
        """
        if len(self.portfolio_returns) < 2:
            return 0.0
        
        # Use recent returns within window
        recent_returns = np.array(self.portfolio_returns[-window:])
        
        if len(recent_returns) == 0:
            return 0.0
        
        # Average return
        avg_return = np.mean(recent_returns)
        
        # Downside deviation: only consider negative returns
        negative_returns = recent_returns[recent_returns < 0]
        
        if len(negative_returns) == 0:
            # No negative returns: high Sortino Ratio
            if avg_return > self.risk_free_rate:
                return 10.0  # Arbitrary high value
            else:
                return 0.0
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            if avg_return > self.risk_free_rate:
                return 10.0
            else:
                return 0.0
        
        # Sortino Ratio
        sortino_ratio = (avg_return - self.risk_free_rate) / downside_deviation
        
        return sortino_ratio
    
    def reset(self) -> None:
        """Reset portfolio history for new episode."""
        self.portfolio_returns = []
        self.portfolio_values = []
    
    def calculate_reward(
        self,
        regime: str,
        portfolio_value_before: float,
        portfolio_value_after: float,
        transaction_cost_incurred: float = 0.0
    ) -> float:
        """
        Calculate reward based on market regime.
        
        Parameters
        ----------
        regime : str
            Market regime: 'Bull', 'Bear', or 'Sideways'.
        portfolio_value_before : float
            Portfolio value before action.
        portfolio_value_after : float
            Portfolio value after action.
        transaction_cost_incurred : float, default=0.0
            Transaction cost incurred.
        
        Returns
        -------
        float
            Reward value.
        """
        regime = regime.lower()
        
        if regime == 'bull':
            return self.calculate_bull_reward(
                portfolio_value_before,
                portfolio_value_after,
                transaction_cost_incurred
            )
        elif regime == 'bear':
            return self.calculate_bear_reward(
                portfolio_value_before,
                portfolio_value_after,
                transaction_cost_incurred
            )
        elif regime == 'sideways':
            return self.calculate_sideways_reward(
                portfolio_value_before,
                portfolio_value_after,
                transaction_cost_incurred
            )
        else:
            raise ValueError(
                f"Unknown regime: {regime}. Must be 'Bull', 'Bear', or 'Sideways'"
            )

