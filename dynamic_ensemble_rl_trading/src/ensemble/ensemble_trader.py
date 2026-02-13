"""
Ensemble decision making module.

각 에이전트의 성능을 추적하여 Dynamic Weighting이 제대로 작동하도록 수정.
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
    
    각 에이전트의 가상 포트폴리오를 추적하여 Dynamic Weighting이 작동하도록 개선.
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
        
        # 각 에이전트의 가상 포트폴리오 추적
        self.agent_portfolio_values: Dict[int, List[float]] = {}
        self.agent_returns: Dict[int, List[float]] = {}
        self.agent_actions: Dict[int, List[int]] = {}  # 각 에이전트가 제안한 액션 기록
    
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
        모든 에이전트의 가상 포트폴리오를 가격 변동으로 업데이트.
        
        이전 스텝의 액션으로 포트폴리오를 업데이트합니다.
        (현재 스텝의 액션은 아직 기록되지 않은 상태)
        """
        # Action → weight mapping (Long-only)
        WEIGHT_MAP = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
        
        for agent_idx in self.agent_portfolio_values:
            if len(self.agent_actions[agent_idx]) < 1:
                continue
            
            prev_pv = self.agent_portfolio_values[agent_idx][-1]
            if prev_pv <= 0:
                continue
            
            # 이전 스텝의 액션 사용 (마지막 액션)
            prev_action = self.agent_actions[agent_idx][-1]
            prev_weight = WEIGHT_MAP.get(prev_action, 0.5)
            
            # 이전 이전 스텝의 weight (거래 비용 계산용)
            prev_prev_weight = 0.0
            if len(self.agent_actions[agent_idx]) > 1:
                prev_prev_action = self.agent_actions[agent_idx][-2]
                prev_prev_weight = WEIGHT_MAP.get(prev_prev_action, 0.5)
            
            # Weight 변화에 따른 거래 비용 (이전 스텝에서 발생한 거래)
            weight_change = abs(prev_weight - prev_prev_weight)
            txn_cost = transaction_cost * weight_change
            
            # 포트폴리오 수익률 = prev_weight * price_change - txn_cost
            portfolio_return = prev_weight * price_change - txn_cost
            
            new_pv = prev_pv * (1 + portfolio_return)
            new_pv = max(new_pv, 1.0)  # floor
            
            self.agent_portfolio_values[agent_idx].append(new_pv)
            
            # Return 계산
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
        
        각 에이전트의 액션을 기록하여 나중에 성능 추적에 사용.
        """
        pool_output = active_pool.get_pool_actions(state, return_probs=True)
        
        individual_probs = pool_output['probabilities']
        individual_actions = pool_output['actions']
        num_agents = len(individual_probs)
        
        # 각 에이전트의 액션 기록
        for i, action in enumerate(individual_actions):
            if i not in self.agent_actions:
                self.agent_actions[i] = []
            self.agent_actions[i].append(int(action))
        
        # Dynamic weights 계산
        weights = self.weight_calculator.calculate_weights()
        
        if len(weights) != num_agents:
            self.weight_calculator.initialize_agents(num_agents)
            weights = self.weight_calculator.calculate_weights()
        
        # 초기에는 균등 가중치 (데이터 부족 시)
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
