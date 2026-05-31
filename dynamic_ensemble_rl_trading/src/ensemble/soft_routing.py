"""
Soft regime routing: blend Bull / Bear / Sideways pool policies by p(R|s).

Instead of hard argmax routing (one pool per bar), compute:

    pi_ensemble(a|s) = sum_R p(R|s) * pi_R(a|s)

where pi_R comes from the dynamic-weight ensemble inside each pool.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from ..agents.agent_manager import HierarchicalAgentManager
from .ensemble_trader import EnsembleTrader
from .weighting import DynamicWeightCalculator

logger = logging.getLogger(__name__)

REGIME_ORDER = ("Bear", "Sideways", "Bull")


class SoftRoutingEnsemble:
    """Three pool-level ensemble traders blended by regime probabilities."""

    def __init__(
        self,
        performance_window: int = 30,
        temperature: float = 5.0,
        initial_capital: float = 10000.0,
        num_agents_per_pool: int = 5,
    ):
        self.num_agents_per_pool = num_agents_per_pool
        self._pool_traders: Dict[str, EnsembleTrader] = {}
        for regime in REGIME_ORDER:
            wc = DynamicWeightCalculator(
                performance_window=performance_window,
                temperature=temperature,
            )
            et = EnsembleTrader(
                weight_calculator=wc,
                performance_window=performance_window,
                temperature=temperature,
                initial_capital=initial_capital,
            )
            et.initialize_agents(num_agents_per_pool)
            self._pool_traders[regime] = et

    def reset(self) -> None:
        for et in self._pool_traders.values():
            et.reset()

    def update_all_with_price_change(
        self, price_change: float, transaction_cost: float
    ) -> None:
        for et in self._pool_traders.values():
            et.update_all_agents_with_price_change(price_change, transaction_cost)

    def _pool_ensemble_probs(
        self, state: np.ndarray, pool_name: str, agent_manager: HierarchicalAgentManager
    ) -> np.ndarray:
        pool = agent_manager.get_pool(pool_name)
        et = self._pool_traders[pool_name]
        result = et.get_ensemble_action(state, pool)
        return np.asarray(result["probabilities"], dtype=np.float64)

    def get_soft_action(
        self,
        state: np.ndarray,
        regime_probs: np.ndarray,
        agent_manager: HierarchicalAgentManager,
    ) -> Dict:
        """
        Blend pool policies by regime probability vector [Bear, Sideways, Bull].
        """
        regime_probs = np.asarray(regime_probs, dtype=np.float64).reshape(3)
        regime_probs = np.clip(regime_probs, 0.0, 1.0)
        total = regime_probs.sum()
        if total <= 0:
            regime_probs = np.ones(3) / 3.0
        else:
            regime_probs = regime_probs / total

        blended = np.zeros(5, dtype=np.float64)
        pool_probs: Dict[str, List[float]] = {}

        for i, regime in enumerate(REGIME_ORDER):
            p_r = regime_probs[i]
            pi_r = self._pool_ensemble_probs(state, regime, agent_manager)
            pool_probs[regime] = pi_r.tolist()
            blended += p_r * pi_r

        blended = blended / (blended.sum() + 1e-10)
        weight_levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
        expected_weight = float(np.sum(blended * weight_levels))

        action = int(np.argmax(blended))
        dominant_regime = REGIME_ORDER[int(np.argmax(regime_probs))]

        return {
            "action": action,
            "probabilities": blended,
            "regime_probs": regime_probs.tolist(),
            "pool_probs": pool_probs,
            "dominant_regime": dominant_regime,
            "weights": regime_probs.tolist(),
            "expected_weight": expected_weight,
        }
