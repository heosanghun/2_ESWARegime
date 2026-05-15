"""
Regime-specific reward functions.

This module implements three regime-aware reward functions for the PPO
agent pools. The v2 redesign (May 2026, following ESWA-D-26-08980
review) targets the failure mode documented in the overnight
diagnostic: even when the classifier is given near-perfect labels via
the lagging SMA scheme, the Bear pool fails to exploit a confirmed
bear test window. The cause was a reward function (Sortino over a
30-bar window) that produced a sparse, low-amplitude signal which
PPO could not credit-assign at hourly granularity.

The v2 reward design (see :pymeth:`RegimeRewardCalculator.calculate_reward`)
combines three terms:

1. **Realised log return per step** (signal for portfolio growth).
2. **Direction-alignment bonus** ``alpha * w * r``, where ``w`` is the
   signed position weight in ``[-1, 1]`` and ``r`` is the bar return.
   This is mathematically related to the realised PnL but is *scaled
   separately*, amplifying the gradient PPO sees toward correctly
   directional positions in each regime.
3. **Cost drag** (transaction cost normalised by previous PV).

A per-regime shaping term then biases the agent toward
regime-appropriate behaviour:

* **Bull**: extra bonus for ``w > 0`` (long bias).
* **Bear**: extra bonus for ``w < 0`` (short bias), with an asymmetric
  penalty for going long during down-bars.
* **Sideways**: penalty proportional to ``|w|`` to favour cash.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RegimeRewardCalculator:
    """
    Calculates regime-specific rewards for RL agents.

    Each market regime has a specialized reward function designed to
    elicit optimal behavior for that specific market condition.

    The v2 API accepts two additional keyword arguments compared to v1:

    * ``target_weight`` — signed portfolio weight the agent just
      committed to in ``[-1, +1]``.
    * ``bar_return`` — the bar's *realised* simple return
      ``(next_close - next_open) / next_open``.

    Both are optional; if omitted the function falls back to the v1
    behaviour (purely PV-based reward, kept for backward compatibility
    with any external callers that have not migrated yet).
    """

    def __init__(
        self,
        transaction_cost: float = 0.0005,  # 0.05%
        risk_free_rate: float = 0.0,
        reward_scale: float = 100.0,  # Scale rewards for hourly data
        direction_bonus_coef: float = 3.0,  # alpha for w*r term
        cost_coef: float = 1.0,
        regime_shaping_coef: float = 0.2,  # weight on regime-specific term
        wrong_side_penalty_coef: float = 1.5,
    ):
        """
        Initialize the reward calculator.
        
        Parameters
        ----------
        transaction_cost : float, default=0.0005
            Transaction cost as fraction (0.05%).
        risk_free_rate : float, default=0.0
            Risk-free rate for Sortino Ratio calculation.
        reward_scale : float, default=100.0
            Reward scaling factor (set >1 for intra-day data where
            per-step returns are very small).
        """
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.reward_scale = reward_scale
        self.direction_bonus_coef = direction_bonus_coef
        self.cost_coef = cost_coef
        self.regime_shaping_coef = regime_shaping_coef
        self.wrong_side_penalty_coef = wrong_side_penalty_coef

        # Track portfolio history for Sortino Ratio calculation (legacy)
        self.portfolio_returns: list = []
        self.portfolio_values: list = []
    
    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _safe_pv_pct(self, pv_before: float, pv_after: float) -> float:
        if pv_before <= 0:
            return 0.0
        return (pv_after - pv_before) / pv_before

    def _normalised_cost(self, txn_cost: float, pv_before: float) -> float:
        if pv_before <= 0:
            return 0.0
        return txn_cost / pv_before

    def _direction_term(
        self,
        target_weight: Optional[float],
        bar_return: Optional[float],
    ) -> float:
        """Signed position * bar return. Zero when either is missing."""
        if target_weight is None or bar_return is None:
            return 0.0
        return float(target_weight) * float(bar_return)

    # ------------------------------------------------------------------
    # Regime-specific rewards (v2)
    # ------------------------------------------------------------------
    def calculate_bull_reward(
        self,
        portfolio_value_before: float,
        portfolio_value_after: float,
        transaction_cost_incurred: float = 0.0,
        target_weight: Optional[float] = None,
        bar_return: Optional[float] = None,
    ) -> float:
        """
        Bull market reward (v2).

        ``R_bull = pv_pct + alpha * w * r - lambda_c * tx_norm
                    + beta * max(0, w) * max(0, r)``

        The first three terms are the unified core reward; the trailing
        term biases the policy toward long exposure during up-bars.
        """
        if portfolio_value_before <= 0:
            return 0.0

        pv_pct = self._safe_pv_pct(portfolio_value_before, portfolio_value_after)
        cost_drag = self._normalised_cost(transaction_cost_incurred, portfolio_value_before)
        direction = self._direction_term(target_weight, bar_return)

        reward = (
            pv_pct
            + self.direction_bonus_coef * direction
            - self.cost_coef * cost_drag
        )

        # Regime shaping: bonus for being long when bar return is positive.
        if target_weight is not None and bar_return is not None:
            long_part = max(0.0, float(target_weight))
            up_part = max(0.0, float(bar_return))
            reward += self.regime_shaping_coef * long_part * up_part

        return reward * self.reward_scale
    
    def calculate_bear_reward(
        self,
        portfolio_value_before: float,
        portfolio_value_after: float,
        transaction_cost_incurred: float,
        target_weight: Optional[float] = None,
        bar_return: Optional[float] = None,
    ) -> float:
        """
        Bear market reward (v2).

        ``R_bear = pv_pct + alpha * w * r - lambda_c * tx_norm
                    + beta * max(0, -w) * max(0, -r)
                    - gamma * max(0, w) * max(0, -r)``

        The Sortino-ratio formulation used in v1 was abandoned because
        it produced a low-amplitude, sparse reward that PPO could not
        credit-assign at hourly granularity. The new shaping has two
        terms:

        * **Short alignment bonus** — reward shorting in down-bars.
        * **Wrong-side penalty** — punish long exposure during
          confirmed down-bars (this is where capital is destroyed in
          a bear regime).
        """
        if portfolio_value_before <= 0:
            return 0.0

        pv_pct = self._safe_pv_pct(portfolio_value_before, portfolio_value_after)
        cost_drag = self._normalised_cost(transaction_cost_incurred, portfolio_value_before)
        direction = self._direction_term(target_weight, bar_return)

        # Keep Sortino tracking for diagnostics (cheap; not used in reward).
        self.portfolio_returns.append(pv_pct)
        self.portfolio_values.append(portfolio_value_after)

        reward = (
            pv_pct
            + self.direction_bonus_coef * direction
            - self.cost_coef * cost_drag
        )

        if target_weight is not None and bar_return is not None:
            short_part = max(0.0, -float(target_weight))
            down_part = max(0.0, -float(bar_return))
            long_part = max(0.0, float(target_weight))

            reward += self.regime_shaping_coef * short_part * down_part
            reward -= self.wrong_side_penalty_coef * long_part * down_part

        return reward * self.reward_scale
    
    def calculate_sideways_reward(
        self,
        portfolio_value_before: float,
        portfolio_value_after: float,
        transaction_cost_incurred: float,
        target_weight: Optional[float] = None,
        bar_return: Optional[float] = None,
    ) -> float:
        """
        Sideways market reward (v2).

        ``R_sideways = pv_pct + alpha * w * r - 5 * lambda_c * tx_norm
                       - beta * |w| * |r|``

        In a sideways regime the rational behaviour is to stay close
        to cash. The ``|w| * |r|`` penalty makes any non-flat position
        costly proportional to bar volatility, while the heavier
        transaction-cost coefficient discourages whipsaw trading.
        """
        if portfolio_value_before <= 0:
            return 0.0

        pv_pct = self._safe_pv_pct(portfolio_value_before, portfolio_value_after)
        cost_drag = self._normalised_cost(transaction_cost_incurred, portfolio_value_before)
        direction = self._direction_term(target_weight, bar_return)

        reward = (
            pv_pct
            + self.direction_bonus_coef * direction
            - 5.0 * self.cost_coef * cost_drag
        )

        if target_weight is not None and bar_return is not None:
            position_size = abs(float(target_weight))
            move_size = abs(float(bar_return))
            reward -= self.regime_shaping_coef * position_size * move_size

        return reward * self.reward_scale
    
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
        transaction_cost_incurred: float = 0.0,
        target_weight: Optional[float] = None,
        bar_return: Optional[float] = None,
    ) -> float:
        """
        Dispatch to the regime-specific reward.

        Parameters
        ----------
        regime : str
            Market regime: ``'Bull'``, ``'Bear'``, or ``'Sideways'``.
        portfolio_value_before, portfolio_value_after : float
            Portfolio value before/after the action.
        transaction_cost_incurred : float, default=0.0
            Transaction cost in absolute units.
        target_weight : float, optional
            Signed portfolio weight just committed to in ``[-1, +1]``.
            When provided, enables the v2 direction-alignment terms.
        bar_return : float, optional
            Simple return between next_open and next_close. When
            provided, enables the v2 direction-alignment terms.
        """
        regime = regime.lower()
        kwargs = {
            "target_weight": target_weight,
            "bar_return": bar_return,
        }

        if regime == "bull":
            return self.calculate_bull_reward(
                portfolio_value_before,
                portfolio_value_after,
                transaction_cost_incurred,
                **kwargs,
            )
        if regime == "bear":
            return self.calculate_bear_reward(
                portfolio_value_before,
                portfolio_value_after,
                transaction_cost_incurred,
                **kwargs,
            )
        if regime == "sideways":
            return self.calculate_sideways_reward(
                portfolio_value_before,
                portfolio_value_after,
                transaction_cost_incurred,
                **kwargs,
            )

        raise ValueError(
            f"Unknown regime: {regime}. Must be 'Bull', 'Bear', or 'Sideways'"
        )

