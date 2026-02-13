"""
Trading environment for reinforcement learning.

Implements a Gymnasium-compatible trading environment
with regime-specific reward functions, realistic transaction costs,
and CORRECT portfolio accounting (cash + holdings_qty * price).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import logging

from .rewards import RegimeRewardCalculator

logger = logging.getLogger(__name__)


class MultiRegimeTradingEnv(gym.Env):
    """
    Multi-regime trading environment for reinforcement learning.

    Action space: Discrete(5) -> {Strong Sell, Sell, Hold, Buy, Strong Buy}
    Observation space: Unified state vector S_t

    Portfolio accounting:
      cash + holdings_qty * current_price = portfolio_value
    """

    metadata = {'render_modes': ['human']}

    # Long-only weight map (논문 기본 설정)
    WEIGHT_MAP = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}

    def __init__(
        self,
        ohlcv_data: pd.DataFrame,
        state_data: pd.DataFrame,
        regime_type: str = 'Bull',
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.0005,   # 0.05%
        slippage: float = 0.0002,           # 0.02%
        max_position: float = 1.0,
        reward_scale: float = 100.0,
        ohlcv_columns: Optional[dict] = None,
    ):
        super().__init__()

        self.ohlcv_data = ohlcv_data
        self.state_data = state_data
        self.regime_type = regime_type
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position = max_position
        self.reward_scale = reward_scale

        if ohlcv_columns is None:
            self.ohlcv_columns = {
                'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'volume': 'volume',
            }
        else:
            self.ohlcv_columns = ohlcv_columns

        # Align indices
        common_indices = self.ohlcv_data.index.intersection(self.state_data.index)
        if len(common_indices) == 0:
            raise ValueError("No common timestamps between OHLCV and state data")
        self.ohlcv_data = self.ohlcv_data.loc[common_indices]
        self.state_data = self.state_data.loc[common_indices]

        # Spaces
        self.action_space = spaces.Discrete(5)
        state_dim = self.state_data.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32,
        )

        # Reward calculator (시간봉 데이터용 스케일링)
        self.reward_calculator = RegimeRewardCalculator(
            transaction_cost=self.transaction_fee,
            reward_scale=self.reward_scale
        )

        # Internal state (reset in reset())
        self.timestamps = self.state_data.index
        self.current_step = 0
        self.cash = self.initial_balance
        self.holdings_qty = 0.0        # number of units held
        self.position = 0.0            # target weight fraction
        self.portfolio_value = self.initial_balance
        self.balance = self.initial_balance  # alias kept for compat
        self.trade_history = []
        self._entry_price = None       # price at which holdings were acquired

        logger.info(
            f"Initialized {regime_type} trading environment with "
            f"{len(self.timestamps)} timesteps"
        )

    # ──────────────────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_balance
        self.holdings_qty = 0.0
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.balance = self.initial_balance
        self.trade_history = []
        self._entry_price = None
        self.reward_calculator.reset()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # ──────────────────────────────────────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        if self.current_step >= len(self.timestamps) - 1:
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info

        cur_ts = self.timestamps[self.current_step]
        nxt_ts = self.timestamps[self.current_step + 1]

        close_price = float(self.ohlcv_data.loc[cur_ts, self.ohlcv_columns['close']])
        next_open   = float(self.ohlcv_data.loc[nxt_ts, self.ohlcv_columns['open']])

        # ── mark-to-market BEFORE executing action ──
        pv_before = self.cash + self.holdings_qty * close_price

        # ── rebalance ──
        exec_price = next_open   # execution at next open
        target_weight = self.WEIGHT_MAP.get(action, 0.5)
        target_value  = target_weight * pv_before
        target_qty    = target_value / exec_price if exec_price > 0 else 0.0
        qty_change    = target_qty - self.holdings_qty

        txn_cost = 0.0
        if abs(qty_change) > 1e-10:
            # slippage direction
            if qty_change > 0:
                actual_price = exec_price * (1 + self.slippage)
            else:
                actual_price = exec_price * (1 - self.slippage)

            trade_value = abs(qty_change * actual_price)
            txn_cost = trade_value * self.transaction_fee

            self.cash -= qty_change * actual_price + txn_cost
            self.holdings_qty = target_qty
            self.position = target_weight

        # ── mark-to-market AFTER rebalance (at close of NEXT bar) ──
        next_close = float(self.ohlcv_data.loc[nxt_ts, self.ohlcv_columns['close']])
        pv_after = self.cash + self.holdings_qty * next_close
        pv_after = max(pv_after, 1.0)  # floor

        self.portfolio_value = pv_after
        self.balance = self.cash

        # ── reward ──
        reward = self.reward_calculator.calculate_reward(
            self.regime_type, pv_before, pv_after, txn_cost,
        )

        # advance
        self.current_step += 1
        obs = self._get_observation()
        terminated = self.current_step >= len(self.timestamps) - 1
        info = self._get_info()
        info['portfolio_value'] = self.portfolio_value
        info['position'] = self.position
        info['reward'] = reward

        return obs, reward, terminated, False, info

    # ──────────────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.timestamps):
            return self.state_data.iloc[-1].values.astype(np.float32)
        ts = self.timestamps[self.current_step]
        return self.state_data.loc[ts].values.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            'step': self.current_step,
            'balance': self.cash,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'regime': self.regime_type,
        }

    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            print(
                f"Step: {self.current_step}, "
                f"Portfolio Value: {self.portfolio_value:.2f}, "
                f"Position: {self.position:.2f}"
            )
