"""
Trading environment for reinforcement learning.

This module implements a Gymnasium-compatible trading environment
with regime-specific reward functions and realistic transaction costs.
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
    
    Supports different reward functions based on market regime:
    - Bull: Profit maximization
    - Bear: Sortino Ratio and drawdown minimization
    - Sideways: Whipsaw avoidance
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        ohlcv_data: pd.DataFrame,
        state_data: pd.DataFrame,
        regime_type: str = 'Bull',
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.0005,  # 0.05%
        slippage: float = 0.0002,  # 0.02%
        max_position: float = 1.0,
        ohlcv_columns: Optional[dict] = None
    ):
        """
        Initialize the trading environment.
        
        Parameters
        ----------
        ohlcv_data : pd.DataFrame
            OHLCV data with datetime index.
        state_data : pd.DataFrame
            Unified state vectors S_t with datetime index.
        regime_type : str, default='Bull'
            Market regime: 'Bull', 'Bear', or 'Sideways'.
        initial_balance : float, default=10000.0
            Initial portfolio balance.
        transaction_fee : float, default=0.0005
            Transaction fee as fraction (0.05%).
        slippage : float, default=0.0002
            Slippage as fraction (0.02%).
        max_position : float, default=1.0
            Maximum position size (1.0 = 100% of capital).
        ohlcv_columns : dict, optional
            Mapping of OHLCV column names.
        """
        super().__init__()
        
        self.ohlcv_data = ohlcv_data
        self.state_data = state_data
        self.regime_type = regime_type
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position = max_position
        
        if ohlcv_columns is None:
            self.ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        else:
            self.ohlcv_columns = ohlcv_columns
        
        # Align data indices
        common_indices = (
            self.ohlcv_data.index.intersection(self.state_data.index)
        )
        if len(common_indices) == 0:
            raise ValueError("No common timestamps between OHLCV and state data")
        
        self.ohlcv_data = self.ohlcv_data.loc[common_indices]
        self.state_data = self.state_data.loc[common_indices]
        
        # Action space: 5 discrete actions
        # 0: Strong Sell, 1: Sell, 2: Hold, 3: Buy, 4: Strong Buy
        self.action_space = spaces.Discrete(5)
        
        # Observation space: unified state vector
        state_dim = self.state_data.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Reward calculator
        self.reward_calculator = RegimeRewardCalculator(
            transaction_cost=self.transaction_fee
        )
        
        # Environment state
        self.current_step = 0
        self.timestamps = self.state_data.index
        self.balance = self.initial_balance
        self.position = 0.0  # Position size (fraction of capital)
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        
        logger.info(
            f"Initialized {regime_type} trading environment with "
            f"{len(self.timestamps)} timesteps"
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Additional options.
        
        Returns
        -------
        observation : np.ndarray
            Initial state vector.
        info : dict
            Additional information.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        self.reward_calculator.reset()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : int
            Action to take (0-4).
        
        Returns
        -------
        observation : np.ndarray
            Next state vector.
        reward : float
            Reward for the action.
        terminated : bool
            Whether episode is terminated.
        truncated : bool
            Whether episode is truncated.
        info : dict
            Additional information.
        """
        if self.current_step >= len(self.timestamps) - 1:
            # End of data
            observation = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # Get current price (use next timestep's open price to avoid look-ahead)
        current_timestamp = self.timestamps[self.current_step]
        next_timestamp = self.timestamps[self.current_step + 1]
        
        current_price = self.ohlcv_data.loc[current_timestamp, self.ohlcv_columns['close']]
        next_open_price = self.ohlcv_data.loc[next_timestamp, self.ohlcv_columns['open']]
        
        # Apply slippage
        execution_price = next_open_price * (1 + self.slippage)
        
        # Portfolio value before action
        portfolio_value_before = self.portfolio_value
        
        # Execute action
        transaction_cost_incurred = self._execute_action(action, execution_price)
        
        # Update portfolio value
        self._update_portfolio_value(execution_price)
        
        # Portfolio value after action
        portfolio_value_after = self.portfolio_value
        
        # Calculate reward based on regime
        reward = self.reward_calculator.calculate_reward(
            self.regime_type,
            portfolio_value_before,
            portfolio_value_after,
            transaction_cost_incurred
        )
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Check if episode is done
        terminated = self.current_step >= len(self.timestamps) - 1
        truncated = False
        
        info = self._get_info()
        info['portfolio_value'] = self.portfolio_value
        info['position'] = self.position
        info['reward'] = reward
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, price: float) -> float:
        """
        Execute trading action.
        
        Parameters
        ----------
        action : int
            Action to execute (0-4).
        price : float
            Execution price.
        
        Returns
        -------
        float
            Transaction cost incurred.
        """
        # Action mapping
        # 0: Strong Sell (-1.0), 1: Sell (-0.5), 2: Hold (0.0),
        # 3: Buy (0.5), 4: Strong Buy (1.0)
        action_weights = [-1.0, -0.5, 0.0, 0.5, 1.0]
        target_position = action_weights[action] * self.max_position
        
        # Calculate position change
        position_change = target_position - self.position
        
        if abs(position_change) < 1e-6:
            # No position change (Hold)
            return 0.0
        
        # Calculate transaction amount
        transaction_amount = abs(position_change) * self.portfolio_value
        
        # Calculate transaction cost
        transaction_cost = transaction_amount * self.transaction_fee
        
        # Update position
        self.position = target_position
        
        # Update balance (simplified: assume we can always execute)
        # In reality, this would depend on available capital
        self.balance -= transaction_cost
        
        return transaction_cost
    
    def _update_portfolio_value(self, current_price: float) -> None:
        """
        Update portfolio value based on current position and price.
        
        Parameters
        ----------
        current_price : float
            Current market price.
        """
        # Simplified portfolio value calculation
        # In a real implementation, this would track actual holdings
        base_value = self.initial_balance
        
        # Portfolio value = base + position * price_change
        # This is a simplified model
        if self.current_step > 0:
            prev_timestamp = self.timestamps[self.current_step - 1]
            prev_price = self.ohlcv_data.loc[prev_timestamp, self.ohlcv_columns['close']]
            price_change = (current_price - prev_price) / prev_price
        else:
            price_change = 0.0
        
        # Update portfolio value
        self.portfolio_value = base_value * (1 + self.position * price_change)
        self.portfolio_value = max(0.0, self.portfolio_value)  # Non-negative
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state vector).
        
        Returns
        -------
        np.ndarray
            Current state vector S_t.
        """
        if self.current_step >= len(self.timestamps):
            # Return last observation
            return self.state_data.iloc[-1].values.astype(np.float32)
        
        timestamp = self.timestamps[self.current_step]
        observation = self.state_data.loc[timestamp].values.astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment state.
        
        Returns
        -------
        dict
            Information dictionary.
        """
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'regime': self.regime_type
        }
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment (optional).
        
        Parameters
        ----------
        mode : str, default='human'
            Rendering mode.
        """
        if mode == 'human':
            print(
                f"Step: {self.current_step}, "
                f"Portfolio Value: {self.portfolio_value:.2f}, "
                f"Position: {self.position:.2f}"
            )

