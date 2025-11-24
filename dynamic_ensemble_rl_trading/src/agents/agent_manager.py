"""
Hierarchical agent manager for regime-specific agent pools.

This module manages three separate agent pools (Bull, Bear, Sideways)
and provides methods to train, save, and load all pools.
"""

from typing import Dict, Optional
from pathlib import Path
import logging

from .pool import PPOAgentPool

logger = logging.getLogger(__name__)


class HierarchicalAgentManager:
    """
    Manages three separate agent pools for different market regimes.
    
    Each pool contains 5 PPO agents specialized for a specific regime:
    - Bull Pool: Optimized for profit maximization
    - Bear Pool: Optimized for risk minimization
    - Sideways Pool: Optimized for whipsaw avoidance
    """
    
    def __init__(
        self,
        bull_env,
        bear_env,
        sideways_env,
        num_agents_per_pool: int = 5,
        base_seed: int = 42,
        agent_kwargs: Optional[Dict] = None
    ):
        """
        Initialize the hierarchical agent manager.
        
        Parameters
        ----------
        bull_env : gym.Env
            Trading environment for Bull market.
        bear_env : gym.Env
            Trading environment for Bear market.
        sideways_env : gym.Env
            Trading environment for Sideways market.
        num_agents_per_pool : int, default=5
            Number of agents in each pool.
        base_seed : int, default=42
            Base seed for generating agent seeds.
        agent_kwargs : dict, optional
            Additional arguments for PPO agent initialization.
        """
        self.num_agents_per_pool = num_agents_per_pool
        
        # Initialize three agent pools
        self.bull_pool = PPOAgentPool(
            env=bull_env,
            num_agents=num_agents_per_pool,
            base_seed=base_seed,
            agent_kwargs=agent_kwargs
        )
        
        self.bear_pool = PPOAgentPool(
            env=bear_env,
            num_agents=num_agents_per_pool,
            base_seed=base_seed + 100,  # Different seed range
            agent_kwargs=agent_kwargs
        )
        
        self.sideways_pool = PPOAgentPool(
            env=sideways_env,
            num_agents=num_agents_per_pool,
            base_seed=base_seed + 200,  # Different seed range
            agent_kwargs=agent_kwargs
        )
        
        logger.info(
            f"Initialized hierarchical agent manager with "
            f"{num_agents_per_pool} agents per pool"
        )
    
    def get_pool(self, regime: str) -> PPOAgentPool:
        """
        Get the agent pool for a specific regime.
        
        Parameters
        ----------
        regime : str
            Market regime: 'Bull', 'Bear', or 'Sideways'.
        
        Returns
        -------
        PPOAgentPool
            The agent pool for the specified regime.
        """
        regime = regime.lower()
        
        if regime == 'bull':
            return self.bull_pool
        elif regime == 'bear':
            return self.bear_pool
        elif regime == 'sideways':
            return self.sideways_pool
        else:
            raise ValueError(
                f"Unknown regime: {regime}. Must be 'Bull', 'Bear', or 'Sideways'"
            )
    
    def train_all_pools(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10
    ) -> None:
        """
        Train all agent pools.
        
        Parameters
        ----------
        total_timesteps : int
            Total timesteps per agent.
        callback : callable, optional
            Callback function for training progress.
        log_interval : int, default=10
            Logging interval.
        """
        logger.info("Training all agent pools")
        
        logger.info("Training Bull pool...")
        self.bull_pool.train_pool(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
        
        logger.info("Training Bear pool...")
        self.bear_pool.train_pool(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
        
        logger.info("Training Sideways pool...")
        self.sideways_pool.train_pool(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
        
        logger.info("All agent pools trained")
    
    def train_pool_by_regime(
        self,
        regime: str,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10
    ) -> None:
        """
        Train a specific agent pool by regime.
        
        Parameters
        ----------
        regime : str
            Market regime: 'Bull', 'Bear', or 'Sideways'.
        total_timesteps : int
            Total timesteps per agent.
        callback : callable, optional
            Callback function for training progress.
        log_interval : int, default=10
            Logging interval.
        """
        pool = self.get_pool(regime)
        logger.info(f"Training {regime} pool...")
        pool.train_pool(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
    
    def save_all_pools(self, base_path: str) -> None:
        """
        Save all agent pools.
        
        Parameters
        ----------
        base_path : str
            Base path for saving. Pools will be saved as:
            {base_path}/bull_pool/
            {base_path}/bear_pool/
            {base_path}/sideways_pool/
        """
        base_path = Path(base_path)
        
        self.bull_pool.save_pool(str(base_path / "bull_pool"))
        self.bear_pool.save_pool(str(base_path / "bear_pool"))
        self.sideways_pool.save_pool(str(base_path / "sideways_pool"))
        
        logger.info(f"Saved all agent pools to {base_path}")
    
    def load_all_pools(self, base_path: str) -> None:
        """
        Load all agent pools.
        
        Parameters
        ----------
        base_path : str
            Base path for loading pools.
        """
        base_path = Path(base_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"Pools directory not found: {base_path}")
        
        self.bull_pool.load_pool(str(base_path / "bull_pool"))
        self.bear_pool.load_pool(str(base_path / "bear_pool"))
        self.sideways_pool.load_pool(str(base_path / "sideways_pool"))
        
        logger.info(f"Loaded all agent pools from {base_path}")

