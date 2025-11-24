"""
Agent pool management for multiple PPO agents.

This module manages a collection of PPO agents with different random seeds
to ensure policy diversity within each regime pool.
"""

import numpy as np
from typing import List, Optional, Dict
import logging

from .ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


class PPOAgentPool:
    """
    Manages a pool of PPO agents for ensemble learning.
    
    Each agent is initialized with a different random seed to ensure
    policy diversity. Agents are trained independently with shuffled
    mini-batches to prevent mode collapse.
    """
    
    def __init__(
        self,
        env,
        num_agents: int = 5,
        base_seed: int = 42,
        agent_kwargs: Optional[Dict] = None
    ):
        """
        Initialize the agent pool.
        
        Parameters
        ----------
        env : gym.Env
            Trading environment.
        num_agents : int, default=5
            Number of agents in the pool.
        base_seed : int, default=42
            Base seed for generating agent seeds.
        agent_kwargs : dict, optional
            Additional arguments for PPO agent initialization.
        """
        self.env = env
        self.num_agents = num_agents
        self.base_seed = base_seed
        
        if agent_kwargs is None:
            agent_kwargs = {}
        
        # Generate seeds for each agent
        agent_seeds = [base_seed + i for i in range(num_agents)]
        
        # Initialize agents with different seeds
        self.agents: List[PPOAgent] = []
        for i, seed in enumerate(agent_seeds):
            agent = PPOAgent(
                env=env,
                seed=seed,
                **agent_kwargs
            )
            self.agents.append(agent)
            logger.info(f"Initialized agent {i+1}/{num_agents} with seed {seed}")
        
        logger.info(f"Created agent pool with {num_agents} agents")
    
    def train_pool(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10
    ) -> None:
        """
        Train all agents in the pool.
        
        Each agent is trained independently with different random seeds,
        ensuring policy diversity through independent mini-batch shuffling.
        
        Parameters
        ----------
        total_timesteps : int
            Total timesteps per agent.
        callback : callable, optional
            Callback function for training progress.
        log_interval : int, default=10
            Logging interval.
        """
        logger.info(
            f"Training {self.num_agents} agents for {total_timesteps} timesteps each"
        )
        
        for i, agent in enumerate(self.agents):
            logger.info(f"Training agent {i+1}/{self.num_agents}")
            agent.train(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=log_interval
            )
        
        logger.info("All agents in pool trained")
    
    def get_pool_actions(
        self,
        observation: np.ndarray,
        return_probs: bool = True
    ) -> Dict:
        """
        Get actions and policies from all agents in the pool.
        
        Parameters
        ----------
        observation : np.ndarray
            Current state observation.
        return_probs : bool, default=True
            Whether to return probability distributions.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'actions': List of actions from each agent
            - 'probabilities': List of probability distributions (if return_probs=True)
        """
        actions = []
        probabilities = []
        
        for agent in self.agents:
            action = agent.predict(observation, deterministic=False)
            actions.append(action)
            
            if return_probs:
                probs = agent.get_policy_distribution(observation)
                probabilities.append(probs)
        
        result = {'actions': actions}
        if return_probs:
            result['probabilities'] = probabilities
        
        return result
    
    def get_agent(self, index: int) -> PPOAgent:
        """
        Get a specific agent from the pool.
        
        Parameters
        ----------
        index : int
            Agent index (0 to num_agents-1).
        
        Returns
        -------
        PPOAgent
            The requested agent.
        """
        if index < 0 or index >= self.num_agents:
            raise IndexError(f"Agent index {index} out of range [0, {self.num_agents-1}]")
        
        return self.agents[index]
    
    def save_pool(self, base_path: str) -> None:
        """
        Save all agents in the pool.
        
        Parameters
        ----------
        base_path : str
            Base path for saving agents. Each agent will be saved as
            {base_path}/agent_{i}.zip
        """
        from pathlib import Path
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent_path = base_path / f"agent_{i}.zip"
            agent.save(str(agent_path))
        
        logger.info(f"Saved {self.num_agents} agents to {base_path}")
    
    def load_pool(self, base_path: str) -> None:
        """
        Load all agents in the pool.
        
        Parameters
        ----------
        base_path : str
            Base path for loading agents.
        """
        from pathlib import Path
        base_path = Path(base_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"Pool directory not found: {base_path}")
        
        for i, agent in enumerate(self.agents):
            agent_path = base_path / f"agent_{i}.zip"
            if agent_path.exists():
                agent.load(str(agent_path))
            else:
                logger.warning(f"Agent {i} file not found: {agent_path}")
        
        logger.info(f"Loaded {self.num_agents} agents from {base_path}")

