"""
PPO (Proximal Policy Optimization) agent implementation.

This module implements PPO agents for reinforcement learning in trading.
Uses Stable Baselines3 or custom PyTorch implementation.
"""

import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import logging

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    logging.warning(
        "Stable Baselines3 not available. Custom PPO implementation required."
    )

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    PPO agent for trading environment.
    
    Wraps Stable Baselines3 PPO or provides custom implementation
    for training and inference in trading environments.
    """
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict] = None,
        seed: int = 42,
        device: str = 'auto'
    ):
        """
        Initialize PPO agent.
        
        Parameters
        ----------
        env : gym.Env
            Trading environment.
        learning_rate : float, default=3e-4
            Learning rate for optimizer.
        batch_size : int, default=64
            Minibatch size.
        n_steps : int, default=2048
            Number of steps to collect per update.
        n_epochs : int, default=10
            Number of epochs for policy update.
        gamma : float, default=0.99
            Discount factor.
        gae_lambda : float, default=0.95
            GAE lambda parameter.
        clip_range : float, default=0.2
            PPO clip range.
        ent_coef : float, default=0.01
            Entropy coefficient.
        vf_coef : float, default=0.5
            Value function coefficient.
        max_grad_norm : float, default=0.5
            Maximum gradient norm for clipping.
        policy_kwargs : dict, optional
            Additional arguments for policy network.
        seed : int, default=42
            Random seed.
        device : str, default='auto'
            Device to use ('cpu', 'cuda', or 'auto').
        """
        self.env = env
        self.seed = seed
        
        if policy_kwargs is None:
            policy_kwargs = {
                'net_arch': [256, 256],
                'activation_fn': 'tanh'
            }
        
        if STABLE_BASELINES3_AVAILABLE:
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                seed=seed,
                device=device,
                verbose=0
            )
            self.use_stable_baselines = True
        else:
            raise NotImplementedError(
                "Stable Baselines3 is required. "
                "Custom PPO implementation not yet available."
            )
        
        logger.info(f"Initialized PPO agent with seed {seed}")
    
    def train(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10
    ) -> None:
        """
        Train the PPO agent.
        
        Parameters
        ----------
        total_timesteps : int
            Total number of timesteps to train.
        callback : callable, optional
            Callback function for training progress.
        log_interval : int, default=10
            Logging interval.
        """
        logger.info(f"Training PPO agent for {total_timesteps} timesteps")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
        
        logger.info("PPO agent training completed")
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> tuple:
        """
        Predict action from observation.
        
        Parameters
        ----------
        observation : np.ndarray
            Current state observation.
        deterministic : bool, default=False
            Whether to use deterministic policy.
        
        Returns
        -------
        action : int
            Predicted action.
        """
        action, _ = self.model.predict(
            observation,
            deterministic=deterministic
        )
        return action
    
    def get_policy_distribution(
        self,
        observation: np.ndarray
    ) -> np.ndarray:
        """
        Get policy probability distribution over actions.
        
        Parameters
        ----------
        observation : np.ndarray
            Current state observation.
        
        Returns
        -------
        np.ndarray
            Probability distribution over actions.
        """
        if not self.use_stable_baselines:
            raise NotImplementedError(
                "Policy distribution extraction requires Stable Baselines3"
            )
        
        # Get policy network output
        obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
        
        # Get action distribution
        distribution = self.model.policy.get_distribution(obs_tensor)
        
        # Get probabilities
        probs = distribution.distribution.probs.detach().cpu().numpy()
        
        return probs.flatten()
    
    def save(self, filepath: str) -> None:
        """
        Save the trained agent.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(filepath))
        logger.info(f"PPO agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained agent.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
        """
        if not STABLE_BASELINES3_AVAILABLE:
            raise ImportError("Stable Baselines3 required for loading models")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = PPO.load(str(filepath), env=self.env)
        logger.info(f"PPO agent loaded from {filepath}")

