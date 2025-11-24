"""
Random seed management for reproducibility.

This module provides functions to set random seeds across different
libraries (NumPy, PyTorch, Python random) to ensure reproducible results.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value to use for all random number generators.
    
    Notes
    -----
    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator (both CPU and CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_agent_seeds(base_seed: int = 42, num_agents: int = 5) -> list:
    """
    Generate different random seeds for multiple agents.
    
    Parameters
    ----------
    base_seed : int, default=42
        Base seed value.
    num_agents : int, default=5
        Number of agents to generate seeds for.
    
    Returns
    -------
    list of int
        List of unique seeds for each agent.
    """
    return [base_seed + i for i in range(num_agents)]

