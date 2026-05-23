"""Reinforcement Learning module for trading."""

# Import main components for easy access
from src.rl.agents import PPOAgent, PPOConfig
from src.rl.envs import (
    PnLReward,
    SharpeReward,
    SingleAssetConfig,
    SingleAssetTradingEnv,
)
from src.rl.metrics import compute_max_drawdown, compute_sharpe
from src.rl.models import MLPNetwork, NetworkConfig
from src.rl.training import ReplayBuffer, RolloutBuffer
from src.rl.utils import fee_curriculum_factor, narrow, narrow_int

__all__ = [
    # Agents
    "PPOAgent",
    "PPOConfig",
    # Environments
    "SingleAssetTradingEnv",
    "SingleAssetConfig",
    # Rewards
    "PnLReward",
    "SharpeReward",
    # Models
    "MLPNetwork",
    "NetworkConfig",
    # Training
    "RolloutBuffer",
    "ReplayBuffer",
    # Metrics
    "compute_sharpe",
    "compute_max_drawdown",
    # Utils
    "narrow",
    "narrow_int",
    "fee_curriculum_factor",
]
