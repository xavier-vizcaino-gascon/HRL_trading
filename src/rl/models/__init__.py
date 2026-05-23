"""Neural network models for RL agents."""

from src.rl.models.base import BaseNetwork, NetworkConfig, init_weights
from src.rl.models.mlp import DuelingMLPNetwork, MLPNetwork, ResidualMLPNetwork
from src.rl.models.policy_heads import (
    CategoricalPolicyHead,
    DeterministicPolicyHead,
    GaussianPolicyHead,
)
from src.rl.models.value_heads import DuelingHead, QHead, TwinQHead, ValueHead

__all__ = [
    # Base
    "BaseNetwork",
    "NetworkConfig",
    "init_weights",
    # MLP models
    "MLPNetwork",
    "ResidualMLPNetwork",
    "DuelingMLPNetwork",
    # Policy heads
    "DeterministicPolicyHead",
    "GaussianPolicyHead",
    "CategoricalPolicyHead",
    # Value heads
    "ValueHead",
    "QHead",
    "DuelingHead",
    "TwinQHead",
]
