"""
Base classes for RL agents.

Provides abstract interfaces and common functionality for all RL agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class AgentConfig:
    """Configuration for RL agents."""

    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    device: str = "cpu"
    seed: int | None = None

    # For compatibility with different agent types
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Tracking metrics for agent performance."""

    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    q_values: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)

    def add_episode(self, reward: float, length: int) -> None:
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def add_loss(self, loss: float) -> None:
        """Add training loss."""
        self.losses.append(loss)

    def add_q_value(self, q_value: float) -> None:
        """Add Q-value."""
        self.q_values.append(q_value)

    def add_entropy(self, entropy: float) -> None:
        """Add policy entropy."""
        self.entropies.append(entropy)

    def get_recent_mean_reward(self, n: int = 100) -> float:
        """Get mean reward over last n episodes."""
        if len(self.episode_rewards) == 0:
            return 0.0
        return float(np.mean(self.episode_rewards[-n:]))

    def get_recent_mean_length(self, n: int = 100) -> float:
        """Get mean episode length over last n episodes."""
        if len(self.episode_lengths) == 0:
            return 0.0
        return float(np.mean(self.episode_lengths[-n:]))

    def reset(self) -> None:
        """Reset all metrics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.losses.clear()
        self.q_values.clear()
        self.entropies.clear()


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.

    Defines the interface that all agents must implement.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize base agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = AgentMetrics()

        # Set seed for reproducibility
        if config.seed is not None:
            self.set_seed(config.seed)

    @abstractmethod
    def select_action(
        self, observation: NDArray[np.float32], deterministic: bool = False
    ) -> NDArray | int:
        """
        Select an action given an observation.

        Args:
            observation: Current observation from environment
            deterministic: Whether to select action deterministically (for evaluation)

        Returns:
            Action to take (discrete int or continuous array)
        """
        pass

    @abstractmethod
    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Dictionary containing training data

        Returns:
            Dictionary of training metrics (loss, etc.)
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Update agent parameters (if needed).

        Some algorithms like PPO perform batch updates after collecting
        multiple episodes, while others like DQN update after each step.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save agent to disk.

        Args:
            path: Path to save location
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Load agent from disk.

        Args:
            path: Path to load location
        """
        pass

    def set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def to_tensor(self, array: NDArray) -> torch.Tensor:
        """Convert numpy array to tensor on device."""
        return torch.FloatTensor(array).to(self.device)

    def to_numpy(self, tensor: torch.Tensor) -> NDArray:
        """Convert tensor to numpy array."""
        return tensor.detach().cpu().numpy()

    def train_mode(self) -> None:
        """Set agent to training mode."""
        pass  # Override in subclasses if needed

    def eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        pass  # Override in subclasses if needed

    def get_metrics(self) -> AgentMetrics:
        """Get agent metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        self.metrics.reset()


class OnPolicyAgent(BaseAgent):
    """
    Base class for on-policy agents (PPO, A2C, etc.).

    On-policy agents learn from data collected with the current policy.
    They typically collect a batch of experiences and update the policy.
    """

    @abstractmethod
    def collect_rollout(self, env, n_steps: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """
        Collect rollout data from environment.

        Args:
            env: Environment to collect from
            n_steps: Number of steps to collect

        Returns:
            Tuple of (rollout_data, episode_info)
        """
        pass


class OffPolicyAgent(BaseAgent):
    """
    Base class for off-policy agents (DQN, SAC, TD3, etc.).

    Off-policy agents can learn from data collected with any policy.
    They typically use a replay buffer to store and reuse experiences.
    """

    @abstractmethod
    def store_transition(
        self,
        state: NDArray,
        action: NDArray | int,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        pass
