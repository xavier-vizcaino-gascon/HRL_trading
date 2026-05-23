"""
Policy head modules for actor-critic algorithms.

Implements different policy distributions for discrete and continuous action spaces.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class DeterministicPolicyHead(nn.Module):
    """
    Deterministic policy head for continuous actions.

    Outputs a single action value without stochasticity.
    Used in algorithms like DDPG.
    """

    def __init__(
        self, input_dim: int, action_dim: int, action_low: float = -1.0, action_high: float = 1.0
    ):
        """
        Initialize deterministic policy head.

        Args:
            input_dim: Input feature dimension
            action_dim: Action space dimension
            action_low: Minimum action value
            action_high: Maximum action value
        """
        super().__init__()

        self.action_low = action_low
        self.action_high = action_high

        self.policy = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Feature tensor [batch_size, input_dim]

        Returns:
            Actions tensor [batch_size, action_dim]
        """
        actions = self.policy(features)

        # Scale from [-1, 1] to [action_low, action_high]
        actions = self.action_low + (actions + 1.0) * 0.5 * (self.action_high - self.action_low)

        return actions


class GaussianPolicyHead(nn.Module):
    """
    Gaussian (Normal) policy head for continuous actions.

    Outputs mean and log_std for a Gaussian distribution.
    Used in algorithms like PPO, SAC for continuous control.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        learn_std: bool = True,
    ):
        """
        Initialize Gaussian policy head.

        Args:
            input_dim: Input feature dimension
            action_dim: Action space dimension
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            learn_std: Whether to learn std or keep it constant
        """
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.learn_std = learn_std

        # Mean head
        self.mean = nn.Linear(input_dim, action_dim)

        # Log std head
        if learn_std:
            self.log_std = nn.Linear(input_dim, action_dim)
        else:
            # Fixed log std
            self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=False)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Feature tensor [batch_size, input_dim]

        Returns:
            Tuple of (mean, log_std)
        """
        mean = self.mean(features)

        if self.learn_std:
            log_std = self.log_std(features)
        else:
            log_std = self.log_std.expand_as(mean)

        # Clamp log_std for stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_distribution(self, features: torch.Tensor) -> Normal:
        """
        Get action distribution.

        Args:
            features: Feature tensor

        Returns:
            Normal distribution
        """
        mean, log_std = self.forward(features)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from distribution.

        Args:
            features: Feature tensor

        Returns:
            Tuple of (action, log_prob)
        """
        dist = self.get_distribution(features)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions
        return action, log_prob

    def evaluate(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.

        Args:
            features: Feature tensor
            actions: Actions to evaluate

        Returns:
            Tuple of (log_prob, entropy)
        """
        dist = self.get_distribution(features)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CategoricalPolicyHead(nn.Module):
    """
    Categorical policy head for discrete actions.

    Outputs logits for a categorical distribution.
    Used in algorithms like PPO, A2C for discrete control.
    """

    def __init__(self, input_dim: int, action_dim: int):
        """
        Initialize categorical policy head.

        Args:
            input_dim: Input feature dimension
            action_dim: Number of discrete actions
        """
        super().__init__()

        self.policy = nn.Linear(input_dim, action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Feature tensor [batch_size, input_dim]

        Returns:
            Logits tensor [batch_size, action_dim]
        """
        logits = self.policy(features)
        return logits

    def get_distribution(self, features: torch.Tensor) -> Categorical:
        """
        Get action distribution.

        Args:
            features: Feature tensor

        Returns:
            Categorical distribution
        """
        logits = self.forward(features)
        return Categorical(logits=logits)

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from distribution.

        Args:
            features: Feature tensor

        Returns:
            Tuple of (action, log_prob)
        """
        dist = self.get_distribution(features)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.

        Args:
            features: Feature tensor
            actions: Actions to evaluate

        Returns:
            Tuple of (log_prob, entropy)
        """
        dist = self.get_distribution(features)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_probs(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities.

        Args:
            features: Feature tensor

        Returns:
            Probabilities tensor [batch_size, action_dim]
        """
        dist = self.get_distribution(features)
        return dist.probs
