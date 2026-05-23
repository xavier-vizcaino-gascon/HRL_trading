"""
Value head modules for critic networks.

Implements value estimation for actor-critic algorithms.
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    State value head V(s).

    Estimates the value of being in a state.
    Used in algorithms like A2C, PPO.
    """

    def __init__(self, input_dim: int):
        """
        Initialize value head.

        Args:
            input_dim: Input feature dimension
        """
        super().__init__()

        self.value = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Feature tensor [batch_size, input_dim]

        Returns:
            Value tensor [batch_size, 1]
        """
        return self.value(features)


class QHead(nn.Module):
    """
    Action-value head Q(s, a).

    Estimates the value of taking an action in a state.
    Used in algorithms like DQN, DDPG.
    """

    def __init__(self, input_dim: int, action_dim: int):
        """
        Initialize Q-value head.

        Args:
            input_dim: Input feature dimension (state + action for continuous)
            action_dim: Action space dimension (for discrete actions)
        """
        super().__init__()

        self.q_value = nn.Linear(input_dim, action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Feature tensor [batch_size, input_dim]

        Returns:
            Q-values tensor [batch_size, action_dim]
        """
        return self.q_value(features)


class DuelingHead(nn.Module):
    """
    Dueling architecture head.

    Combines value V(s) and advantage A(s,a) streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A))

    Used in Dueling DQN.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize dueling head.

        Args:
            input_dim: Input feature dimension
            action_dim: Action space dimension
            hidden_dim: Hidden dimension for streams
        """
        super().__init__()

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.

        Args:
            features: Feature tensor [batch_size, input_dim]

        Returns:
            Q-values tensor [batch_size, action_dim]
        """
        value = self.value_stream(features)  # [batch_size, 1]
        advantage = self.advantage_stream(features)  # [batch_size, action_dim]

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class TwinQHead(nn.Module):
    """
    Twin Q-networks for reducing overestimation bias.

    Uses two Q-networks and takes the minimum for stability.
    Used in algorithms like SAC, TD3.
    """

    def __init__(self, input_dim: int, action_dim: int):
        """
        Initialize twin Q-head.

        Args:
            input_dim: Input feature dimension
            action_dim: Action space dimension
        """
        super().__init__()

        self.q1 = nn.Linear(input_dim, action_dim)
        self.q2 = nn.Linear(input_dim, action_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.

        Args:
            features: Feature tensor [batch_size, input_dim]

        Returns:
            Tuple of (q1_values, q2_values)
        """
        q1_values = self.q1(features)
        q2_values = self.q2(features)

        return q1_values, q2_values

    def get_min_q(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get minimum Q-values across both networks.

        Args:
            features: Feature tensor

        Returns:
            Minimum Q-values tensor
        """
        q1_values, q2_values = self.forward(features)
        return torch.min(q1_values, q2_values)
