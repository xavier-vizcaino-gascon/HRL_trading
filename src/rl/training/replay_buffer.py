"""
Replay buffers for storing and sampling experiences.

Implements various replay buffer strategies for RL training:
- Standard replay buffer for off-policy algorithms (DQN, SAC, etc.)
- Rollout buffer for on-policy algorithms (PPO, A2C, etc.)
- Prioritized replay buffer (future extension)
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class Transition:
    """Single transition (s, a, r, s', done)."""

    state: NDArray[np.float32]
    action: NDArray | int
    reward: float
    next_state: NDArray[np.float32]
    done: bool
    info: dict[str, Any] | None = None


@dataclass
class RolloutStep:
    """Single step in on-policy rollout."""

    observation: NDArray[np.float32]
    action: NDArray | int
    reward: float
    done: bool
    value: float  # V(s) from critic
    log_prob: float  # log π(a|s) from policy
    info: dict[str, Any] | None = None


class ReplayBuffer:
    """
    Standard replay buffer for off-policy algorithms.

    Stores transitions and samples random batches for training.
    Inspired by the experienceReplayBuffer from the reference notebooks.
    """

    def __init__(self, capacity: int = 50000, seed: int | None = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = np.random.RandomState(seed)

    def push(
        self,
        state: NDArray,
        action: NDArray | int,
        reward: float,
        next_state: NDArray,
        done: bool,
        info: dict | None = None,
    ) -> None:
        """
        Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            info: Additional information
        """
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary with batched tensors
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough samples in buffer. Have {len(self.buffer)}, need {batch_size}"
            )

        # Sample random indices
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]

        # Stack into batches
        batch = {
            "states": np.stack([t.state for t in transitions]),
            "actions": np.array([t.action for t in transitions]),
            "rewards": np.array([t.reward for t in transitions], dtype=np.float32),
            "next_states": np.stack([t.next_state for t in transitions]),
            "dones": np.array([t.done for t in transitions], dtype=np.float32),
        }

        # Convert to tensors
        return {k: torch.FloatTensor(v) for k, v in batch.items()}

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= min_size

    def get_capacity_ratio(self) -> float:
        """Get buffer fullness ratio."""
        return len(self.buffer) / self.capacity


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO, A2C).

    Stores trajectories collected with current policy and computes
    advantages/returns for training.
    """

    def __init__(
        self,
        capacity: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize rollout buffer.

        Args:
            capacity: Maximum number of steps to store
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for advantage estimation
        """
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage
        self.observations: list[NDArray] = []
        self.actions: list[NDArray | int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []

        # Computed quantities
        self.advantages: NDArray | None = None
        self.returns: NDArray | None = None

    def push(
        self,
        observation: NDArray,
        action: NDArray | int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        Add step to rollout buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            done: Whether episode terminated
            value: Value estimate V(s)
            log_prob: Log probability of action
        """
        if len(self.observations) >= self.capacity:
            raise ValueError(f"Rollout buffer full (capacity: {self.capacity})")

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> None:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        GAE formula:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = δ_t + (γ * λ) * δ_{t+1} + ... + (γ * λ)^{T-t} * δ_T

        Args:
            last_value: Value estimate of final state (0 if terminal)
        """
        n_steps = len(self.rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        returns = np.zeros(n_steps, dtype=np.float32)

        # Append last value for bootstrapping
        values = np.array(self.values + [last_value], dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute advantages using GAE
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )

        # Returns = advantages + values
        returns = advantages + values[:-1]

        self.advantages = advantages
        self.returns = returns

    def get(self) -> dict[str, torch.Tensor]:
        """
        Get all data as batched tensors.

        Returns:
            Dictionary with all rollout data
        """
        if self.advantages is None or self.returns is None:
            raise ValueError("Must call compute_returns_and_advantages() first")

        # Convert actions to array
        actions = np.array(self.actions)
        if len(actions.shape) == 1:
            # Discrete actions, keep as 1D
            pass
        else:
            # Continuous actions, already correct shape
            pass

        batch = {
            "observations": torch.FloatTensor(np.array(self.observations)),
            "actions": (
                torch.FloatTensor(actions)
                if actions.dtype != np.int64
                else torch.LongTensor(actions)
            ),
            "values": torch.FloatTensor(np.array(self.values)),
            "log_probs": torch.FloatTensor(np.array(self.log_probs)),
            "advantages": torch.FloatTensor(self.advantages),
            "returns": torch.FloatTensor(self.returns),
        }

        return batch

    def clear(self) -> None:
        """Clear buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.observations)

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.observations) >= self.capacity


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.

    Samples transitions based on their TD error, giving priority to
    more "surprising" transitions. This can speed up learning.

    Implementation placeholder for future extension.
    """

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta: float = 0.4,
        seed: int | None = None,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (anneals to 1)
            seed: Random seed
        """
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(
        self,
        state: NDArray,
        action: NDArray | int,
        reward: float,
        next_state: NDArray,
        done: bool,
        info: dict | None = None,
    ) -> None:
        """Add transition with maximum priority."""
        super().push(state, action, reward, next_state, done, info)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample batch with prioritized sampling."""
        # For now, fall back to uniform sampling
        # TODO: Implement proper prioritized sampling
        return super().sample(batch_size)

    def update_priorities(self, indices: list[int], td_errors: NDArray) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of sampled transitions
            td_errors: TD errors for each transition
        """
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
