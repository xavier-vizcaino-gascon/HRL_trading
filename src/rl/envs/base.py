"""
Base classes for trading environments.

This module provides the foundational abstractions for creating Gymnasium-compatible
trading environments, including observation space, action space, and reward calculators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from src.common.events import EventType, publish_event


@dataclass
class ObservationSpace:
    """Wrapper for observation space configuration."""

    features: int  # Number of technical features
    portfolio_state_size: int = 3  # cash, position, value
    low: float = 0.0
    high: float = np.inf

    @property
    def total_size(self) -> int:
        """Total observation space size."""
        return self.features + self.portfolio_state_size

    def to_gym_space(self) -> gym.Space:
        """Convert to Gymnasium Box space."""
        return gym.spaces.Box(
            low=self.low,
            high=self.high,
            shape=(self.total_size,),
            dtype=np.float32,
        )


@dataclass
class ActionSpace:
    """Wrapper for action space configuration."""

    action_type: str  # 'discrete' or 'continuous'
    n_actions: int | None = None  # For discrete: number of actions
    action_low: float | None = None  # For continuous: lower bound
    action_high: float | None = None  # For continuous: upper bound
    action_shape: tuple[int, ...] | None = None  # For continuous: shape

    def to_gym_space(self) -> gym.Space:
        """Convert to Gymnasium space."""
        if self.action_type == "discrete":
            if self.n_actions is None:
                raise ValueError("n_actions must be specified for discrete action space")
            return gym.spaces.Discrete(self.n_actions)
        elif self.action_type == "continuous":
            if self.action_low is None or self.action_high is None:
                raise ValueError(
                    "action_low and action_high must be specified for continuous action space"
                )
            shape = self.action_shape or (1,)
            return gym.spaces.Box(
                low=self.action_low,
                high=self.action_high,
                shape=shape,
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")


class RewardCalculator(ABC):
    """Abstract base class for reward calculation strategies."""

    @abstractmethod
    def calculate(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        action: Any,
        info: dict[str, Any],
    ) -> float:
        """
        Calculate reward for a step.

        Args:
            prev_portfolio_value: Portfolio value before action
            current_portfolio_value: Portfolio value after action
            action: Action taken by agent
            info: Additional information dictionary

        Returns:
            Calculated reward value
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state of the reward calculator."""
        pass


class BaseTradingEnvironment(gym.Env, ABC):
    """
    Abstract base class for all trading environments.

    This class provides common functionality for trading environments including:
    - Gymnasium compatibility
    - Event publishing for auditability
    - Episode management
    - Observation/action validation
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        observation_config: ObservationSpace,
        action_config: ActionSpace,
        reward_calculator: RewardCalculator,
        initial_balance: float = 10000.0,
        max_steps: int | None = None,
    ):
        """
        Initialize the trading environment.

        Args:
            observation_config: Configuration for observation space
            action_config: Configuration for action space
            reward_calculator: Reward calculation strategy
            initial_balance: Starting capital
            max_steps: Maximum steps per episode (None for unlimited)
        """
        super().__init__()

        self.observation_config = observation_config
        self.action_config = action_config
        self.reward_calculator = reward_calculator
        self.initial_balance = initial_balance
        self.max_steps = max_steps

        # Define Gymnasium spaces
        self.observation_space = observation_config.to_gym_space()
        self.action_space = action_config.to_gym_space()

        # Episode state
        self.current_step = 0
        self.portfolio_value = initial_balance
        self.prev_portfolio_value = initial_balance
        self.episode_id = 0

        # Metrics tracking
        self.episode_trades = 0
        self.episode_profit = 0.0

    @abstractmethod
    def _get_observation(self) -> NDArray[np.float32]:
        """
        Get current observation.

        Returns:
            Observation array matching observation_space
        """
        pass

    @abstractmethod
    def _execute_action(self, action: Any) -> dict[str, Any]:
        """
        Execute the given action in the environment.

        Args:
            action: Action from agent (discrete or continuous)

        Returns:
            Info dictionary with execution details
        """
        pass

    @abstractmethod
    def _update_portfolio_value(self) -> float:
        """
        Calculate and return current portfolio value.

        Returns:
            Current total portfolio value
        """
        pass

    @abstractmethod
    def _is_terminal(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if episode is done, False otherwise
        """
        pass

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.episode_id += 1
        self.episode_trades = 0
        self.episode_profit = 0.0

        # Reset reward calculator
        self.reward_calculator.reset()

        # Publish reset event
        publish_event(
            EventType.EPISODE_START,
            {
                "episode_id": self.episode_id,
                "initial_balance": self.initial_balance,
            },
        )

        observation = self._get_observation()
        info = {"episode_id": self.episode_id}

        return observation, info

    def step(self, action: Any) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space")

        # Store previous value
        self.prev_portfolio_value = self.portfolio_value

        # Execute action
        action_info = self._execute_action(action)

        # Update portfolio value
        self.portfolio_value = self._update_portfolio_value()

        # Calculate reward
        reward = self.reward_calculator.calculate(
            prev_portfolio_value=self.prev_portfolio_value,
            current_portfolio_value=self.portfolio_value,
            action=action,
            info=action_info,
        )

        # Get next observation
        observation = self._get_observation()

        # Check termination
        terminated = self._is_terminal()
        truncated = self.max_steps is not None and self.current_step >= self.max_steps

        # Update step counter
        self.current_step += 1

        # Compile info
        info = {
            "episode_id": self.episode_id,
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "profit": self.portfolio_value - self.initial_balance,
            "profit_pct": ((self.portfolio_value - self.initial_balance) / self.initial_balance)
            * 100,
            **action_info,
        }

        # Publish step event
        publish_event(
            EventType.RL_STEP,
            {
                "episode_id": self.episode_id,
                "step": self.current_step,
                "action": action,
                "reward": reward,
                "portfolio_value": self.portfolio_value,
            },
        )

        # If episode ended, publish end event
        if terminated or truncated:
            self.episode_profit = self.portfolio_value - self.initial_balance
            publish_event(
                EventType.EPISODE_END,
                {
                    "episode_id": self.episode_id,
                    "total_steps": self.current_step,
                    "final_value": self.portfolio_value,
                    "total_profit": self.episode_profit,
                    "total_trades": self.episode_trades,
                    "terminated": terminated,
                    "truncated": truncated,
                },
            )

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment (optional, for debugging)."""
        if self.current_step % 100 == 0:
            profit_pct = (
                (self.portfolio_value - self.initial_balance) / self.initial_balance
            ) * 100
            print(
                f"Step {self.current_step}: "
                f"Portfolio Value: ${self.portfolio_value:.2f} "
                f"(Profit: {profit_pct:.2f}%)"
            )

    def close(self) -> None:
        """Clean up resources."""
        pass
