"""
Reward calculation strategies for trading environments.

This module implements various reward functions for RL trading agents,
including simple P&L, risk-adjusted metrics, and composite rewards.
"""

from typing import Any

import numpy as np

from src.rl.envs.base import RewardCalculator


class PnLReward(RewardCalculator):
    """Simple profit/loss reward - difference in portfolio value."""

    def __init__(self, normalize: bool = True):
        """
        Initialize P&L reward calculator.

        Args:
            normalize: Whether to normalize by previous portfolio value
        """
        self.normalize = normalize

    def calculate(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        action: Any,
        info: dict[str, Any],
    ) -> float:
        """Calculate reward as change in portfolio value."""
        pnl = current_portfolio_value - prev_portfolio_value

        if self.normalize and prev_portfolio_value > 0:
            # Return percentage change
            return (pnl / prev_portfolio_value) * 100
        else:
            # Return absolute change
            return pnl

    def reset(self) -> None:
        """No state to reset."""
        pass


class SharpeReward(RewardCalculator):
    """Risk-adjusted reward using Sharpe ratio calculation."""

    def __init__(self, window_size: int = 100, risk_free_rate: float = 0.0):
        """
        Initialize Sharpe reward calculator.

        Args:
            window_size: Window size for calculating rolling Sharpe
            risk_free_rate: Annual risk-free rate (as decimal)
        """
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.returns_history: list[float] = []

    def calculate(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        action: Any,
        info: dict[str, Any],
    ) -> float:
        """Calculate reward as rolling Sharpe ratio."""
        # Calculate return
        if prev_portfolio_value > 0:
            ret = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            ret = 0.0

        self.returns_history.append(ret)

        # Keep only recent history
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # Calculate Sharpe ratio if we have enough data
        if len(self.returns_history) >= 2:
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return > 0:
                # Annualized Sharpe ratio approximation
                sharpe = (mean_return - self.risk_free_rate) / std_return
                return sharpe
            else:
                return 0.0
        else:
            # Not enough data, return simple return
            return ret

    def reset(self) -> None:
        """Reset returns history."""
        self.returns_history = []


class SortinoReward(RewardCalculator):
    """Downside risk-adjusted reward using Sortino ratio."""

    def __init__(self, window_size: int = 100, target_return: float = 0.0):
        """
        Initialize Sortino reward calculator.

        Args:
            window_size: Window size for calculating rolling Sortino
            target_return: Minimum acceptable return (MAR)
        """
        self.window_size = window_size
        self.target_return = target_return
        self.returns_history: list[float] = []

    def calculate(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        action: Any,
        info: dict[str, Any],
    ) -> float:
        """Calculate reward as rolling Sortino ratio."""
        # Calculate return
        if prev_portfolio_value > 0:
            ret = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            ret = 0.0

        self.returns_history.append(ret)

        # Keep only recent history
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # Calculate Sortino ratio if we have enough data
        if len(self.returns_history) >= 2:
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)

            # Calculate downside deviation (only negative returns)
            downside_returns = returns_array[returns_array < self.target_return]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    sortino = (mean_return - self.target_return) / downside_std
                    return sortino
                else:
                    return mean_return
            else:
                # No downside, return mean return
                return mean_return
        else:
            # Not enough data, return simple return
            return ret

    def reset(self) -> None:
        """Reset returns history."""
        self.returns_history = []


class DirectionalReward(RewardCalculator):
    """
    Reward based on taking correct directional actions.

    Inspired by the notebooks: rewards correct predictions of market direction.
    """

    def __init__(self, reward_correct: float = 1.0, penalty_wrong: float = -1.0):
        """
        Initialize directional reward calculator.

        Args:
            reward_correct: Reward for correct directional action
            penalty_wrong: Penalty for wrong directional action
        """
        self.reward_correct = reward_correct
        self.penalty_wrong = penalty_wrong
        self.prev_price = None

    def calculate(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        action: Any,
        info: dict[str, Any],
    ) -> float:
        """
        Reward correct directional predictions.

        Logic from notebooks:
        - If price went UP and we bought (or held long): reward
        - If price went DOWN and we sold (or stayed flat): reward
        - Otherwise: penalty
        """
        current_price = info.get("current_price", 0.0)
        position = info.get("position", 0)  # -1, 0, 1 for short/flat/long

        if self.prev_price is None:
            self.prev_price = current_price
            return 0.0

        # Determine price direction
        price_change = current_price - self.prev_price

        reward = 0.0

        if price_change > 0:  # Price went UP
            if position > 0:  # We are long
                reward = self.reward_correct
            elif position < 0:  # We are short
                reward = self.penalty_wrong
        elif price_change < 0:  # Price went DOWN
            if position < 0:  # We are short
                reward = self.reward_correct
            elif position > 0:  # We are long
                reward = self.penalty_wrong
        # If price_change == 0, no reward/penalty

        self.prev_price = current_price
        return reward

    def reset(self) -> None:
        """Reset price history."""
        self.prev_price = None


class CompositeReward(RewardCalculator):
    """Weighted combination of multiple reward calculators."""

    def __init__(self, calculators: list[tuple[RewardCalculator, float]]):
        """
        Initialize composite reward calculator.

        Args:
            calculators: List of (calculator, weight) tuples
        """
        self.calculators = calculators

        # Normalize weights
        total_weight = sum(weight for _, weight in calculators)
        if total_weight > 0:
            self.calculators = [(calc, weight / total_weight) for calc, weight in calculators]

    def calculate(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        action: Any,
        info: dict[str, Any],
    ) -> float:
        """Calculate weighted sum of all sub-rewards."""
        total_reward = 0.0

        for calculator, weight in self.calculators:
            reward = calculator.calculate(
                prev_portfolio_value, current_portfolio_value, action, info
            )
            total_reward += reward * weight

        return total_reward

    def reset(self) -> None:
        """Reset all sub-calculators."""
        for calculator, _ in self.calculators:
            calculator.reset()


class CustomReward(RewardCalculator):
    """
    User-defined reward function.

    Allows passing a custom function for reward calculation.
    """

    def __init__(self, reward_fn):
        """
        Initialize custom reward calculator.

        Args:
            reward_fn: Callable with signature:
                (prev_value, current_value, action, info) -> float
        """
        self.reward_fn = reward_fn

    def calculate(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        action: Any,
        info: dict[str, Any],
    ) -> float:
        """Call custom reward function."""
        return self.reward_fn(prev_portfolio_value, current_portfolio_value, action, info)

    def reset(self) -> None:
        """No state to reset."""
        pass
