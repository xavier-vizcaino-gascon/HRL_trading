"""
Single asset trading environment.

This environment simulates trading a single asset (e.g., BTC/USD) with
discrete or continuous action spaces, technical indicators, and portfolio management.

Inspired by the reference notebooks but adapted to the project's architecture.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray

from src.rl.envs.base import (
    ActionSpace,
    BaseTradingEnvironment,
    ObservationSpace,
    RewardCalculator,
)


@dataclass
class SingleAssetConfig:
    """Configuration for single asset trading environment."""

    symbol: str
    initial_balance: float = 10000.0
    max_position_size: float = 1.0  # Fraction of portfolio
    transaction_cost_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    action_type: str = "discrete"  # 'discrete' or 'continuous'
    normalize_observations: bool = True
    max_drawdown_pct: float | None = None  # Stop if drawdown exceeds this


class SingleAssetTradingEnv(BaseTradingEnvironment):
    """
    Trading environment for a single asset.

    Observation space:
        - Price features (normalized)
        - Technical indicators (from feature pipeline)
        - Portfolio state (cash, position, value)

    Action space:
        Discrete: [0=SELL, 1=HOLD, 2=BUY] or [0=HOLD, 1=BUY, 2=SELL]
        Continuous: [-1, 1] where -1=full short, 0=flat, 1=full long

    Reward:
        Configurable via RewardCalculator (P&L, Sharpe, directional, etc.)
    """

    def __init__(
        self,
        config: SingleAssetConfig,
        data: pl.DataFrame,
        reward_calculator: RewardCalculator,
        features: list[str] | None = None,
    ):
        """
        Initialize single asset trading environment.

        Args:
            config: Environment configuration
            data: Polars DataFrame with OHLCV data and technical indicators
            reward_calculator: Reward calculation strategy
            features: List of column names to use as features
                     If None, uses all numeric columns except OHLCV
        """
        self.config = config
        self.data = data
        self.n_steps = len(data)

        # Determine features to use
        if features is None:
            # Use all numeric columns except basic OHLCV
            exclude_cols = {"open", "high", "low", "close", "volume", "timestamp"}
            self.features = [
                col for col in data.select(cs.numeric()).columns if col.lower() not in exclude_cols
            ]
        else:
            self.features = features

        self.n_features = len(self.features)

        # Setup observation space
        # Features + [cash, position, portfolio_value]
        obs_config = ObservationSpace(
            features=self.n_features, portfolio_state_size=3, low=0.0, high=np.inf
        )

        # Setup action space
        if config.action_type == "discrete":
            action_config = ActionSpace(action_type="discrete", n_actions=3)
        elif config.action_type == "continuous":
            action_config = ActionSpace(
                action_type="continuous",
                action_low=-1.0,
                action_high=1.0,
                action_shape=(1,),
            )
        else:
            raise ValueError(f"Unknown action_type: {config.action_type}")

        # Initialize parent
        super().__init__(
            observation_config=obs_config,
            action_config=action_config,
            reward_calculator=reward_calculator,
            initial_balance=config.initial_balance,
            max_steps=self.n_steps,
        )

        # Trading state
        self.cash = config.initial_balance
        self.position = 0.0  # Number of units held
        self.position_value = 0.0
        self.entry_price = 0.0
        self.total_trades = 0

        # Tracking
        self.max_portfolio_value = config.initial_balance
        self.trade_history: list[dict] = []

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float32], dict]:
        """Reset environment to initial state."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset trading state
        self.current_step = 0
        self.cash = self.config.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.max_portfolio_value = self.config.initial_balance
        self.trade_history = []

        # Update observation and portfolio value
        self.portfolio_value = self._update_portfolio_value()
        obs = self._get_observation()

        return obs, info

    def _get_observation(self) -> NDArray[np.float32]:
        """
        Get current observation.

        Returns:
            Array with [features..., cash_norm, position_norm, value_norm]
        """
        if self.current_step >= self.n_steps:
            self.current_step = self.n_steps - 1

        # Get feature values for current step
        feature_values = np.array(
            self.data.select(self.features).row(self.current_step),
            dtype=np.float32,
        )

        # Handle NaN/inf values
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalize features if configured
        if self.config.normalize_observations:
            # Simple min-max normalization to [0, 1]
            feature_values = np.clip(feature_values, -1e6, 1e6)
            feature_min = feature_values.min() if len(feature_values) > 0 else 0
            feature_max = feature_values.max() if len(feature_values) > 0 else 1
            if feature_max > feature_min:
                feature_values = (feature_values - feature_min) / (feature_max - feature_min)

        # Portfolio state (normalized)
        cash_norm = self.cash / self.config.initial_balance
        position_norm = self.position / (self.config.initial_balance / self._get_current_price())
        value_norm = self.portfolio_value / self.config.initial_balance

        portfolio_state = np.array([cash_norm, position_norm, value_norm], dtype=np.float32)

        # Concatenate
        observation = np.concatenate([feature_values, portfolio_state])

        return observation

    def _execute_action(self, action: int | NDArray) -> dict:
        """
        Execute trading action.

        Args:
            action: Discrete (0/1/2) or continuous ([-1, 1])

        Returns:
            Info dictionary with execution details
        """
        current_price = self._get_current_price()

        # Parse action
        if self.config.action_type == "discrete":
            # 0=SELL, 1=HOLD, 2=BUY
            if action == 0:
                target_position = -1.0  # Sell/short
            elif action == 1:
                target_position = 0.0  # Hold/flat
            else:  # action == 2
                target_position = 1.0  # Buy/long
        else:  # continuous
            # action is already in [-1, 1]
            target_position = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # Calculate target position size in units
        max_units = self.config.max_position_size * self.portfolio_value / current_price
        target_units = target_position * max_units

        # Calculate trade needed
        trade_units = target_units - self.position
        trade_value = abs(trade_units) * current_price

        # Apply transaction costs and slippage
        if trade_units != 0:
            cost_pct = self.config.transaction_cost_pct + self.config.slippage_pct
            transaction_cost = trade_value * cost_pct

            # Check if we have enough cash to buy
            if trade_units > 0:  # Buying
                required_cash = trade_value + transaction_cost
                if required_cash > self.cash:
                    # Can't afford full position, reduce it
                    affordable_units = (self.cash - transaction_cost) / (
                        current_price * (1 + cost_pct)
                    )
                    trade_units = max(0, affordable_units)
                    trade_value = trade_units * current_price
                    transaction_cost = trade_value * cost_pct

            # Execute trade
            if abs(trade_units) > 1e-6:  # Avoid tiny trades
                self.cash -= trade_units * current_price + transaction_cost
                self.position += trade_units

                if self.position != 0:
                    self.entry_price = current_price

                self.total_trades += 1
                self.trade_history.append(
                    {
                        "step": self.current_step,
                        "action": action,
                        "trade_units": trade_units,
                        "price": current_price,
                        "cost": transaction_cost,
                    }
                )

        # Update position value
        self.position_value = self.position * current_price

        return {
            "action": action,
            "current_price": current_price,
            "position": self.position,
            "cash": self.cash,
            "trade_units": trade_units,
            "transaction_cost": transaction_cost if trade_units != 0 else 0.0,
        }

    def _update_portfolio_value(self) -> float:
        """Calculate current total portfolio value."""
        current_price = self._get_current_price()
        self.position_value = self.position * current_price
        portfolio_value = self.cash + self.position_value

        # Update max value for drawdown calculation
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value

        return portfolio_value

    def _is_terminal(self) -> bool:
        """
        Check if episode should terminate.

        Terminates if:
        - Max drawdown exceeded
        - Portfolio value too low
        - End of data
        """
        if self.current_step >= self.n_steps - 1:
            return True

        # Check max drawdown
        if self.config.max_drawdown_pct is not None:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if drawdown > self.config.max_drawdown_pct:
                return True

        # Check if portfolio value dropped too much (default: 50% loss)
        if self.portfolio_value < self.config.initial_balance * 0.5:
            return True

        return False

    def _get_current_price(self) -> float:
        """Get current close price."""
        if self.current_step >= self.n_steps:
            self.current_step = self.n_steps - 1

        return float(self.data["close"][self.current_step])

    def get_metrics(self) -> dict:
        """
        Get performance metrics for current episode.

        Returns:
            Dictionary with various performance metrics
        """
        total_return = (
            (self.portfolio_value - self.config.initial_balance) / self.config.initial_balance
        ) * 100

        drawdown = (
            ((self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value) * 100
            if self.max_portfolio_value > 0
            else 0
        )

        return {
            "total_return_pct": total_return,
            "final_value": self.portfolio_value,
            "total_trades": self.total_trades,
            "max_drawdown_pct": drawdown,
            "cash": self.cash,
            "position": self.position,
        }
