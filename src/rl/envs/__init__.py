"""RL environments for trading."""

from src.rl.envs.base import (
    ActionSpace,
    BaseTradingEnvironment,
    ObservationSpace,
    RewardCalculator,
)
from src.rl.envs.rewards import (
    CompositeReward,
    CustomReward,
    DirectionalReward,
    PnLReward,
    SharpeReward,
    SortinoReward,
)
from src.rl.envs.crypto_market_env import (
    CryptoMarketEnv,
    PositionSide,
    make_env,
    TAKER_FEE,
    MAKER_FEE,
    MARGIN_OPENING_LONG,
    MARGIN_OPENING_SHORT,
    ROLLOVER_LONG_4H,
    ROLLOVER_SHORT_4H,
    MAX_LEVERAGE,
    STEPS_PER_4H,
    INITIAL_BALANCE,
    MAX_EPISODE_STEPS,
    W_PNL,
    W_SORTINO,
    W_COST,
    W_DD,
    SORTINO_WINDOW,
)
from src.rl.envs.crypto_market_env_consolidated_reward import (
    CryptoMarketEnvConsolidatedReward,
    make_env_consolidated,
)
from src.rl.envs.single_asset_env import SingleAssetConfig, SingleAssetTradingEnv

__all__ = [
    # Base classes
    "BaseTradingEnvironment",
    "ObservationSpace",
    "ActionSpace",
    "RewardCalculator",
    # Reward calculators
    "PnLReward",
    "SharpeReward",
    "SortinoReward",
    "DirectionalReward",
    "CompositeReward",
    "CustomReward",
    # Environments
    "SingleAssetTradingEnv",
    "SingleAssetConfig",
    "CryptoMarketEnv",
    "CryptoMarketEnvConsolidatedReward",
    "PositionSide",
    "make_env",
    "make_env_consolidated",
    # Constants (fees + env defaults)
    "TAKER_FEE",
    "MAKER_FEE",
    "MARGIN_OPENING_LONG",
    "MARGIN_OPENING_SHORT",
    "ROLLOVER_LONG_4H",
    "ROLLOVER_SHORT_4H",
    "MAX_LEVERAGE",
    "STEPS_PER_4H",
    "INITIAL_BALANCE",
    "MAX_EPISODE_STEPS",
    "W_PNL",
    "W_SORTINO",
    "W_COST",
    "W_DD",
    "SORTINO_WINDOW",
]
