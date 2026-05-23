"""Training utilities for RL agents."""

from src.rl.training.replay_buffer import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
    RolloutStep,
    Transition,
)
from src.rl.training.hierarchical_buffer import HierarchicalRolloutBuffer
from src.rl.training.hrl_updates import (
    compute_intrinsic_reward,
    ppo_update_manager,
    ppo_update_worker,
)
from src.rl.training.hrl_callbacks import FeeCurriculumCallback, HrlEvalCallback
from src.rl.training.hrl_training import _make_optimizers, train_hrl
from src.rl.training.synthetic_data import (
    derive_synthetic_features,
    generate_synthetic_prices,
    make_synthetic_df,
    synthetic_ohlc_from_close,
)
from src.rl.training.pretrain_hrl import (
    PreTrainingConfig,
    SyntheticPhaseConfig,
    pretrain_and_train_hrl,
)

__all__ = [
    # Replay buffers
    "ReplayBuffer",
    "RolloutBuffer",
    "PrioritizedReplayBuffer",
    # Data structures
    "Transition",
    "RolloutStep",
    # HRL
    "HierarchicalRolloutBuffer",
    "compute_intrinsic_reward",
    "ppo_update_worker",
    "ppo_update_manager",
    "HrlEvalCallback",
    "FeeCurriculumCallback",
    "_make_optimizers",
    "train_hrl",
    # Synthetic pre-training
    "generate_synthetic_prices",
    "synthetic_ohlc_from_close",
    "derive_synthetic_features",
    "make_synthetic_df",
    "PreTrainingConfig",
    "SyntheticPhaseConfig",
    "pretrain_and_train_hrl",
]
