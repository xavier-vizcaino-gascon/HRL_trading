"""Pre-training orchestration for HRL FeUdal agents with synthetic data.

Provides configurable multi-phase synthetic pre-training before real BTC training.

Usage:
    from src.rl.training.pretrain_hrl import (
        PreTrainingConfig, SyntheticPhaseConfig, pretrain_and_train_hrl
    )

    pretrain_cfg = PreTrainingConfig(
        enabled=True,
        phases=[
            SyntheticPhaseConfig(phase=1, timesteps=50_000),
            SyntheticPhaseConfig(phase=2, timesteps=50_000),
            SyntheticPhaseConfig(phase=3, timesteps=100_000),
        ],
    )

    pretrain_and_train_hrl(
        agent, opt_p, opt_m, opt_w, buffer,
        df_tr, df_vl, total_timesteps, cfg,
        pretrain_cfg, train_hrl_fn, make_env_fn, feature_cols,
        callbacks=[...], writer=writer,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from src.rl.training.synthetic_data import make_synthetic_df

if TYPE_CHECKING:
    import polars as pl
    import torch

    from src.rl.agents.feudal_agent import FeudalAgent
    from src.rl.training.hierarchical_buffer import HierarchicalRolloutBuffer


@dataclass
class SyntheticPhaseConfig:
    """Configuration for one synthetic pre-training phase.

    Attributes:
        phase: 1 (fixed freq sine), 2 (chirp sine), or 3 (fractal sine)
        timesteps: Training steps for this phase
        base_freq: Base frequency in radians/step (phase 1)
        amplitude_pct: Oscillation amplitude relative to base_price
        base_price: Base price level (e.g., 50000 for BTC)
        n_data_steps: Length of synthetic DataFrame (must be >> max_episode_steps)
        freq_sweep_range: (f_min, f_max) for phase 2 chirp
        fractal_freqs: List of frequencies for phase 3 multi-sine
        fractal_amplitudes: List of amplitudes for phase 3 (must match fractal_freqs length)
    """

    phase: int
    timesteps: int
    base_freq: float = 0.02
    amplitude_pct: float = 0.10
    base_price: float = 50_000.0
    n_data_steps: int = 10_000
    # Phase 2 specific
    freq_sweep_range: tuple[float, float] = (0.005, 0.05)
    # Phase 3 specific
    fractal_freqs: list[float] = field(default_factory=lambda: [0.005, 0.02, 0.08, 0.2])
    fractal_amplitudes: list[float] = field(default_factory=lambda: [0.06, 0.03, 0.015, 0.005])


@dataclass
class PreTrainingConfig:
    """Full pre-training configuration.

    Attributes:
        enabled: If False, skip pre-training entirely (direct to real training)
        phases: List of synthetic phases to run sequentially
        reset_optimizers_before_real: If True, recreate optimizers before real training
            (recommended: optimizer momentum from synthetic data may be stale)
    """

    enabled: bool = False
    phases: list[SyntheticPhaseConfig] = field(
        default_factory=lambda: [
            SyntheticPhaseConfig(phase=1, timesteps=50_000),
            SyntheticPhaseConfig(phase=2, timesteps=50_000),
            SyntheticPhaseConfig(phase=3, timesteps=100_000),
        ]
    )
    reset_optimizers_before_real: bool = True


def pretrain_and_train_hrl(
    agent: "FeudalAgent",
    opt_perception: "torch.optim.Optimizer",
    opt_manager: "torch.optim.Optimizer",
    opt_worker: "torch.optim.Optimizer",
    buffer: "HierarchicalRolloutBuffer",
    df_tr: "pl.DataFrame",
    df_vl: "pl.DataFrame",
    total_timesteps: int,
    cfg: dict,
    pretrain_cfg: PreTrainingConfig,
    train_hrl_fn: Callable,
    make_env_fn: Callable | None,
    feature_cols: list[str],
    callbacks: "list | None" = None,
    writer=None,
    verbose: bool = True,
    log_project: "str | None" = None,
    log_dir=None,
) -> list:
    """Run synthetic pre-training phases then real training.

    If pretrain_cfg.enabled is False, delegates directly to train_hrl_fn.

    If True:
      1. For each phase in pretrain_cfg.phases:
         - Generate synthetic DataFrame via make_synthetic_df()
         - Call train_hrl_fn with synthetic df, alpha=0, no callbacks (no eval/pruning)
         - Agent weights preserved across phases
      2. Optionally reset optimizer states before real training
      3. Call train_hrl_fn with real df_tr/df_vl for total_timesteps with full callbacks

    Args:
        agent: FeudalAgent instance (weights preserved across all phases)
        opt_perception, opt_manager, opt_worker: Optimizers (may be reset before real training)
        buffer: HierarchicalRolloutBuffer (reused across phases)
        df_tr, df_vl: Real training/validation DataFrames (used only in final phase)
        total_timesteps: Training steps for REAL data (synthetic steps defined in pretrain_cfg)
        cfg: Hyperparameter dict (passed to train_hrl_fn)
        pretrain_cfg: Pre-training configuration
        train_hrl_fn: The actual train_hrl function to call (notebook or library version)
        make_env_fn: Env factory (if None, assumes train_hrl_fn hardcodes env creation)
        feature_cols: List of Norm_* column names (from dm.features)
        callbacks: Callbacks for REAL training only (synthetic phases use no callbacks)
        writer: TensorBoard SummaryWriter (used for all phases)
        verbose: Print progress
        log_project, log_dir: Logging config (passed to train_hrl_fn)

    Returns:
        Result of final train_hrl_fn call (typically empty list, eval_cb has history)

    Notes:
        - Synthetic phases use alpha=0 (pure intrinsic reward) to force Manager-Worker communication
        - No eval/pruning during synthetic phases (metrics are meaningless for Optuna)
        - Optimizer momentum from synthetic data is typically stale → reset before real training
    """
    if not pretrain_cfg.enabled:
        # No pre-training: delegate directly to real training
        if verbose:
            print("Pre-training disabled. Starting real training directly.")
        return train_hrl_fn(
            agent,
            opt_perception,
            opt_manager,
            opt_worker,
            buffer,
            df_tr,
            df_vl,
            total_timesteps,
            cfg,
            make_env_fn,  # Pass through make_env_fn
            callbacks=callbacks,
            writer=writer,
            verbose=verbose,
            log_project=log_project,
            log_dir=log_dir,
        )

    # Pre-training enabled: run synthetic phases
    if verbose:
        print(f"\n{'='*60}")
        print(f"PRE-TRAINING: {len(pretrain_cfg.phases)} synthetic phases")
        print(f"{'='*60}")

    for i, phase_cfg in enumerate(pretrain_cfg.phases, 1):
        if verbose:
            print(f"\n[Phase {i}/{len(pretrain_cfg.phases)}] "
                  f"Synthetic phase {phase_cfg.phase} — {phase_cfg.timesteps:,} steps")

        # Generate synthetic DataFrame
        synth_df = make_synthetic_df(
            n_steps=phase_cfg.n_data_steps,
            phase=phase_cfg.phase,
            feature_cols=feature_cols,
            base_price=phase_cfg.base_price,
            amplitude_pct=phase_cfg.amplitude_pct,
            base_freq=phase_cfg.base_freq,
            freq_sweep_range=phase_cfg.freq_sweep_range,
            fractal_freqs=phase_cfg.fractal_freqs,
            fractal_amplitudes=phase_cfg.fractal_amplitudes,
            seed=cfg.get("seed", 42) + i,  # Different seed per phase
        )

        # Construct synthetic training config: alpha=0 (pure intrinsic reward)
        cfg_synth = cfg.copy()
        cfg_synth["alpha"] = 0.0

        if verbose:
            print(f"  Synthetic df: {len(synth_df):,} rows, {len(feature_cols)} features")
            print(f"  Config: alpha=0.0 (pure intrinsic reward)")

        # Call train_hrl with synthetic data, NO callbacks (no eval/pruning)
        train_hrl_fn(
            agent,
            opt_perception,
            opt_manager,
            opt_worker,
            buffer,
            synth_df,  # Use synthetic as df_tr
            synth_df,  # Use synthetic as df_vl (not used, no eval callback)
            phase_cfg.timesteps,
            cfg_synth,
            make_env_fn,  # Pass through make_env_fn
            callbacks=[],  # NO callbacks during synthetic phases
            writer=writer,  # Still log training losses
            verbose=verbose,
            log_project=None,  # No CSV logging for synthetic
            log_dir=None,
        )

        if verbose:
            print(f"  [OK] Phase {i} complete. Agent weights updated in-place.")

    # Reset optimizers before real training (Adam momentum from synthetic is stale)
    if pretrain_cfg.reset_optimizers_before_real:
        if verbose:
            print(f"\n{'='*60}")
            print("Resetting optimizer states before real training...")
            print(f"{'='*60}")
        # Recreate optimizers with same hyperparams
        from src.rl.training.hrl_training import _make_optimizers

        lr = cfg.get("learning_rate", 1e-4)
        manager_lr = cfg.get("manager_lr", 1e-4)
        weight_decay = cfg.get("weight_decay", 0.0)
        opt_perception, opt_manager, opt_worker = _make_optimizers(
            agent, lr, manager_lr, weight_decay
        )

    # Real training with full callbacks
    if verbose:
        print(f"\n{'='*60}")
        print(f"REAL TRAINING: {total_timesteps:,} steps on BTC data")
        print(f"{'='*60}\n")

    return train_hrl_fn(
        agent,
        opt_perception,
        opt_manager,
        opt_worker,
        buffer,
        df_tr,
        df_vl,
        total_timesteps,
        cfg,
        make_env_fn,  # Pass through make_env_fn
        callbacks=callbacks,
        writer=writer,
        verbose=verbose,
        log_project=log_project,
        log_dir=log_dir,
    )
