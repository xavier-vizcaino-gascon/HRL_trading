"""Model summary utility for PyTorch nn.Module networks.

Provides a Keras-like model.summary() table showing layer names,
output shapes, and parameter counts for any nn.Module in the project.

Usage examples:

    from src.rl.model_summary import model_summary

    # 03b — ActorCritic (single input)
    model_summary(ac, input_size=(1, OBS_DIM))

    # 03a — SB3 PPO policy
    model_summary(model.policy, input_size=(1, OBS_DIM))

    # 04a/04b — FeudalAgent sub-modules (no forward() on FeudalAgent itself)
    model_summary(agent.perception, input_size=(1, OBS_DIM))
    model_summary(agent.manager, input_size=(1, latent_dim))
    model_summary(agent.worker, input_data=[
        torch.zeros(1, latent_dim), torch.zeros(1, goal_dim)
    ])
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def model_summary(
    model: nn.Module,
    input_size: tuple | list[tuple] | None = None,
    input_data: torch.Tensor | list[torch.Tensor] | None = None,
    depth: int = 3,
    verbose: int = 1,
    col_names: Sequence[str] = ("input_size", "output_size", "num_params", "trainable"),
    col_width: int = 20,
    row_settings: Sequence[str] = ("var_names",),
):
    """Print a Keras-like summary table for any nn.Module.

    Thin wrapper around torchinfo.summary() with project-standard defaults.

    Args:
        model:       any nn.Module — ActorCritic, FeudalAgent sub-module,
                     SB3 model.policy, etc.
        input_size:  tuple or list of tuples for input dimensions (batch dim
                     excluded). E.g. (71,) for single-input, or
                     [(256,), (32,)] for WorkerNetwork(z, g).
        input_data:  alternative to input_size — pass actual torch.Tensor(s).
                     Use when the module expects multiple tensor inputs.
        depth:       levels of nested sub-modules to expand (default 3).
        verbose:     0 = silent, 1 = standard table, 2 = detailed.
        col_names:   columns to display in the table.
        col_width:   character width of each column.
        row_settings: extra row annotations (default shows variable names).

    Returns:
        torchinfo.ModelStatistics object (also printed to stdout).

    Raises:
        ImportError: if torchinfo is not installed
                     (pip install torchinfo  or  uv add torchinfo).
    """
    try:
        from torchinfo import summary
    except ImportError as e:
        raise ImportError(
            "torchinfo is required for model_summary. "
            "Install it with: pip install torchinfo"
        ) from e

    try:
        return summary(
            model,
            input_size=input_size,
            input_data=input_data,
            depth=depth,
            verbose=verbose,
            col_names=col_names,
            col_width=col_width,
            row_settings=row_settings,
        )
    except RuntimeError:
        # Fallback: torchinfo requires forward(); print a manual param table.
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{type(model).__name__} — parameter summary (torchinfo fallback)")
        print(f"{'Module':<40} {'Params':>12} {'Trainable':>12}")
        print("-" * 66)
        for name, m in model.named_children():
            n = sum(p.numel() for p in m.parameters())
            t = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  {name:<38} {n:>12,} {t:>12,}")
        print("-" * 66)
        print(f"  {'Total':<38} {total:>12,} {trainable:>12,}")
        print()
        return None
