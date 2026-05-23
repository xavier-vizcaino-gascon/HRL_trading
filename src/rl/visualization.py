"""Standard RL visualization plots shared across all agent notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_optuna_search(
    study: Any,
    title_prefix: str = "",
    color: str = "#2ca02c",
    save_path: Path | None = None,
) -> plt.Figure:
    """1×2 subplot: trial return distribution + Optuna convergence curve.

    Args:
        study:        completed optuna.Study object.
        title_prefix: prepended to subplot titles (e.g. "SB3 PPO").
        color:        bar/scatter color (default green).
        save_path:    if provided, saves figure to this path (dpi=120).

    Returns:
        matplotlib Figure.
    """
    trial_values = [
        t.value for t in study.trials
        if t.value is not None and np.isfinite(t.value)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    prefix = f"{title_prefix} — " if title_prefix else ""

    # Histograma de returns
    axes[0].hist(trial_values, bins=15, color=color, edgecolor="white")
    axes[0].axvline(
        study.best_value, color="red", linestyle="--",
        label=f"Best: {study.best_value:.3f}%",
    )
    axes[0].set_xlabel("Return (%) (val)")
    axes[0].set_ylabel("Trials")
    axes[0].set_title(f"{prefix}Distribució Return (%) per trial")
    axes[0].legend()

    # Convergència (best-so-far)
    trial_ns = list(range(len(trial_values)))
    axes[1].scatter(trial_ns, trial_values, alpha=0.6, color=color, s=30)
    axes[1].plot(
        trial_ns, np.maximum.accumulate(trial_values),
        color="red", linewidth=2, label="Best so far",
    )
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Return (%) (val)")
    axes[1].set_title(f"{prefix}Convergència Optuna")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_training_curves(
    eval_history: list[dict],
    title_prefix: str = "",
    total_timesteps: int | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """2×2 subplot: Sharpe, return, drawdown, trades vs timesteps.

    Args:
        eval_history:     list of dicts with keys: timestep, sharpe,
                          mean_return, max_drawdown, mean_trades.
        title_prefix:     prepended to suptitle (e.g. "SB3 PPO").
        total_timesteps:  shown in suptitle if provided.
        save_path:        if provided, saves figure to this path (dpi=120).

    Returns:
        matplotlib Figure, or None if eval_history is empty.
    """
    if not eval_history:
        print("Sense historial d'avaluació.")
        return None

    ts_vals  = [h["timestep"]     for h in eval_history]
    sh_vals  = [h["sharpe"]       for h in eval_history]
    ret_vals = [h["mean_return"]  for h in eval_history]
    dd_vals  = [h["max_drawdown"] for h in eval_history]
    tr_vals  = [h["mean_trades"]  for h in eval_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(ts_vals, sh_vals, color="#2ca02c", linewidth=2)
    axes[0, 0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Sharpe ratio (val)")
    axes[0, 0].set_xlabel("Timesteps")

    axes[0, 1].plot(ts_vals, ret_vals, color="#1f77b4", linewidth=2)
    axes[0, 1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Retorn mig (val, %)")
    axes[0, 1].set_xlabel("Timesteps")

    axes[1, 0].plot(ts_vals, dd_vals, color="#d62728", linewidth=2)
    axes[1, 0].set_title("Max Drawdown (val, %)")
    axes[1, 0].set_xlabel("Timesteps")

    axes[1, 1].plot(ts_vals, tr_vals, color="#ff7f0e", linewidth=2)
    axes[1, 1].set_title("Trades mig per episodi (val)")
    axes[1, 1].set_xlabel("Timesteps")

    ts_str = f" — {total_timesteps:,} ts" if total_timesteps else ""
    plt.suptitle(f"Entrenament {title_prefix}{ts_str}", fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_walk_forward(
    wf_results: dict,
    wf_metrics: dict,
    initial_balance: float,
    title_prefix: str = "",
    agent_label: str = "Agent",
    save_path: Path | None = None,
    action_save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Walk-forward visualization: equity curve, drawdown, episode returns + action sequence.

    Args:
        wf_results:       dict with keys: equities, prices, actions, ep_returns.
        wf_metrics:       dict with keys: total_return, sharpe, max_drawdown, action_dist_pct.
        initial_balance:  starting portfolio value (for B&H reference line).
        title_prefix:     model name shown in suptitle (e.g. "SB3 PPO").
        agent_label:      legend label for the equity curve (e.g. "HRL FeUdal").
        save_path:        if provided, saves main figure (dpi=120).
        action_save_path: if provided, saves action sequence figure (dpi=120).

    Returns:
        (fig_main, fig_actions) — both matplotlib Figures.
    """
    eq     = np.array(wf_results["equities"])
    pr     = np.array(wf_results["prices"])
    ep_ret = np.array(wf_results["ep_returns"])
    actions = wf_results.get("actions", [])
    act_d   = wf_metrics.get("action_dist_pct", {})

    # ── Main figure: 3×1 (equity+B&H, drawdown, episode returns) ──────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    # Equity curve + Buy & Hold
    axes[0].plot(eq, color="royalblue", lw=1.2, label=agent_label)
    if len(pr) > 0:
        bh = initial_balance * pr / pr[0]
        axes[0].plot(bh, color="gray", lw=1.0, linestyle="--", alpha=0.7,
                     label="Buy & Hold")
    axes[0].axhline(initial_balance, color="black", lw=0.8, linestyle=":")
    axes[0].set_title(f"Corba d'equitat — Walk-Forward Test ({title_prefix})")
    axes[0].set_ylabel("Valor cartera ($)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Drawdown
    run = np.maximum.accumulate(eq)
    dd  = (eq - run) / (run + 1e-10) * 100
    axes[1].fill_between(range(len(dd)), dd, 0, color="crimson", alpha=0.55)
    axes[1].set_title("Drawdown (%)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(alpha=0.3)

    # Episode returns
    colors_ep = ["seagreen" if r >= 0 else "crimson" for r in ep_ret]
    axes[2].bar(range(len(ep_ret)), ep_ret, color=colors_ep, alpha=0.75, width=0.8)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Retorn per episodi (%)")
    axes[2].set_xlabel("Episodi")
    axes[2].set_ylabel("Retorn (%)")
    axes[2].grid(axis="y", alpha=0.3)

    m = wf_metrics
    plt.suptitle(
        f"{title_prefix} — Walk-Forward | "
        f"Ret: {m.get('total_return', 0):+.2f}% | "
        f"Sharpe: {m.get('sharpe', 0):.3f} | "
        f"MaxDD: {m.get('max_drawdown', 0):.2f}%",
        fontsize=12,
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")

    # ── Action sequence figure ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    if actions:
        colors_map = {0: "#7f7f7f", 1: "#2ca02c", 2: "#d62728", 3: "#ff7f0e"}
        ax2.scatter(
            range(len(actions)), actions,
            c=[colors_map.get(a, "#888888") for a in actions],
            s=1, alpha=0.3,
        )
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(["HOLD", "LONG", "SHORT", "CLOSE"])
        ax2.set_title(f"Seqüència d'accions — {title_prefix}")
        ax2.set_xlabel("Pas")
    elif act_d:
        # Fallback: pie chart of action distribution
        colors_pie = ["#4a9eff", "#2ecc71", "#e74c3c", "#f39c12"]
        ax2.pie(
            list(act_d.values()),
            labels=list(act_d.keys()),
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        ax2.set_title(f"Distribució d'accions — {title_prefix}")

    plt.tight_layout()
    if action_save_path is not None:
        plt.savefig(action_save_path, dpi=120, bbox_inches="tight")

    return fig, fig2
