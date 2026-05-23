"""Avaluació d'agents HRL FeUdal: evaluate_agent i walk_forward_eval_hrl.

Totes dues funcions reben make_env_fn com a paràmetre per desacoblar-les
del DataFrame de configuració de l'entorn (feature_cols, timeframe, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from src.rl.envs.crypto_market_env import MAX_EPISODE_STEPS
from src.rl.metrics import compute_max_drawdown, compute_sharpe

if TYPE_CHECKING:
    import polars as pl

    from src.rl.agents.feudal_agent import FeudalAgent


def evaluate_agent(
    agent: "FeudalAgent",
    df: "pl.DataFrame",
    n_episodes: int,
    device: "torch.device | str",
    c: int,
    make_env_fn: Callable,
    seed: int = 42,
    deterministic: bool = True,
) -> dict:
    """Avaluació de l'agent HRL sobre n_episodes episodis amb inici aleatori.

    Args:
        agent:        FeudalAgent en mode eval
        df:           DataFrame de dades (train / val / test)
        n_episodes:   nombre d'episodis
        device:       dispositiu torch
        c:            periode del Manager (passos entre goals)
        make_env_fn:  callable(df) -> CryptoMarketEnv
        seed:         seed base per a reproducibilitat
        deterministic: si True, accions deterministes (argmax)

    Returns:
        dict amb mean_return, std_return, sharpe, max_drawdown,
             mean_trades, position_frac, action_dist_pct
    """
    rng = np.random.default_rng(seed)
    env = make_env_fn(df)
    agent.eval()
    all_returns, all_equities, all_trades, all_pos_sides, all_actions = [], [], [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            ep_eq = [info["equity"]]
            done = False
            step_ep = 0
            cur_goal = None

            while not done:
                action, cur_goal = agent.predict_step(
                    obs, step_ep, c, cur_goal, device, deterministic=deterministic
                )
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
                ep_eq.append(info["equity"])
                all_pos_sides.append(info["pos_side"])
                all_actions.append(action)
                step_ep += 1

            all_returns.append((ep_eq[-1] / ep_eq[0] - 1) * 100)
            all_equities.append(ep_eq)
            all_trades.append(info["n_trades"])

    agent.train()
    n_total       = len(all_pos_sides)
    n_actions     = len(all_actions)
    position_frac = sum(1 for p in all_pos_sides if p != 0) / max(n_total, 1)
    action_dist_pct = {
        "HOLD":  all_actions.count(0) / max(n_actions, 1) * 100,
        "LONG":  all_actions.count(1) / max(n_actions, 1) * 100,
        "SHORT": all_actions.count(2) / max(n_actions, 1) * 100,
        "CLOSE": all_actions.count(3) / max(n_actions, 1) * 100,
    }
    return {
        "mean_return":    float(np.mean(all_returns)),
        "std_return":     float(np.std(all_returns)),
        "sharpe":         float(np.mean([compute_sharpe(eq) for eq in all_equities])),
        "max_drawdown":   float(compute_max_drawdown([e for eq in all_equities for e in eq])),
        "mean_trades":    float(np.mean(all_trades)),
        "position_frac":  position_frac,
        "action_dist_pct": action_dist_pct,
    }


def walk_forward_eval_hrl(
    agent: "FeudalAgent",
    df: "pl.DataFrame",
    device: "torch.device | str",
    c: int,
    make_env_fn: Callable,
    ep_steps: int = MAX_EPISODE_STEPS,
    seed: int = 42,
    log_project: "str | None" = None,
    log_dir: "Path | None" = None,
) -> dict:
    """Avaluació walk-forward seqüencial per a FeudalAgent.

    Args:
        agent:       FeudalAgent
        df:          DataFrame de dades (normalment test split)
        device:      dispositiu torch
        c:           periode del Manager
        make_env_fn: callable(df, max_episode_steps=...) -> CryptoMarketEnv
        ep_steps:    passos per episodi
        seed:        seed per a reset inicial (no s'usa en walk-forward pur)
        log_project: nom de projecte per a CSV logging opcional
        log_dir:     directori de logging opcional

    Returns:
        dict amb equities, prices, actions, ep_returns i metrics
        (total_return, sharpe, max_drawdown, n_episodes, n_steps, action_dist_pct)
    """
    n_rows = len(df)
    all_equities, all_actions, all_prices, ep_returns = [], [], [], []
    start  = 0
    agent.eval()
    env = make_env_fn(df, max_episode_steps=ep_steps)

    if log_project is not None and log_dir is not None:
        lp = env.enable_log(project=log_project, log_dir=log_dir)
        print(f"Log CSV (walk-forward): {lp}")

    with torch.no_grad():
        while start + ep_steps + 1 < n_rows:
            obs, info = env.reset(options={"episode_start": start})
            ep_eq = [info["equity"]]
            done = False
            step_ep = 0
            cur_goal = None

            while not done:
                action, cur_goal = agent.predict_step(
                    obs, step_ep, c, cur_goal, device, deterministic=True
                )
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
                ep_eq.append(info["equity"])
                all_actions.append(action)
                all_prices.append(info.get("current_price", float("nan")))
                step_ep += 1

            all_equities.extend(ep_eq)
            ep_returns.append((ep_eq[-1] / ep_eq[0] - 1) * 100)
            start += len(ep_eq) - 1

    if log_project is not None:
        env.disable_log()
    agent.train()

    eq_arr = np.array(all_equities, dtype=np.float64)
    rets   = np.diff(eq_arr) / (eq_arr[:-1] + 1e-10)
    sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(365 * 288))
    max_dd = compute_max_drawdown(all_equities)

    cum = 1.0
    for r in ep_returns:
        cum *= 1 + r / 100
    tot_ret = (cum - 1) * 100 if ep_returns else 0.0

    n = len(all_actions)
    act_dist = {
        "HOLD":  all_actions.count(0) / n * 100 if n else 0,
        "LONG":  all_actions.count(1) / n * 100 if n else 0,
        "SHORT": all_actions.count(2) / n * 100 if n else 0,
        "CLOSE": all_actions.count(3) / n * 100 if n else 0,
    }
    return {
        "equities":   all_equities,
        "prices":     all_prices,
        "actions":    all_actions,
        "ep_returns": ep_returns,
        "metrics": {
            "total_return":    tot_ret,
            "sharpe":          sharpe,
            "max_drawdown":    max_dd,
            "n_episodes":      len(ep_returns),
            "n_steps":         n,
            "action_dist_pct": act_dist,
        },
    }
