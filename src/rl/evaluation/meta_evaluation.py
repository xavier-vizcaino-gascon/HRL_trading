"""Avaluació meta-learning per a FeudalAgent: meta_rolling_walk_forward.

Protocol:
  Per cada finestra de M setmanes sobre df_test:
    1. Agafa les N setmanes prèvies com a suport (de df_history + df_test vist fins ara)
    2. deepcopy(meta_agent) → phi_w; adapta phi_w amb K passos de gradient
    3. phi_w prediu M setmanes, l'entorn rep:
         - initial_equity = equity final de la finestra anterior
         - cur_goal       = goal vector final de la finestra anterior
  Resultats concatenats → corba d'equity contínua.

Exporta:
    meta_rolling_walk_forward(meta_agent, df_test, df_history, device,
                               cfg, make_env_fn, ...) -> dict
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from src.rl.envs.crypto_market_env import MAX_EPISODE_STEPS
from src.rl.metrics import compute_max_drawdown, compute_sharpe
from src.rl.training.hierarchical_buffer import HierarchicalRolloutBuffer
from src.rl.training.hrl_training import _make_optimizers, train_hrl

if TYPE_CHECKING:
    import polars as pl

    from src.rl.agents.feudal_agent import FeudalAgent

STEPS_PER_WEEK = 2_016  # 1 setmana a 5 min


def _obs_dim_from_agent(agent: "FeudalAgent") -> int:
    """Infereix obs_dim del primer layer de percepció."""
    first_linear = next(m for m in agent.perception if hasattr(m, "in_features"))
    return first_linear.in_features


def meta_rolling_walk_forward(
    meta_agent: "FeudalAgent",
    df_test: "pl.DataFrame",
    df_history: "pl.DataFrame",
    device: "str | torch.device",
    cfg: dict,
    make_env_fn: Callable,
    n_support_weeks: int = 4,
    n_eval_weeks: int = 1,
    n_adapt_steps: int = 10_000,
    steps_per_week: int = STEPS_PER_WEEK,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Avaluació rolling-window amb fast-adaptation per a FeudalAgent.

    Args:
        meta_agent:       FeudalAgent meta-entrenat (no es modifica)
        df_test:          DataFrame del període de test
        df_history:       DataFrame anterior al test (per al suport inicial)
        device:           Dispositiu torch
        cfg:              Configuració HRL (c, lr, gamma, n_steps, etc.)
        make_env_fn:      callable(df, max_episode_steps=...) -> CryptoMarketEnv
        n_support_weeks:  Setmanes de suport per a fast-adaptation (N)
        n_eval_weeks:     Setmanes a predir per finestra (M, default=1)
        n_adapt_steps:    Passos de gradient per a fast-adaptation (K)
        steps_per_week:   Passos per setmana (2016 a 5 min)
        seed:             Seed base (s'incrementa per finestra)
        verbose:          Si True, imprimeix progrés per finestra

    Returns:
        dict amb:
          equities      - corba d'equity contínua (list[float])
          prices        - preus associats (list[float])
          actions       - accions (list[int])
          ep_returns    - retorns per finestra % (list[float])
          metrics       - dict amb total_return, sharpe, max_drawdown,
                          n_windows, n_steps, action_dist_pct
    """
    import polars as pl

    device = torch.device(device) if isinstance(device, str) else device
    c         = cfg["c"]
    obs_dim   = _obs_dim_from_agent(meta_agent)
    n_support_rows = n_support_weeks * steps_per_week
    n_eval_rows    = n_eval_weeks * steps_per_week
    n_test_rows    = len(df_test)

    all_equities: list[float] = []
    all_actions:  list[int]   = []
    all_prices:   list[float] = []
    ep_returns:   list[float] = []

    initial_equity: float | None = None
    cur_goal:       np.ndarray | None = None
    test_start = 0
    window_idx = 0

    while test_start + n_eval_rows + 1 < n_test_rows:
        # ── 1. Dades de suport ────────────────────────────────────────────
        if test_start == 0:
            support_full = df_history
        else:
            support_full = pl.concat([df_history, df_test[:test_start]])

        if len(support_full) >= n_support_rows:
            df_support = support_full[-n_support_rows:]
        else:
            df_support = support_full  # menys del desitjat: usa tot el disponible

        if verbose:
            print(f"[window {window_idx}] support={len(df_support)} rows, "
                  f"test_start={test_start}, eval_rows={n_eval_rows}")

        # ── 2. Fast-adaptation ────────────────────────────────────────────
        phi_w = copy.deepcopy(meta_agent).to(device)
        phi_w.train()

        lr         = cfg.get("lr", 3e-4)
        manager_lr = cfg.get("manager_lr", lr)
        opt_p, opt_m, opt_w = _make_optimizers(phi_w, lr, manager_lr)

        buf = HierarchicalRolloutBuffer(
            n_steps=cfg["n_steps"],
            obs_dim=obs_dim,
            latent_dim=phi_w.latent_dim,
            goal_dim=phi_w.goal_dim,
            c=c,
            device=device,
        )

        adapt_cfg = {
            **cfg,
            "seed": seed + window_idx,
            "max_episode_steps": max(1, len(df_support) - 1),
        }
        train_hrl(
            phi_w, opt_p, opt_m, opt_w, buf,
            df_tr=df_support,
            df_vl=df_support,
            total_timesteps=n_adapt_steps,
            cfg=adapt_cfg,
            make_env_fn=make_env_fn,
            callbacks=[],
            verbose=False,
        )
        phi_w.eval()

        # ── 3. Predicció M setmanes ───────────────────────────────────────
        df_eval = df_test[test_start: test_start + n_eval_rows + 1]
        env     = make_env_fn(df_eval, max_episode_steps=n_eval_rows)

        reset_opts: dict = {"episode_start": 0}
        if initial_equity is not None:
            reset_opts["initial_equity"] = initial_equity

        obs, info = env.reset(options=reset_opts)
        if initial_equity is None:
            initial_equity = float(info["equity"])

        ep_eq = [float(info["equity"])]
        done  = False
        step_ep = 0
        # cur_goal es passa de la finestra anterior (o None per a la primera)
        window_goal = cur_goal

        with torch.no_grad():
            while not done:
                action, window_goal = phi_w.predict_step(
                    obs, step_ep, c, window_goal, device, deterministic=True
                )
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
                ep_eq.append(float(info["equity"]))
                all_actions.append(action)
                all_prices.append(float(info.get("current_price", float("nan"))))
                step_ep += 1

        # ── 4. Actualitza estat per a la finestra següent ─────────────────
        all_equities.extend(ep_eq)
        w_ret = (ep_eq[-1] / ep_eq[0] - 1) * 100
        ep_returns.append(w_ret)
        initial_equity = ep_eq[-1]
        cur_goal       = window_goal          # propaga el goal vector
        test_start    += len(ep_eq) - 1
        window_idx    += 1

    # ── Mètriques ─────────────────────────────────────────────────────────
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
        "HOLD":  all_actions.count(0) / n * 100 if n else 0.0,
        "LONG":  all_actions.count(1) / n * 100 if n else 0.0,
        "SHORT": all_actions.count(2) / n * 100 if n else 0.0,
        "CLOSE": all_actions.count(3) / n * 100 if n else 0.0,
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
            "n_windows":       window_idx,
            "n_steps":         n,
            "action_dist_pct": act_dist,
        },
    }
