"""Bucle d'entrenament FeUdal HRL i helpers d'optimitzadors.

Exporta:
    _make_optimizers(agent, lr, manager_lr) -> (opt_p, opt_m, opt_w)
    train_hrl(agent, ..., make_env_fn, ...) -> list
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.rl.training.hrl_updates import (
    compute_intrinsic_reward,
    ppo_update_manager,
    ppo_update_worker,
)
from src.rl.training.hrl_callbacks import FeeCurriculumCallback

if TYPE_CHECKING:
    import polars as pl

    from src.rl.agents.feudal_agent import FeudalAgent
    from src.rl.training.hierarchical_buffer import HierarchicalRolloutBuffer


def _make_optimizers(
    agent: "FeudalAgent",
    lr: float,
    manager_lr: float,
    weight_decay: float = 0.0,
) -> "tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]":
    """Adam optimizers per a percepció, manager i worker.

    Args:
        agent: FeudalAgent instance (04a architecture: perception, manager, worker modules)
        lr: Learning rate per perception i worker
        manager_lr: Learning rate per manager (+ phi_proj, goal_log_std)
        weight_decay: Weight decay (L2 regularization) aplicat a tots els optimizers

    Returns:
        (opt_perception, opt_manager, opt_worker)
    """
    opt_p = torch.optim.Adam(
        agent.perception.parameters(), lr=lr, eps=1e-5, weight_decay=weight_decay
    )
    opt_m = torch.optim.Adam(
        list(agent.manager.parameters()) + [agent.goal_log_std] + list(agent.phi_proj.parameters()),
        lr=manager_lr, eps=1e-5, weight_decay=weight_decay,
    )
    opt_w = torch.optim.Adam(
        agent.worker.parameters(), lr=lr, eps=1e-5, weight_decay=weight_decay
    )
    return opt_p, opt_m, opt_w


def _init_hrl_callbacks(callbacks: list, agent: "FeudalAgent") -> None:
    """Inicialitza callbacks SB3 (model=agent, n_calls=0, num_timesteps=0)."""
    for cb in callbacks:
        cb.model         = agent
        cb.n_calls       = 0
        cb.num_timesteps = 0
        if hasattr(cb, "_init_callback"):
            cb._init_callback()


def train_hrl(
    agent:           "FeudalAgent",
    opt_perception:  torch.optim.Optimizer,
    opt_manager:     torch.optim.Optimizer,
    opt_worker:      torch.optim.Optimizer,
    buffer:          "HierarchicalRolloutBuffer",
    df_tr:           "pl.DataFrame",
    df_vl:           "pl.DataFrame",
    total_timesteps: int,
    cfg:             dict,
    make_env_fn:     Callable,
    callbacks:       "list | None" = None,
    writer=None,
    verbose:         bool = True,
    log_project:     "str | None" = None,
    log_dir=None,
) -> list:
    """Bucle d'entrenament FeUdal HRL.

    Retorna [] (eval_history es troba a HrlEvalCallback.history).

    Camps cfg necessaris:
        n_steps, c, alpha, gamma, gae_lambda, clip_range,
        vf_coef, ent_coef, max_grad_norm, n_epochs, batch_size,
        hold_penalty, seed

    Camps cfg opcionals (per a make_env_fn):
        variable_length, min_episode_steps, max_episode_steps
    """
    callbacks = callbacks or []
    env_ref   = []
    _init_hrl_callbacks(callbacks, agent)

    n_steps       = cfg["n_steps"]
    c             = cfg["c"]
    alpha         = cfg["alpha"]
    gamma         = cfg["gamma"]
    gamma_m       = gamma ** c
    gae_lambda    = cfg["gae_lambda"]
    clip_range    = cfg["clip_range"]
    vf_coef       = cfg["vf_coef"]
    ent_coef      = cfg["ent_coef"]
    max_grad_norm = cfg["max_grad_norm"]
    n_epochs      = cfg["n_epochs"]
    batch_size    = cfg["batch_size"]
    device        = buffer.device
    seed          = cfg.get("seed", 42)

    env_kwargs: dict = {}
    for k in ("hold_penalty", "variable_length", "min_episode_steps", "max_episode_steps"):
        if k in cfg:
            env_kwargs[k] = cfg[k]

    env = make_env_fn(df_tr, **env_kwargs)
    env_ref.append(env)
    for cb in callbacks:
        if isinstance(cb, FeeCurriculumCallback):
            cb._env_ref = env_ref

    if log_project is not None and log_dir is not None:
        env.enable_log(project=log_project, log_dir=log_dir)

    obs, _ = env.reset(seed=seed)

    timestep       = 0
    step_in_ep     = 0
    cur_goal_norm  = None
    cur_goal_raw   = None
    cur_goal_lp    = 0.0
    cur_goal_val   = 0.0
    mgr_start_step = None
    mgr_acc_rew    = 0.0
    mgr_gamma_acc  = 1.0
    phi_z_history: list = []
    g_history:     list = []
    _ep_action_counts = [0, 0, 0, 0]  # HOLD, LONG, SHORT, CLOSE
    _ep_count = 0

    pbar = tqdm(total=total_timesteps, desc="train_hrl", disable=not verbose)
    agent.train()

    while timestep < total_timesteps:
        buffer.reset()

        for _ in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                z      = agent.perceive(obs_t)
                phi_zn = agent.phi_z(z).squeeze(0).cpu().numpy()
            z_np = z.squeeze(0).cpu().numpy()

            is_mgr_step = (step_in_ep % c == 0)
            if is_mgr_step:
                if mgr_start_step is not None and buffer.m_ptr < buffer.n_mgr_max:
                    buffer.add_manager(
                        z=z_np, g_raw=cur_goal_raw, rew=mgr_acc_rew,
                        val=cur_goal_val, lp=cur_goal_lp, done=0.0,
                    )
                with torch.no_grad():
                    g_norm_t, g_raw_t, lp_m, val_m, _ = agent.manager_sample(z)
                cur_goal_norm  = g_norm_t.squeeze(0).cpu().numpy()
                cur_goal_raw   = g_raw_t.squeeze(0).cpu().numpy()
                cur_goal_lp    = lp_m.item()
                cur_goal_val   = val_m.item()
                mgr_start_step = step_in_ep
                mgr_acc_rew    = 0.0
                mgr_gamma_acc  = 1.0
            elif cur_goal_norm is None:
                with torch.no_grad():
                    g_norm_t, g_raw_t, lp_m, val_m, _ = agent.manager_sample(z)
                cur_goal_norm  = g_norm_t.squeeze(0).cpu().numpy()
                cur_goal_raw   = g_raw_t.squeeze(0).cpu().numpy()
                cur_goal_lp    = lp_m.item()
                cur_goal_val   = val_m.item()
                mgr_start_step = step_in_ep
                mgr_acc_rew    = 0.0
                mgr_gamma_acc  = 1.0

            r_int = compute_intrinsic_reward(phi_zn, phi_z_history, g_history, c)

            g_t = torch.tensor(cur_goal_norm, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act_t, lp_w, _, val_w = agent.worker_sample(z, g_t)
            action = act_t.item()
            lp_w   = lp_w.item()
            val_w  = val_w.item()
            _ep_action_counts[action] += 1

            obs_next, r_ext, term, trunc, info = env.step(action)

            if trunc and not term:
                obs_nxt_t = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    z_nxt  = agent.perceive(obs_nxt_t)
                    g_nxt  = torch.tensor(
                        cur_goal_norm, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    _, v_b = agent.worker_forward(z_nxt, g_nxt)
                r_ext = r_ext + gamma * v_b.item()

            r_worker = alpha * r_ext + (1 - alpha) * r_int
            buffer.add_worker(
                obs=obs, z=z_np, goal=cur_goal_norm, act=action,
                rew=r_worker, val=val_w, lp=lp_w, done=float(term or trunc),
            )
            mgr_acc_rew   += mgr_gamma_acc * r_ext
            mgr_gamma_acc *= gamma

            phi_z_history.append(phi_zn)
            g_history.append(cur_goal_norm)
            if len(phi_z_history) > c + 2:
                phi_z_history.pop(0)
            if len(g_history) > c + 2:
                g_history.pop(0)

            obs         = obs_next
            timestep   += 1
            step_in_ep += 1
            pbar.update(1)

            if term or trunc:
                if mgr_start_step is not None and buffer.m_ptr < buffer.n_mgr_max:
                    buffer.add_manager(
                        z=z_np, g_raw=cur_goal_raw, rew=mgr_acc_rew,
                        val=cur_goal_val, lp=cur_goal_lp, done=1.0,
                    )
                _ep_count += 1
                _ep_total = sum(_ep_action_counts)
                if writer is not None and _ep_total > 0:
                    for i, name in enumerate(["HOLD", "LONG", "SHORT", "CLOSE"]):
                        writer.add_scalar(
                            f"train_actions/{name}_pct",
                            _ep_action_counts[i] / _ep_total * 100,
                            _ep_count,
                        )
                _ep_action_counts = [0, 0, 0, 0]
                obs, _ = env.reset()
                phi_z_history = []
                g_history     = []
                step_in_ep    = 0
                cur_goal_norm = None
                cur_goal_raw  = None
                mgr_start_step = None
                mgr_acc_rew    = 0.0
                mgr_gamma_acc  = 1.0

            for cb in callbacks:
                cb.n_calls       += 1
                cb.num_timesteps  = timestep
                if not cb._on_step():
                    pbar.close()
                    if log_project is not None:
                        env.disable_log()
                    if HAS_OPTUNA:
                        raise optuna.TrialPruned()
                    else:
                        raise RuntimeError("Trial pruned (optuna not installed)")

            if timestep >= total_timesteps:
                break

        # ── GAE ──────────────────────────────────────────────────────────────
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            z_last = agent.perceive(obs_t)
            g_last = torch.tensor(
                cur_goal_norm if cur_goal_norm is not None else np.zeros(buffer.goal_dim),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            _, last_w_val = agent.worker_forward(z_last, g_last)
            _, last_m_val = agent.manager_forward(z_last)
            last_w_val = last_w_val.item()
            last_m_val = (
                last_m_val.item()
                if last_m_val.ndim == 0
                else last_m_val.squeeze(-1).item()
            )

        adv_w, ret_w = buffer.compute_gae_worker(last_w_val, gamma,   gae_lambda)
        adv_m, ret_m = buffer.compute_gae_manager(last_m_val, gamma_m, gae_lambda)

        dev    = device
        obs_t  = torch.tensor(buffer.w_obs[:buffer.w_ptr],   device=dev)
        g_t    = torch.tensor(buffer.w_goals[:buffer.w_ptr], device=dev)
        act_t  = torch.tensor(buffer.w_acts[:buffer.w_ptr],  device=dev)
        adv_wt = torch.tensor(adv_w,                         device=dev)
        ret_wt = torch.tensor(ret_w,                         device=dev)
        lp_wt  = torch.tensor(buffer.w_lps[:buffer.w_ptr],  device=dev)
        m_z_t  = torch.tensor(buffer.m_z[:buffer.m_ptr],     device=dev)
        m_gr_t = torch.tensor(buffer.m_graws[:buffer.m_ptr], device=dev)
        adv_mt = torch.tensor(adv_m,                         device=dev)
        ret_mt = torch.tensor(ret_m,                         device=dev)
        lp_mt  = torch.tensor(buffer.m_lps[:buffer.m_ptr],  device=dev)

        stats_w = ppo_update_worker(
            agent, opt_perception, opt_worker,
            obs_t, g_t, act_t, ret_wt, adv_wt, lp_wt,
            n_epochs, batch_size, clip_range, vf_coef, ent_coef, max_grad_norm,
        )
        if buffer.m_ptr > 1:
            stats_m = ppo_update_manager(
                agent, opt_manager,
                m_z_t, m_gr_t, ret_mt, adv_mt, lp_mt,
                n_epochs, batch_size, clip_range, vf_coef, ent_coef, max_grad_norm,
            )
        else:
            stats_m = {}

        if writer is not None:
            writer.add_scalar("worker/policy_loss", stats_w["policy_loss"], timestep)
            writer.add_scalar("worker/value_loss",  stats_w["value_loss"],  timestep)
            writer.add_scalar("worker/entropy",     stats_w["entropy"],     timestep)
            if stats_m:
                writer.add_scalar("manager/policy_loss", stats_m.get("policy_loss", 0), timestep)

    pbar.close()
    if log_project is not None:
        env.disable_log()

    return []
