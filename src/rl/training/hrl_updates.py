"""PPO updates per Worker i Manager del FeUdal HRL, i recompensa intrínseca."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def compute_intrinsic_reward(
    phi_z_np: np.ndarray,
    phi_z_history: list,
    g_history: list,
    c: int,
) -> float:
    """r_int = cos_sim(phi(z_t) - phi(z_{t-c}), g_{t-c}).

    Retorna 0.0 si l'historial és insuficient (primeres c passes).
    (FeUdal Networks, Vezhnevets et al. 2017, eq. 9)
    """
    if len(phi_z_history) < c or len(g_history) < c:
        return 0.0
    delta_z = phi_z_np - phi_z_history[-c]
    g_prev  = g_history[-c]
    denom   = (np.linalg.norm(delta_z) + 1e-8) * (np.linalg.norm(g_prev) + 1e-8)
    return float(np.clip(np.dot(delta_z, g_prev) / denom, -1.0, 1.0))


def ppo_update_worker(
    agent,
    opt_perception: torch.optim.Optimizer,
    opt_worker: torch.optim.Optimizer,
    obs_t: torch.Tensor,
    goals_t: torch.Tensor,
    acts_t: torch.Tensor,
    rets_t: torch.Tensor,
    adv_t: torch.Tensor,
    lps_old_t: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    old_vals_t: torch.Tensor | None = None,
    clip_value: bool = True,
) -> dict:
    """PPO update per al Worker (+ Percepció compartida).

    Args:
        old_vals_t: Value predictions del rollout (per value clipping). Si None, no clip.
        clip_value: Si True, aplica value clipping (SB3 standard).
    """
    n     = obs_t.shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=obs_t.device)
        for start in range(0, n, batch_size):
            b      = perm[start: start + batch_size]
            mb_adv = adv_t[b]
            if len(mb_adv) > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            z_new  = agent.perceive(obs_t[b])
            g_det  = goals_t[b].detach()
            logits, new_val = agent.worker_forward(z_new, g_det)
            dist   = Categorical(logits=logits)
            new_lp = dist.log_prob(acts_t[b])
            ent    = dist.entropy()

            ratio = torch.exp(new_lp - lps_old_t[b])
            pg1   = -mb_adv * ratio
            pg2   = -mb_adv * ratio.clamp(1 - clip_range, 1 + clip_range)
            pl    = torch.max(pg1, pg2).mean()

            # Value clipping (SB3 standard)
            if clip_value and old_vals_t is not None:
                old_val_b = old_vals_t[b]
                val_clipped = old_val_b + torch.clamp(
                    new_val - old_val_b, -clip_range, clip_range
                )
                vl1 = F.mse_loss(new_val, rets_t[b])
                vl2 = F.mse_loss(val_clipped, rets_t[b])
                vl  = torch.max(vl1, vl2)
            else:
                vl = F.mse_loss(new_val, rets_t[b])

            el    = ent.mean()
            loss  = pl + vf_coef * vl - ent_coef * el

            opt_perception.zero_grad()
            opt_worker.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.perception.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(agent.worker.parameters(), max_grad_norm)
            opt_perception.step()
            opt_worker.step()

            with torch.no_grad():
                kl = (lps_old_t[b] - new_lp).mean().item()
            stats["policy_loss"].append(pl.item())
            stats["value_loss"].append(vl.item())
            stats["entropy"].append(el.item())
            stats["approx_kl"].append(kl)

    return {k: float(np.mean(v)) for k, v in stats.items()}


def ppo_update_manager(
    agent,
    opt_manager: torch.optim.Optimizer,
    z_t: torch.Tensor,
    graws_t: torch.Tensor,
    rets_t: torch.Tensor,
    adv_t: torch.Tensor,
    lps_old_t: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    old_vals_t: torch.Tensor | None = None,
    clip_value: bool = True,
) -> dict:
    """PPO update per al Manager (cada c passos).

    Args:
        old_vals_t: Value predictions del rollout (per value clipping). Si None, no clip.
        clip_value: Si True, aplica value clipping (SB3 standard).
    """
    n     = z_t.shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=z_t.device)
        for start in range(0, n, batch_size):
            b      = perm[start: start + batch_size]
            mb_adv = adv_t[b]
            if len(mb_adv) > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            g_mean, new_val = agent.manager_forward(z_t[b].detach())
            std     = agent.goal_log_std.exp().expand_as(g_mean)
            dist    = Normal(g_mean, std)
            new_lp  = dist.log_prob(graws_t[b]).sum(-1)
            entropy = dist.entropy().sum(-1)

            ratio = torch.exp(new_lp - lps_old_t[b])
            pg1   = -mb_adv * ratio
            pg2   = -mb_adv * ratio.clamp(1 - clip_range, 1 + clip_range)
            pl    = torch.max(pg1, pg2).mean()

            # Value clipping (SB3 standard)
            if clip_value and old_vals_t is not None:
                old_val_b = old_vals_t[b]
                val_clipped = old_val_b + torch.clamp(
                    new_val - old_val_b, -clip_range, clip_range
                )
                vl1 = F.mse_loss(new_val, rets_t[b])
                vl2 = F.mse_loss(val_clipped, rets_t[b])
                vl  = torch.max(vl1, vl2)
            else:
                vl = F.mse_loss(new_val, rets_t[b])

            el    = entropy.mean()
            loss  = pl + vf_coef * vl - ent_coef * el

            opt_manager.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(agent.manager.parameters()) + [agent.goal_log_std],
                max_grad_norm,
            )
            opt_manager.step()
            with torch.no_grad():
                agent.goal_log_std.clamp_(-2.0, 0.5)

            with torch.no_grad():
                kl = (lps_old_t[b] - new_lp).mean().item()
            stats["policy_loss"].append(pl.item())
            stats["value_loss"].append(vl.item())
            stats["entropy"].append(el.item())
            stats["approx_kl"].append(kl)

    return {k: float(np.mean(v)) for k, v in stats.items()}
