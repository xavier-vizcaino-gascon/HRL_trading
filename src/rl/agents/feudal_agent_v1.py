"""
FeudalAgentV1 — HRL FeUdal v1 (05d_1 / 05s_1, LeakyReLU MLP).

Arquitectura extreta de 05d_1_hrl_FeUdal_agent.ipynb:
  - Percepció:  obs → hidden → z_dim            (LeakyReLU 0.01, no act. final)
  - Manager:    fMspace(z→g_dim) + fMrnn(g→g)   (Sequential MLP, LeakyReLU)
  - Worker:     fWrnn(z→32) + action_embed(→A×K) (Sequential MLP, LeakyReLU)
  - goal_log_std paràmetre entrenable

load() auto-detecta totes les dimensions des del state dict del checkpoint.

Usage:
    from src.rl.agents.feudal_agent_v1 import FeudalAgentV1
    agent = FeudalAgentV1.load("checkpoints_dense/hrl_dense_feudal_agent_relu/hrl_feudal_best.pt")
    action, goal = agent.predict_step(obs, step_ep, c, cur_goal, device)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ── Helpers ───────────────────────────────────────────────────────────────────

def _leaky_seq(in_dim: int, hidden: list, out_dim: int) -> nn.Sequential:
    dims = [in_dim] + list(hidden) + [out_dim]
    layers: list = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.LeakyReLU(0.01))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def _seq_dims(sd: dict, prefix: str):
    """Returns (out_dim, hidden_dims) by inspecting Linear weights in a Sequential."""
    keys = sorted(
        [k for k in sd
         if k.startswith(prefix + ".") and k.endswith(".weight")
         and k.split(".")[-2].isdigit()],
        key=lambda k: int(k.split(".")[-2]),
    )
    out_dim = sd[keys[-1]].shape[0]
    hidden  = [sd[k].shape[0] for k in keys[:-1]]
    return out_dim, hidden


# ── Sub-modules ───────────────────────────────────────────────────────────────

class _Perception(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, hidden: list):
        super().__init__()
        self.net = _leaky_seq(obs_dim, hidden, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Manager(nn.Module):
    def __init__(self, z_dim: int, g_dim: int,
                 fmspace_hidden: list, fmrnn_hidden: list, value_hidden: list):
        super().__init__()
        self.fMspace   = _leaky_seq(z_dim, fmspace_hidden, g_dim)
        self.fMrnn     = _leaky_seq(g_dim, fmrnn_hidden, g_dim)
        self.value_net = _leaky_seq(z_dim, value_hidden, 1)

    def forward(self, z: torch.Tensor) -> dict:
        st    = self.fMspace(z)
        g_hat = self.fMrnn(st)
        goal  = F.normalize(g_hat, dim=-1)
        value = self.value_net(z)
        return {"goal": goal, "value": value, "st": st}


class _Worker(nn.Module):
    def __init__(self, z_dim: int, g_dim: int, k_dim: int, n_actions: int,
                 fwrnn_hidden: list, fwrnn_out: int, value_hidden: list):
        super().__init__()
        self.action_dim = n_actions
        self.k_dim      = k_dim
        self.fWrnn      = _leaky_seq(z_dim, fwrnn_hidden, fwrnn_out)
        self.action_embed = nn.Linear(fwrnn_out, n_actions * k_dim)
        self.phi        = nn.Linear(g_dim, k_dim, bias=False)
        self.value_net  = _leaky_seq(z_dim + g_dim, value_hidden, 1)

    def forward(self, z: torch.Tensor, goal: torch.Tensor):
        B     = z.size(0)
        h     = self.fWrnn(z)
        U     = self.action_embed(h).view(B, self.action_dim, self.k_dim)
        w     = self.phi(goal)
        logits = torch.bmm(U, w.unsqueeze(-1)).squeeze(-1)
        value  = self.value_net(torch.cat([z, goal], dim=-1))
        return logits, value


# ── FeudalAgentV1 ─────────────────────────────────────────────────────────────

class FeudalAgentV1(nn.Module):
    """HRL FeUdal v1 — LeakyReLU(0.01) MLP, sense LSTM."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        g_dim: int,
        k_dim: int,
        perception_hidden: list,
        fmspace_hidden: list,
        fmrnn_hidden: list,
        value_hidden: list,
        fwrnn_hidden: list,
        fwrnn_out: int = 32,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.g_dim      = g_dim

        self.perception = _Perception(obs_dim, z_dim, perception_hidden)
        self.manager    = _Manager(z_dim, g_dim, fmspace_hidden, fmrnn_hidden, value_hidden)
        self.worker     = _Worker(z_dim, g_dim, k_dim, action_dim,
                                  fwrnn_hidden, fwrnn_out, value_hidden)
        self.goal_log_std = nn.Parameter(torch.full((g_dim,), -1.0))

    def predict_step(
        self,
        obs,
        step_ep: int,
        c: int,
        cur_goal,
        device,
        deterministic: bool = True,
    ):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            z = self.perception(obs_t)
            if cur_goal is None or step_ep % c == 0:
                out      = self.manager(z)
                cur_goal = out["goal"].squeeze(0).cpu().numpy()
            g_t    = torch.tensor(cur_goal, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = self.worker(z, g_t)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = Categorical(logits=logits).sample().item()
        return action, cur_goal

    @classmethod
    def load(cls, path, device="cpu") -> "FeudalAgentV1":
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        sd        = ckpt["model_state_dict"]
        obs_dim   = int(ckpt["obs_dim"])
        n_actions = int(ckpt["n_actions"])

        z_dim,   perception_hidden = _seq_dims(sd, "perception.net")
        g_dim,   fmspace_hidden    = _seq_dims(sd, "manager.fMspace")
        _,       fmrnn_hidden      = _seq_dims(sd, "manager.fMrnn")
        _,       value_hidden      = _seq_dims(sd, "manager.value_net")
        fwrnn_out, fwrnn_hidden    = _seq_dims(sd, "worker.fWrnn")
        k_dim  = sd["worker.phi.weight"].shape[0]

        agent = cls(
            obs_dim=obs_dim, action_dim=n_actions,
            z_dim=z_dim, g_dim=g_dim, k_dim=k_dim,
            perception_hidden=perception_hidden,
            fmspace_hidden=fmspace_hidden,
            fmrnn_hidden=fmrnn_hidden,
            value_hidden=value_hidden,
            fwrnn_hidden=fwrnn_hidden,
            fwrnn_out=fwrnn_out,
        )
        agent.load_state_dict(sd)
        agent.eval().to(device)
        return agent
