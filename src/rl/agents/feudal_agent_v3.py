"""
FeudalAgentV3 — HRL FeUdal v3 (05d_3 / 05s_3, Tanh + LSTM paper-style).

Arquitectura extreta de 05d_3_hrl_FeUdal_agent.ipynb:
  - Percepció:    obs → hidden → z_dim              (Tanh, Xavier init)
  - Manager:      fMspace: Linear(z→g_dim)+Tanh
                  fMrnn: LSTM(g_dim → g_dim)        (hidden=g_dim)
  - Worker:       fWpre: Linear(z→fwpre_h)+Tanh
                  fWrnn: LSTM(fwpre_h → A×K)
  - Sense goal_log_std
  - Hidden LSTM gestionats internament (reset quan cur_goal is None)

predict_step retorna (action, cur_goal) compatible amb l'avaluació.

Usage:
    from src.rl.agents.feudal_agent_v3 import FeudalAgentV3
    agent = FeudalAgentV3.load("checkpoints_dense/hrl_dense_feudal_agent_relu_N/hrl_feudal_best.pt")
    action, goal = agent.predict_step(obs, step_ep, c, cur_goal, device)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tanh_seq(in_dim: int, hidden: list, out_dim: int) -> nn.Sequential:
    dims = [in_dim] + list(hidden) + [out_dim]
    layers: list = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.Tanh())
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
        self.net = _tanh_seq(obs_dim, hidden, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Manager(nn.Module):
    def __init__(self, z_dim: int, g_dim: int, value_hidden: list):
        super().__init__()
        self.fMspace   = nn.Sequential(nn.Linear(z_dim, g_dim), nn.Tanh())
        self.fMrnn     = nn.LSTM(input_size=g_dim, hidden_size=g_dim,
                                  num_layers=1, batch_first=True)
        self.value_net = _tanh_seq(z_dim, value_hidden, 1)

    def forward(self, z: torch.Tensor, hx=None) -> dict:
        st           = self.fMspace(z)
        g_seq, new_hx = self.fMrnn(st.unsqueeze(1), hx)
        g_hat        = g_seq.squeeze(1)
        goal         = F.normalize(g_hat, dim=-1)
        value        = self.value_net(z)
        return {"goal": goal, "value": value, "st": st, "hx": new_hx}


class _Worker(nn.Module):
    def __init__(self, z_dim: int, g_dim: int, k_dim: int,
                 n_actions: int, fwpre_hidden: int, value_hidden: list):
        super().__init__()
        self.action_dim = n_actions
        self.k_dim      = k_dim
        output_dim      = n_actions * k_dim
        self.fWpre      = nn.Sequential(nn.Linear(z_dim, fwpre_hidden), nn.Tanh())
        self.fWrnn      = nn.LSTM(input_size=fwpre_hidden, hidden_size=output_dim,
                                   num_layers=1, batch_first=True)
        self.phi        = nn.Linear(g_dim, k_dim, bias=False)
        self.value_net  = _tanh_seq(z_dim, value_hidden, 1)

    def forward(self, z: torch.Tensor, goal: torch.Tensor, hx=None):
        B          = z.size(0)
        pre        = self.fWpre(z)
        U_seq, new_hx = self.fWrnn(pre.unsqueeze(1), hx)
        U          = U_seq.squeeze(1).view(B, self.action_dim, self.k_dim)
        w          = self.phi(goal)
        logits     = torch.bmm(U, w.unsqueeze(-1)).squeeze(-1)
        value      = self.value_net(z)
        return logits, value, new_hx


# ── FeudalAgentV3 ─────────────────────────────────────────────────────────────

class FeudalAgentV3(nn.Module):
    """HRL FeUdal v3 — Tanh + LSTM(g_dim) Manager, LSTM(A×K) Worker."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        g_dim: int,
        k_dim: int,
        perception_hidden: list,
        value_hidden: list,
        fwpre_hidden: int,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.g_dim      = g_dim

        self.perception = _Perception(obs_dim, z_dim, perception_hidden)
        self.manager    = _Manager(z_dim, g_dim, value_hidden)
        self.worker     = _Worker(z_dim, g_dim, k_dim, action_dim, fwpre_hidden, value_hidden)

        # Internal LSTM hidden states (reset at episode start)
        self._mgr_hx: Optional[Tuple] = None
        self._wkr_hx: Optional[Tuple] = None

    def reset_hidden(self):
        self._mgr_hx = None
        self._wkr_hx = None

    def predict_step(
        self,
        obs,
        step_ep: int,
        c: int,
        cur_goal,
        device,
        deterministic: bool = True,
    ):
        if cur_goal is None:
            self.reset_hidden()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            z = self.perception(obs_t)
            if cur_goal is None or step_ep % c == 0:
                mgr_out      = self.manager(z, hx=self._mgr_hx)
                cur_goal     = mgr_out["goal"].squeeze(0).cpu().numpy()
                self._mgr_hx = mgr_out["hx"]
            g_t            = torch.tensor(cur_goal, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _, wkr_hx = self.worker(z, g_t, hx=self._wkr_hx)
            self._wkr_hx   = wkr_hx
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = Categorical(logits=logits).sample().item()
        return action, cur_goal

    @classmethod
    def load(cls, path, device="cpu") -> "FeudalAgentV3":
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        sd        = ckpt["model_state_dict"]
        obs_dim   = int(ckpt["obs_dim"])
        n_actions = int(ckpt["n_actions"])

        z_dim, perception_hidden = _seq_dims(sd, "perception.net")
        # manager.fMspace is a 2-element Sequential (Linear+Tanh): g_dim from weight shape
        g_dim = sd["manager.fMspace.0.weight"].shape[0]
        _, value_hidden = _seq_dims(sd, "manager.value_net")

        # worker.fWpre is Linear(z→fwpre)+Tanh
        fwpre_hidden = sd["worker.fWpre.0.weight"].shape[0]
        k_dim        = sd["worker.phi.weight"].shape[0]

        agent = cls(
            obs_dim=obs_dim, action_dim=n_actions,
            z_dim=z_dim, g_dim=g_dim, k_dim=k_dim,
            perception_hidden=perception_hidden,
            value_hidden=value_hidden,
            fwpre_hidden=fwpre_hidden,
        )
        agent.load_state_dict(sd)
        agent.eval().to(device)
        return agent
