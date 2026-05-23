"""
FeudalAgentV9 — HRL FeUdal v9 (07d_hrl_import, FeudalNetwork reference impl).

Reimplementació mínima (inference-only) que coincideix exactament amb l'estructura
del state dict guardat per 07d_hrl_import.ipynb. Importa únicament DilatedLSTM
del directori de referència (sense dependència de gym).

Checkpoint format:
    {'model': feudalnet.state_dict(), 'hp': {...}, 'eval_history': [...]}

HP dict keys rellevants: hidden_dim_manager (d), hidden_dim_worker (k),
                         time_horizon (c), dilation (r), perception_hidden.

Usage:
    from src.rl.agents.feudal_agent_v9 import FeudalAgentV9
    agent = FeudalAgentV9.load("results_dense/07d_hrl_import/feudalnet_final.pt")
    action, goal = agent.predict_step(obs, step_ep, c, cur_goal, device)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ── Import DilatedLSTM des del directori de referència (no depèn de gym) ──────
_REF = Path(__file__).parent.parent.parent.parent / "references" / "feudalnets-pytorch-master"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

try:
    from dilated_lstm import DilatedLSTM
except ImportError as e:
    raise ImportError(
        f"No s'ha pogut importar DilatedLSTM des de {_REF}. Error: {e}"
    )


# ── Mòduls d'arquitectura (inference-only, coincidents amb l'state dict) ──────

class _Percept(nn.Module):
    """Percepció MLP: obs → perception_hidden (ReLU) → d (ReLU)."""

    def __init__(self, obs_dim: int, d: int, perception_hidden: int = 256):
        super().__init__()
        self.percept = nn.Sequential(
            nn.Linear(obs_dim, perception_hidden), nn.ReLU(),
            nn.Linear(perception_hidden, d), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.percept(x)


class _Manager(nn.Module):
    """Manager: Mspace (Linear+ReLU) + DilatedLSTM + critic (Linear)."""

    def __init__(self, d: int, r: int):
        super().__init__()
        self.Mspace = nn.Linear(d, d)
        self.Mrnn   = DilatedLSTM(d, d, r)
        self.critic = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor, hidden, mask: torch.Tensor):
        state  = self.Mspace(z).relu()
        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(state, hidden)
        value  = self.critic(goal_hat)
        goal   = F.normalize(goal_hat, dim=-1)
        return goal, hidden, state.detach(), value


class _Worker(nn.Module):
    """Worker: LSTMCell + phi (Linear, no bias) + critic Sequential."""

    def __init__(self, d: int, k: int, n_actions: int):
        super().__init__()
        self.k        = k
        self.n_actions = n_actions
        self.Wrnn   = nn.LSTMCell(d, k * n_actions)
        self.phi    = nn.Linear(d, k, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(k * n_actions, 50), nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, z: torch.Tensor, goals: List[torch.Tensor],
                hidden, mask: torch.Tensor):
        hidden   = (mask * hidden[0], mask * hidden[1])
        u, cx    = self.Wrnn(z, hidden)
        hidden   = (u, cx)
        g_sum    = torch.stack(goals).detach().sum(dim=0)
        w        = self.phi(g_sum)
        value    = self.critic(u)
        u_r      = u.reshape(u.shape[0], self.k, self.n_actions)
        a        = torch.einsum("bk, bka -> ba", w, u_r).softmax(dim=-1)
        return a, hidden, value


class _FeudalNetV9(nn.Module):
    """Xarxa FeUdal completa per a inferència (coincident amb l'state dict de 07d)."""

    def __init__(self, obs_dim: int, d: int, k: int, n_actions: int,
                 c: int, r: int, perception_hidden: int, device: str):
        super().__init__()
        self.b        = 1           # batch = 1 durant inferència
        self.c        = c
        self.d        = d
        self.k        = k
        self.r        = r
        self.n_actions = n_actions
        self.device   = device

        self.percept = _Percept(obs_dim, d, perception_hidden)
        self.manager = _Manager(d, r)
        self.worker  = _Worker(d, k, n_actions)

        # Inicialitzem hidden states (s'actualitzen durant forward)
        self.hidden_m = self._zero_hidden(r * d)
        self.hidden_w = self._zero_hidden(k * n_actions)

    def _zero_hidden(self, h_dim: int):
        return (
            torch.zeros(1, h_dim, device=self.device),
            torch.zeros(1, h_dim, device=self.device),
        )

    def init_obj(self):
        template = torch.zeros(1, self.d, device=self.device)
        goals  = [torch.zeros_like(template) for _ in range(2 * self.c + 1)]
        states = [torch.zeros_like(template) for _ in range(2 * self.c + 1)]
        masks  = [torch.ones(1, 1, device=self.device) for _ in range(2 * self.c + 1)]
        return goals, states, masks

    def repackage_hidden(self):
        self.hidden_m = tuple(h.detach() for h in self.hidden_m)
        self.hidden_w = tuple(h.detach() for h in self.hidden_w)

    def forward(self, x: torch.Tensor, goals: list, states: list,
                mask: torch.Tensor, save: bool = True):
        z = self.percept(x)

        goal, hidden_m, state, value_m = self.manager(z, self.hidden_m, mask)

        if len(goals) > (2 * self.c + 1):
            goals.pop(0)
            states.pop(0)
        goals.append(goal)
        states.append(state)

        action_dist, hidden_w, value_w = self.worker(
            z, goals[:self.c + 1], self.hidden_w, mask
        )

        if save:
            self.hidden_m = hidden_m
            self.hidden_w = hidden_w

        return action_dist, goals, states, value_m, value_w


# ── FeudalAgentV9 ─────────────────────────────────────────────────────────────

class FeudalAgentV9(nn.Module):
    """
    Wrapper de _FeudalNetV9 amb interfície compatible amb els notebooks d'avaluació.
    Gestiona internament goals, states i masks entre passos d'un episodi.
    """

    def __init__(self, net: _FeudalNetV9, c: int):
        super().__init__()
        self._net = net
        self.c    = c
        self._goals: Optional[list]  = None
        self._states: Optional[list] = None
        self._masks: Optional[list]  = None

    def reset_episode(self):
        self._goals, self._states, self._masks = self._net.init_obj()
        self._net.hidden_m = self._net._zero_hidden(self._net.r * self._net.d)
        self._net.hidden_w = self._net._zero_hidden(self._net.k * self._net.n_actions)
        self._net.manager.Mrnn.dilation = 0   # reinicia el comptador del DilatedLSTM

    def _ensure_init(self):
        if self._goals is None:
            self._goals, self._states, self._masks = self._net.init_obj()

    def predict_step(
        self,
        obs,
        step_ep: int,
        c: int,
        cur_goal,
        device,
        deterministic: bool = True,
    ):
        """Un pas d'inferència.

        cur_goal s'accepta per compatibilitat d'interfície però la gestió interna
        del goal la fa _FeudalNetV9. Retorna (action, cur_goal_np).
        """
        if cur_goal is None:
            self.reset_episode()
        else:
            self._ensure_init()

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._net.device)
        mask  = torch.ones(1, 1, device=self._net.device)

        with torch.no_grad():
            action_dist, self._goals, self._states, _, _ = self._net(
                obs_t, self._goals, self._states, mask
            )
            if self._masks is not None:
                self._masks.pop(0)
                self._masks.append(mask)

        if deterministic:
            action = torch.argmax(action_dist, dim=-1).item()
        else:
            action = Categorical(probs=action_dist).sample().item()

        cur_goal_np = self._goals[-1].squeeze(0).detach().cpu().numpy()
        return action, cur_goal_np

    @classmethod
    def load(cls, path, device="cpu") -> "FeudalAgentV9":
        """Carrega FeudalAgentV9 des del checkpoint de 07d_hrl_import.

        Espera clau 'model' (state_dict) i 'hp' (hiperparàmetres).
        """
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        sd   = ckpt.get("model") or ckpt.get("model_state_dict")
        if sd is None:
            raise KeyError(
                f"El checkpoint no conté 'model' ni 'model_state_dict'. "
                f"Claus: {list(ckpt.keys())}"
            )

        hp = ckpt.get("hp", {})
        d                 = int(hp.get("hidden_dim_manager",  sd["manager.Mspace.weight"].shape[0]))
        k                 = int(hp.get("hidden_dim_worker",   sd["worker.phi.weight"].shape[0]))
        c                 = int(hp.get("time_horizon",        10))
        r                 = int(hp.get("dilation",            10))
        perception_hidden = int(hp.get("perception_hidden",   sd["percept.percept.0.weight"].shape[0]))

        # Infer obs_dim i n_actions des del state dict
        obs_dim   = sd["percept.percept.0.weight"].shape[1]
        k_n_act   = sd["worker.Wrnn.weight_hh"].shape[1]   # k * n_actions
        n_actions = k_n_act // k

        net = _FeudalNetV9(
            obs_dim=obs_dim, d=d, k=k, n_actions=n_actions,
            c=c, r=r, perception_hidden=perception_hidden, device=device,
        )
        net.load_state_dict(sd)
        net.eval().to(device)

        return cls(net=net, c=c)

    def forward(self, *args, **kwargs):
        return self._net(*args, **kwargs)
