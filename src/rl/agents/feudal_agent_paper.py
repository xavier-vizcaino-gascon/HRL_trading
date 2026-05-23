"""
FeUdalAgent — Paper Architecture (05d series).

Implementació fidel a FeUdal Networks (Vezhnevets et al. 2017) amb suport
LSTM (function_type='recurrent') i MLP (function_type='fc_lineal').

Extreta del notebook 05d_4_hrl_FeUdal_agent.ipynb per ús en el pipeline
d'avaluació (10d_evaluation.ipynb).

Arquitectura del checkpoint 05d_4 (recurrent):
  Perception:  obs(71) → 64(tanh) → 64(tanh) → z(64)
  fMspace:     z(64)   → 32(tanh) = St
  fMrnn LSTM:  St(32)  → g_dim(32)            (Manager goal)
  fWpre:       z(64)   → 32(tanh) = h
  fWrnn LSTM:  h(32)   → A×K(32)              (Worker U matrix)
  phi:         g(32)   → w(8)                  (goal → key, no bias)
  logits:      U(4×8) · w(8) → (4,)
  value_net:   z(64)   → 64(tanh) → 32(tanh) → 1  (shared M/W)

Punt d'entrada:
    from src.rl.agents.feudal_agent_paper import FeUdalAgent
    agent = FeUdalAgent.load("checkpoints_dense/hrl_dense_feudal_agent_paper/hrl_feudal_best.pt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ── Mòduls d'arquitectura ─────────────────────────────────────────────────────

class PerceptionModule(nn.Module):
    """Percepció compartida: obs_dim → [hidden layers (tanh)] → z_dim (linear)."""

    def __init__(self, obs_dim: int, z_dim: int, hidden: List[int]):
        super().__init__()
        dims = [obs_dim] + hidden + [z_dim]
        layers: list = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ManagerModule(nn.Module):
    """Manager: genera goals normalitzats i estima V^M.

    fMspace: z → g_dim (Tanh)        — representació interna St
    fMrnn:   St → g (LSTM o MLP)     — goal normalitzat L2
    value_net: z → 1                 — V^M
    """

    def __init__(
        self,
        z_dim: int,
        g_dim: int,
        value_hidden: List[int],
        function_type: str = "recurrent",
        fmspace_hidden: Optional[List[int]] = None,
        fmrnn_hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        self.function_type = function_type

        # fMspace: z → St  (paper: z → g_dim via Linear + Tanh)
        self.fMspace = nn.Sequential(nn.Linear(z_dim, g_dim), nn.Tanh())

        # fMrnn: St → g_hat (LSTM o MLP)
        if self.function_type == "recurrent":
            self.fMrnn = nn.LSTM(input_size=g_dim, hidden_size=g_dim,
                                 num_layers=1, batch_first=True)
        else:
            self.fMrnn = nn.Sequential(
                nn.Linear(g_dim, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, g_dim),
            )

        # Value head: z → hidden → 1
        value_dims = [z_dim] + value_hidden + [1]
        layers: list = []
        for i in range(len(value_dims) - 2):
            layers.append(nn.Linear(value_dims[i], value_dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(value_dims[-2], value_dims[-1]))
        self.value_net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, hx=None) -> dict:
        st = self.fMspace(z)
        if self.function_type == "recurrent":
            g_seq, (h_n, c_n) = self.fMrnn(st.unsqueeze(1), hx)
            g_hat = g_seq.squeeze(1)
            new_hx = (h_n, c_n)
        else:
            g_hat = self.fMrnn(st)
            new_hx = None
        goal = F.normalize(g_hat, dim=-1)
        value = self.value_net(z)
        return {"goal": goal, "value": value, "st": st, "hx": new_hx}


class WorkerModule(nn.Module):
    """Worker: selecciona accions via U·w (FuN eq. 3).

    fWpre:    z → h (Tanh)
    fWrnn:    h → U flat (LSTM o MLP)  → U ∈ ℝ^(A×K)
    phi:      g → w ∈ ℝ^K             (sense bias)
    logits:   U·w → π
    value_net: z → 1
    """

    def __init__(
        self,
        z_dim: int,
        g_dim: int,
        k_dim: int,
        n_actions: int,
        fwpre_hidden: int,
        value_hidden: List[int],
        function_type: str = "recurrent",
    ):
        super().__init__()
        self.action_dim = n_actions
        self.k_dim = k_dim
        self.function_type = function_type

        # fWpre: z → h (Tanh)
        self.fWpre = nn.Sequential(nn.Linear(z_dim, fwpre_hidden), nn.Tanh())

        # fWrnn: h → U flat (A×K)
        output_dim = n_actions * k_dim
        if self.function_type == "recurrent":
            self.fWrnn = nn.LSTM(input_size=fwpre_hidden, hidden_size=output_dim,
                                 num_layers=1, batch_first=True)
        else:
            self.fWrnn = nn.Sequential(
                nn.Linear(fwpre_hidden, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, output_dim),
            )

        # phi: g → w (sense bias, paper FuN)
        self.phi = nn.Linear(g_dim, k_dim, bias=False)

        # Value head: z → hidden → 1
        value_dims = [z_dim] + value_hidden + [1]
        layers: list = []
        for i in range(len(value_dims) - 2):
            layers.append(nn.Linear(value_dims[i], value_dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(value_dims[-2], value_dims[-1]))
        self.value_net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, goal: torch.Tensor, hx=None) -> Tuple:
        B = z.size(0)
        pre = self.fWpre(z)
        if self.function_type == "recurrent":
            U_seq, (h_n, c_n) = self.fWrnn(pre.unsqueeze(1), hx)
            U_flat = U_seq.squeeze(1)
            new_hx = (h_n, c_n)
        else:
            U_flat = self.fWrnn(pre)
            new_hx = None
        U = U_flat.view(B, self.action_dim, self.k_dim)
        w = self.phi(goal)
        logits = torch.bmm(U, w.unsqueeze(-1)).squeeze(-1)
        value = self.value_net(z)
        return logits, value, new_hx


# ── FeUdalAgent ───────────────────────────────────────────────────────────────

class FeUdalAgent(nn.Module):
    """
    Agent HRL FeUdal (paper architecture, sèrie 05d).

    Compatible amb HRLAgentWrapper de 10d_evaluation.ipynb:
      predict_step() retorna (action, cur_goal) i gestiona internament
      els hidden states LSTM entre passos d'un episodi.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        g_dim: int,
        k_dim: int,
        perception_hidden: List[int],
        value_hidden: List[int],
        fwpre_hidden: int,
        function_type: str = "recurrent",
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.z_dim      = z_dim
        self.g_dim      = g_dim
        self.k_dim      = k_dim
        self.function_type = function_type

        self.perception = PerceptionModule(obs_dim, z_dim, perception_hidden)
        self.manager    = ManagerModule(
            z_dim=z_dim, g_dim=g_dim,
            value_hidden=value_hidden,
            function_type=function_type,
        )
        self.worker = WorkerModule(
            z_dim=z_dim, g_dim=g_dim, k_dim=k_dim, n_actions=action_dim,
            fwpre_hidden=fwpre_hidden,
            value_hidden=value_hidden,
            function_type=function_type,
        )

        # LSTM hidden states per predict_step (gestionats internament)
        self._mgr_hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._wkr_hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ── Core forward ──────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        z           = self.perception(obs)
        manager_out = self.manager(z)
        goal        = manager_out["goal"]
        worker_logits, worker_value, _ = self.worker(z, goal.detach())
        return {
            "logits":        worker_logits,
            "worker_value":  worker_value,
            "goal":          goal,
            "manager_value": manager_out["value"],
            "st":            manager_out["st"],
            "z":             z,
        }

    # ── Helpers per entrenament ───────────────────────────────────────────────

    def perceive(self, obs: torch.Tensor) -> torch.Tensor:
        return self.perception(obs)

    def phi_z(self, z: torch.Tensor) -> torch.Tensor:
        """St = fMspace(z) sense LSTM (per reward intrínsec)."""
        with torch.no_grad():
            return self.manager.fMspace(z)

    def manager_forward(self, z: torch.Tensor, hx=None):
        out = self.manager(z, hx=hx)
        return out["goal"], out["value"], out["hx"]

    def manager_sample(self, z: torch.Tensor, hx=None):
        """Manager determinístic (TPG): retorna (g, g, dummy_lp, val, g, hx)."""
        out  = self.manager(z, hx=hx)
        g    = out["goal"]
        val  = out["value"]
        dummy_lp = torch.zeros(g.size(0), 1, device=g.device)
        return g, g, dummy_lp, val, g, out["hx"]

    def worker_forward(self, z: torch.Tensor, goal: torch.Tensor, hx=None):
        logits, value, hx_new = self.worker(z, goal.detach(), hx=hx)
        return logits, value, hx_new

    def worker_sample(self, z: torch.Tensor, goal: torch.Tensor, hx=None):
        logits, val, hx_new = self.worker(z, goal.detach(), hx=hx)
        dist = Categorical(logits=logits)
        act  = dist.sample()
        lp   = dist.log_prob(act)
        return act, lp, logits, val, hx_new

    def init_manager_hidden(self, batch_size: int = 1, device=None):
        if self.function_type != "recurrent":
            return None
        dev = device or next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.g_dim, device=dev)
        c = torch.zeros(1, batch_size, self.g_dim, device=dev)
        return h, c

    def init_worker_hidden(self, batch_size: int = 1, device=None):
        if self.function_type != "recurrent":
            return None
        dev = device or next(self.parameters()).device
        size = self.worker.action_dim * self.worker.k_dim
        h = torch.zeros(1, batch_size, size, device=dev)
        c = torch.zeros(1, batch_size, size, device=dev)
        return h, c

    def freeze_manager(self):
        for p in self.manager.parameters():
            p.requires_grad = False

    def unfreeze_manager(self):
        for p in self.manager.parameters():
            p.requires_grad = True

    def freeze_worker(self):
        for p in self.worker.parameters():
            p.requires_grad = False

    def unfreeze_worker(self):
        for p in self.worker.parameters():
            p.requires_grad = True

    # ── Inferència ────────────────────────────────────────────────────────────

    def predict_step(
        self,
        obs,
        step_ep: int,
        c: int,
        cur_goal,
        device,
        deterministic: bool = True,
        mgr_hx=None,
        wkr_hx=None,
    ) -> Tuple:
        """Un pas d'inferència jeràrquica.

        Accepta hidden states externs (mgr_hx, wkr_hx) per walk_forward_eval_hrl.
        Quan no es passen, gestiona els estados internament (HRLAgentWrapper mode).
        Retorna sempre (action, cur_goal, mgr_hx, wkr_hx).
        """
        # Mode extern: usar hidden states passats com a argument
        # Mode intern: usar self._mgr_hx / self._wkr_hx
        external = mgr_hx is not None or wkr_hx is not None
        if not external:
            if cur_goal is None:
                self._mgr_hx = None
                self._wkr_hx = None
            mgr_hx = self._mgr_hx
            wkr_hx = self._wkr_hx

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            z = self.perceive(obs_t)
            if cur_goal is None or step_ep % c == 0:
                goal_t, _, mgr_hx = self.manager_forward(z, hx=mgr_hx)
                cur_goal = goal_t.squeeze(0).cpu().numpy()
            g_t    = torch.tensor(cur_goal, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _, wkr_hx = self.worker_forward(z, g_t, hx=wkr_hx)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = Categorical(logits=logits).sample().item()

        if not external:
            self._mgr_hx = mgr_hx
            self._wkr_hx = wkr_hx
        return action, cur_goal, mgr_hx, wkr_hx

    # ── Utils ─────────────────────────────────────────────────────────────────

    def count_params(self) -> dict:
        def _n(it):
            return sum(p.numel() for p in it if p.requires_grad)
        return {
            "perception": _n(self.perception.parameters()),
            "manager":    _n(self.manager.parameters()),
            "worker":     _n(self.worker.parameters()),
            "total":      _n(self.parameters()),
        }

    # ── Checkpoint ────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path, device="cpu") -> "FeUdalAgent":
        """Carrega des d'un checkpoint 05d (auto-detecta l'arquitectura).

        El checkpoint ha de contenir:
            'model_state_dict', 'obs_dim', 'n_actions'
        """
        ckpt      = torch.load(str(path), map_location=device, weights_only=False)
        sd        = ckpt["model_state_dict"]
        obs_dim   = int(ckpt["obs_dim"])
        n_actions = int(ckpt["n_actions"])

        z_dim, perception_hidden = _infer_perception_dims(sd)
        g_dim          = sd["manager.fMspace.0.weight"].shape[0]
        k_dim          = sd["worker.phi.weight"].shape[0]
        fwpre_hidden   = sd["worker.fWpre.0.weight"].shape[0]
        value_hidden   = _infer_sequential_hidden(sd, "manager.value_net")
        function_type  = (
            "recurrent" if "manager.fMrnn.weight_ih_l0" in sd else "fc_lineal"
        )

        agent = cls(
            obs_dim=obs_dim,
            action_dim=n_actions,
            z_dim=z_dim,
            g_dim=g_dim,
            k_dim=k_dim,
            perception_hidden=perception_hidden,
            value_hidden=value_hidden,
            fwpre_hidden=fwpre_hidden,
            function_type=function_type,
        )
        agent.load_state_dict(sd)
        agent.eval().to(device)
        return agent


# ── Helpers d'inferència d'arquitectura ──────────────────────────────────────

def _infer_perception_dims(sd: dict) -> Tuple[int, List[int]]:
    """Infereix z_dim i perception_hidden des del state dict."""
    linear_keys = sorted(
        [k for k in sd if k.startswith("perception.net.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[-2]),
    )
    z_dim            = sd[linear_keys[-1]].shape[0]
    perception_hidden = [sd[k].shape[0] for k in linear_keys[:-1]]
    return z_dim, perception_hidden


def _infer_sequential_hidden(sd: dict, prefix: str) -> List[int]:
    """Infereix les capes ocultes intermèdies d'un Sequential Linear+Tanh."""
    linear_keys = sorted(
        [
            k for k in sd
            if k.startswith(prefix + ".") and k.endswith(".weight")
            and k.split(".")[-2].isdigit()
        ],
        key=lambda k: int(k.split(".")[-2]),
    )
    # Totes les capes excepte l'última (output) són capes ocultes
    return [sd[k].shape[0] for k in linear_keys[:-1]]
