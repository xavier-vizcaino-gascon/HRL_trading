"""
FeudalAgentV8 — HRL FeUdal v8 (05d_8, paper final, dense only).

Implementació autònoma extreta del notebook 05d_8_hrl_FeUdal_agent.ipynb.

Diferències clau respecte versions anteriors:
  - PerceptionModule: obs → hidden×2 (Tanh) → z (Linear) → ReLU
  - ManagerModule.fMrnn: DilatedLSTM (LSTMCell amb radius=c)
  - WorkerModule: sense fWpre — z va directament al LSTM

Usage:
    from src.rl.agents.feudal_agent_v8 import FeudalAgentV8
    agent = FeudalAgentV8.load("checkpoints_dense/hrl_dense_feudal_agent_paper_final/hrl_feudal_best.pt")
    action, goal, *_ = agent.predict_step(obs, step_ep, c, cur_goal, device)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ── Mòduls d'arquitectura ─────────────────────────────────────────────────────

class PerceptionModule(nn.Module):
    """Percepció compartida: obs_dim → hidden×2 (Tanh) → z_dim (Linear) → ReLU."""

    def __init__(self, obs_dim: int, z_dim: int, hidden: List[int]):
        super().__init__()
        dims = [obs_dim] + hidden + [z_dim]
        layers: list = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilatedLSTM(nn.Module):
    """Dilated LSTM (Vezhnevets et al. 2017 / feudalnets-pytorch).

    A cada timestep actualitza la slice step_idx % radius.
    Retorna el pooling de les radius slices.
    """

    def __init__(self, input_size: int, hidden_size: int, radius: int):
        super().__init__()
        self.radius = radius
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, hidden=None, step_idx: int = 0):
        B = x.size(0)
        idx = step_idx % self.radius

        if hidden is None:
            hx = torch.zeros(B, self.radius, self.hidden_size, device=x.device, dtype=x.dtype)
            cx = torch.zeros(B, self.radius, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            hx, cx = hidden

        new_hx_slice, new_cx_slice = self.rnn(x, (hx[:, idx, :], cx[:, idx, :]))

        hx_parts = [hx[:, i, :] if i != idx else new_hx_slice for i in range(self.radius)]
        cx_parts = [cx[:, i, :] if i != idx else new_cx_slice for i in range(self.radius)]
        new_hx = torch.stack(hx_parts, dim=1)
        new_cx = torch.stack(cx_parts, dim=1)

        if self.radius > 1:
            other_mask = torch.arange(self.radius, device=x.device) != idx
            sum_other = new_hx[:, other_mask, :].detach().sum(dim=1)
            y = (new_hx_slice + sum_other) / self.radius
        else:
            y = new_hx_slice

        return y, (new_hx, new_cx)


class ManagerModule(nn.Module):
    """Manager: genera goals normalitzats (DilatedLSTM) i estima V^M."""

    def __init__(
        self,
        z_dim: int,
        g_dim: int,
        value_hidden: List[int],
        function_type: str = "recurrent",
        c: int = 5,
    ):
        super().__init__()
        self.function_type = function_type

        self.fMspace = nn.Sequential(nn.Linear(z_dim, g_dim), nn.Tanh())

        if self.function_type == "recurrent":
            self.fMrnn = DilatedLSTM(input_size=g_dim, hidden_size=g_dim, radius=c)
        else:
            self.fMrnn = nn.Sequential(
                nn.Linear(g_dim, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, g_dim),
            )

        value_dims = [z_dim] + value_hidden + [1]
        layers: list = []
        for i in range(len(value_dims) - 2):
            layers.append(nn.Linear(value_dims[i], value_dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(value_dims[-2], value_dims[-1]))
        self.value_net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, hx=None, step_idx: int = 0) -> dict:
        st = self.fMspace(z)
        if self.function_type == "recurrent":
            g_hat, new_hx = self.fMrnn(st, hidden=hx, step_idx=step_idx)
        else:
            g_hat = self.fMrnn(st)
            new_hx = None
        goal = F.normalize(g_hat, dim=-1)
        value = self.value_net(z)
        return {"goal": goal, "value": value, "st": st, "hx": new_hx}


class WorkerModule(nn.Module):
    """Worker: selecciona accions via U·w (FuN eq. 3).

    Sense fWpre — z va directament al LSTM (paper FuN, feudalnets-pytorch).
    """

    def __init__(
        self,
        z_dim: int,
        g_dim: int,
        k_dim: int,
        n_actions: int,
        value_hidden: List[int],
        function_type: str = "recurrent",
    ):
        super().__init__()
        self.action_dim = n_actions
        self.k_dim = k_dim
        self.function_type = function_type

        output_dim = n_actions * k_dim
        if self.function_type == "recurrent":
            self.fWrnn = nn.LSTM(input_size=z_dim, hidden_size=output_dim,
                                 num_layers=1, batch_first=True)
        else:
            self.fWrnn = nn.Sequential(
                nn.Linear(z_dim, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(),
                nn.Linear(32, output_dim),
            )

        self.phi = nn.Linear(g_dim, k_dim, bias=False)

        value_dims = [z_dim] + value_hidden + [1]
        layers: list = []
        for i in range(len(value_dims) - 2):
            layers.append(nn.Linear(value_dims[i], value_dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(value_dims[-2], value_dims[-1]))
        self.value_net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, goal: torch.Tensor, hx=None) -> Tuple:
        B = z.size(0)
        if self.function_type == "recurrent":
            U_seq, (h_n, c_n) = self.fWrnn(z.unsqueeze(1), hx)
            U_flat = U_seq.squeeze(1)
            new_hx = (h_n, c_n)
        else:
            U_flat = self.fWrnn(z)
            new_hx = None
        U = U_flat.view(B, self.action_dim, self.k_dim)
        w = self.phi(goal)
        logits = torch.bmm(U, w.unsqueeze(-1)).squeeze(-1)
        value = self.value_net(z)
        return logits, value, new_hx


# ── FeudalAgentV8 ─────────────────────────────────────────────────────────────

class FeudalAgentV8(nn.Module):
    """Agent HRL FeUdal v8 (paper final, DilatedLSTM manager, worker sense fWpre)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        g_dim: int,
        k_dim: int,
        perception_hidden: List[int],
        value_hidden: List[int],
        function_type: str = "recurrent",
        c: int = 5,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.z_dim      = z_dim
        self.g_dim      = g_dim
        self.k_dim      = k_dim
        self.c          = c
        self.function_type = function_type

        self.perception = PerceptionModule(obs_dim, z_dim, perception_hidden)
        self.manager    = ManagerModule(
            z_dim=z_dim, g_dim=g_dim,
            value_hidden=value_hidden,
            function_type=function_type,
            c=c,
        )
        self.worker = WorkerModule(
            z_dim=z_dim, g_dim=g_dim, k_dim=k_dim, n_actions=action_dim,
            value_hidden=value_hidden,
            function_type=function_type,
        )

        self._mgr_hx = None
        self._wkr_hx = None

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

    def perceive(self, obs: torch.Tensor) -> torch.Tensor:
        return self.perception(obs)

    def phi_z(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.manager.fMspace(z)

    def manager_forward(self, z: torch.Tensor, hx=None, step_idx: int = 0):
        out = self.manager(z, hx=hx, step_idx=step_idx)
        return out["goal"], out["value"], out["hx"], out["st"]

    def manager_sample(self, z: torch.Tensor, hx=None):
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
        hx = torch.zeros(batch_size, self.c, self.g_dim, device=dev)
        cx = torch.zeros(batch_size, self.c, self.g_dim, device=dev)
        return hx, cx

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
                goal_t, _, mgr_hx, _ = self.manager_forward(z, hx=mgr_hx, step_idx=step_ep)
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

    def count_params(self) -> dict:
        def _n(it):
            return sum(p.numel() for p in it if p.requires_grad)
        return {
            "perception": _n(self.perception.parameters()),
            "manager":    _n(self.manager.parameters()),
            "worker":     _n(self.worker.parameters()),
            "total":      _n(self.parameters()),
        }

    @classmethod
    def load(cls, path, device="cpu") -> "FeudalAgentV8":
        ckpt      = torch.load(str(path), map_location=device, weights_only=False)
        sd        = ckpt["model_state_dict"]
        obs_dim   = int(ckpt["obs_dim"])
        n_actions = int(ckpt["n_actions"])
        c         = int(ckpt.get("config", {}).get("c", 5))

        z_dim, perception_hidden = _infer_perception_dims(sd)
        g_dim        = sd["manager.fMspace.0.weight"].shape[0]
        k_dim        = sd["worker.phi.weight"].shape[0]
        value_hidden = _infer_sequential_hidden(sd, "manager.value_net")
        function_type = (
            "recurrent" if "manager.fMrnn.rnn.weight_ih" in sd else "fc_lineal"
        )

        agent = cls(
            obs_dim=obs_dim, action_dim=n_actions,
            z_dim=z_dim, g_dim=g_dim, k_dim=k_dim,
            perception_hidden=perception_hidden,
            value_hidden=value_hidden,
            function_type=function_type,
            c=c,
        )
        agent.load_state_dict(sd)
        agent.eval().to(device)
        return agent


# ── Helpers d'inferència d'arquitectura ──────────────────────────────────────

def _infer_perception_dims(sd: dict) -> Tuple[int, List[int]]:
    linear_keys = sorted(
        [k for k in sd if k.startswith("perception.net.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[-2]),
    )
    z_dim             = sd[linear_keys[-1]].shape[0]
    perception_hidden = [sd[k].shape[0] for k in linear_keys[:-1]]
    return z_dim, perception_hidden


def _infer_sequential_hidden(sd: dict, prefix: str) -> List[int]:
    linear_keys = sorted(
        [
            k for k in sd
            if k.startswith(prefix + ".") and k.endswith(".weight")
            and k.split(".")[-2].isdigit()
        ],
        key=lambda k: int(k.split(".")[-2]),
    )
    return [sd[k].shape[0] for k in linear_keys[:-1]]
