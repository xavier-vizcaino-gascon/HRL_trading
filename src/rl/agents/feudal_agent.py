"""
FeudalAgent — FeUdal Networks HRL, arquitectura multiplicativa (04a_hrl_x_agent).

Còpia literal de les classes definides al notebook 04a_hrl_x_agent.ipynb,
extreta a src/ perquè pugui ser importada per qualsevol notebook d'avaluació.

Arquitectura (04a, ~37k params):
  • Percepció (compartida):  obs_dim → 64 → 32 (z_dim)  (tanh)
  • φ (phi_proj, entrenable): z(32) → g_space(16)  (sense bias, entrenat amb Manager)
  • Manager policy:          z(32) → 64 → 64 → 16 (g_dim)  (L2-normalitzat)
  • Manager value:           z(32) → 64 → 64 → 1
  • Worker policy (U·w):     z(32) → 64 → 64 → action_dim*8 → reshape U ∈ ℝ^(A×K)
                             g(16) → 8 (goal_embedding, sense bias) → w ∈ ℝ^K
                             logits = U·w → SoftMax → π
  • Worker value:            Concat(z, g) [48] → 64 → 64 → 1

Principis FuN (Vezhnevets et al. 2017):
  • Goal normalitzat L2: g = ĝ/||ĝ|| (eq. 2)
  • No gradient del Worker al Manager: goal.detach()
  • Reward intrínsec: cos_sim(φ(z_t) - φ(z_{t-c}), g_{t-c})  (eq. 9)
  • φ entrenable (NOVETAT 04a): l'espai de goals s'adapta amb el Manager

Punt d'entrada:
    from src.rl.agents.feudal_agent import FeudalAgent
    agent = FeudalAgent.load("checkpoints/hrl_multiplicative/hrl_feudal_best.pt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

if TYPE_CHECKING:
    pass


# ── Mòduls d'arquitectura ─────────────────────────────────────────────────────

class PerceptionModule(nn.Module):
    """Percepció compartida: obs_dim → perception_hidden → z_dim (tanh)."""

    def __init__(self, obs_dim: int, z_dim: int, perception_hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, perception_hidden)
        self.fc2 = nn.Linear(perception_hidden, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        return torch.tanh(self.fc2(x))


class ManagerModule(nn.Module):
    """Manager: genera goals normalitzats i estima V^M.

    Policy: z_dim → hidden[0] → hidden[1] → g_dim  (L2-norm, eq. 2 FuN)
    Value:  z_dim → hidden[0] → hidden[1] → 1
    """

    def __init__(self, z_dim: int, g_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.policy_fc1 = nn.Linear(z_dim, hidden_dims[0])
        self.policy_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.goal_head  = nn.Linear(hidden_dims[1], g_dim)

        self.value_fc1  = nn.Linear(z_dim, hidden_dims[0])
        self.value_fc2  = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.value_head = nn.Linear(hidden_dims[1], 1)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h1    = torch.tanh(self.policy_fc1(z))
        h2    = torch.tanh(self.policy_fc2(h1))
        g_hat = self.goal_head(h2)
        goal  = F.normalize(g_hat, dim=-1)          # g = ĝ/||ĝ||

        v1    = torch.tanh(self.value_fc1(z))
        v2    = torch.tanh(self.value_fc2(v1))
        value = self.value_head(v2)

        return {"goal": goal, "value": value}


class WorkerModule(nn.Module):
    """Worker amb integració multiplicativa del goal (U·w).

    Policy:  z → FC → FC → FC → reshape U ∈ ℝ^(A×K)
             g → φ (FC g_dim→k_dim, sense bias) → w ∈ ℝ^K
             logits = U·w  →  SoftMax → π
    Value:   Concat(z, g) → FC → FC → 1
    """

    def __init__(
        self,
        z_dim: int,
        g_dim: int,
        k_dim: int,
        action_dim: int,
        hidden_dims: List[int],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.k_dim      = k_dim

        self.policy_fc1   = nn.Linear(z_dim, hidden_dims[0])
        self.policy_fc2   = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.action_embed = nn.Linear(hidden_dims[1], action_dim * k_dim)

        self.goal_embedding = nn.Linear(g_dim, k_dim, bias=False)   # φ, sense bias

        concat_dim      = z_dim + g_dim
        self.value_fc1  = nn.Linear(concat_dim, hidden_dims[0])
        self.value_fc2  = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.value_head = nn.Linear(hidden_dims[1], 1)

    def forward(self, z: torch.Tensor, goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = z.size(0)

        h1     = torch.tanh(self.policy_fc1(z))
        h2     = torch.tanh(self.policy_fc2(h1))
        U_flat = self.action_embed(h2)
        U      = U_flat.view(batch_size, self.action_dim, self.k_dim)
        w      = self.goal_embedding(goal)
        logits = torch.bmm(U, w.unsqueeze(-1)).squeeze(-1)

        zg     = torch.cat([z, goal], dim=-1)
        v1     = torch.tanh(self.value_fc1(zg))
        v2     = torch.tanh(self.value_fc2(v1))
        value  = self.value_head(v2)

        return {"logits": logits, "value": value}


# ── FeudalAgent ───────────────────────────────────────────────────────────────

class FeudalAgent(nn.Module):
    """
    Agent HRL FeUdal amb integració multiplicativa del goal.

    Principis FuN (Vezhnevets et al. 2017):
    - Manager genera goals NORMALITZATS g = ĝ/||ĝ|| (eq. 2)
    - Worker rep goal via goal.detach() → cap gradient del Worker al Manager
    - Reward intrínsec = cosine similarity entre desplaçaments i goal (eq. 9)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int             = 32,
        g_dim: int             = 16,
        k_dim: int             = 8,
        perception_hidden: int = 64,
        manager_hidden: List[int] = None,
        worker_hidden:  List[int] = None,
    ):
        super().__init__()
        if manager_hidden is None:
            manager_hidden = [64, 64]
        if worker_hidden is None:
            worker_hidden = [64, 64]

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.z_dim      = z_dim
        self.g_dim      = g_dim
        self.k_dim      = k_dim

        self.perception = PerceptionModule(obs_dim, z_dim, perception_hidden)
        self.manager    = ManagerModule(z_dim=z_dim, g_dim=g_dim, hidden_dims=manager_hidden)
        self.worker     = WorkerModule(z_dim=z_dim, g_dim=g_dim, k_dim=k_dim,
                                       action_dim=action_dim, hidden_dims=worker_hidden)

        # φ: projecció z → espai dels goals (FuN sec. 3.1), ENTRENABLE pel Manager
        import math
        self.phi_proj = nn.Linear(z_dim, g_dim, bias=False)
        with torch.no_grad():
            # Random init amb seed 42 (compatibilitat amb 04a)
            _g = torch.Generator().manual_seed(42)
            self.phi_proj.weight.copy_(
                torch.empty(g_dim, z_dim).normal_(0.0, 1.0 / math.sqrt(z_dim), generator=_g)
            )

        self.goal_log_std = nn.Parameter(torch.zeros(g_dim))

    # ── Core forward ──────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        z           = self.perception(obs)
        manager_out = self.manager(z)
        goal        = manager_out["goal"]
        worker_out  = self.worker(z, goal.detach())   # no gradient Worker→Manager
        return {
            "logits":        worker_out["logits"],
            "worker_value":  worker_out["value"],
            "goal":          goal,
            "manager_value": manager_out["value"],
            "z":             z,
        }

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, Dict]:
        with torch.no_grad():
            outputs = self.forward(obs.unsqueeze(0))
            probs   = F.softmax(outputs["logits"], dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                action = Categorical(probs).sample().item()
            return action, outputs

    # ── Wrappers per train_hrl / ppo_update ───────────────────────────────────

    def perceive(self, obs: torch.Tensor) -> torch.Tensor:
        return self.perception(obs)

    def phi_z(self, z: torch.Tensor) -> torch.Tensor:
        """s_t = φ(z_t): projecció a l'espai dels goals (FuN, eq. 9).

        Amb phi_proj ENTRENABLE (no-grad aquí per rollout + logging).
        """
        with torch.no_grad():
            return self.phi_proj(z)

    def manager_forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.manager(z)
        return out["goal"], out["value"]

    def manager_sample(self, z: torch.Tensor):
        """Pas estocàstic del Manager. Retorna (g_norm, g_raw, lp, val, g_det)."""
        out   = self.manager(z)
        g_det = out["goal"]
        val   = out["value"]
        std   = self.goal_log_std.exp().expand_as(g_det)
        dist  = Normal(g_det, std)
        g_raw = dist.rsample()
        lp    = dist.log_prob(g_raw).sum(-1, keepdim=True)
        g_norm = F.normalize(g_raw, dim=-1)
        return g_norm, g_raw, lp, val, g_det

    def worker_forward(self, z: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.worker(z, goal.detach())
        return out["logits"], out["value"]

    def worker_sample(self, z: torch.Tensor, goal: torch.Tensor):
        """Pas estocàstic del Worker. Retorna (act, lp, logits, val)."""
        out    = self.worker(z, goal.detach())
        logits = out["logits"]
        val    = out["value"]
        dist   = Categorical(logits=logits)
        act    = dist.sample()
        lp     = dist.log_prob(act)
        return act, lp, logits, val

    def predict_step(
        self,
        obs: "np.ndarray",
        step_ep: int,
        c: int,
        cur_goal: "np.ndarray | None",
        device: "torch.device | str",
        deterministic: bool = True,
    ) -> "tuple[int, np.ndarray]":
        """Un pas d'inferència jeràrquica. Retorna (action, cur_goal)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            z = self.perceive(obs_t)
            if cur_goal is None or step_ep % c == 0:
                goal_t, _ = self.manager_forward(z)
                cur_goal  = goal_t.squeeze(0).cpu().numpy()
            g_t    = torch.tensor(cur_goal, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = self.worker_forward(z, g_t)
            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = Categorical(logits=logits).sample().item()
        return action, cur_goal

    # ── Utils ─────────────────────────────────────────────────────────────────

    def count_params(self) -> dict:
        def _n(it): return sum(p.numel() for p in it if p.requires_grad)
        return {
            "perception":   _n(self.perception.parameters()),
            "manager":      _n(self.manager.parameters()),
            "worker":       _n(self.worker.parameters()),
            "phi_proj":     _n(self.phi_proj.parameters()),
            "goal_log_std": self.goal_log_std.numel(),
            "total":        _n(self.parameters()),
        }

    # ── Checkpoint ────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: "str | Path",
        device: "str | torch.device" = "cpu",
        z_dim: int             = 32,
        g_dim: int             = 16,
        k_dim: int             = 8,
        perception_hidden: int = 64,
        manager_hidden: List[int] = None,
        worker_hidden:  List[int] = None,
    ) -> "FeudalAgent":
        """Carrega un FeudalAgent des d'un checkpoint de 04a_hrl_x_agent.

        El checkpoint conté:
            {
              'model_state_dict': <state_dict>,
              'config':           <best_params + training cfg>,
              'obs_dim':          int,
              'n_actions':        int,
            }

        L'arquitectura (z_dim, g_dim, k_dim, hidden) no s'inclou al config
        (són constants fixes del notebook), per tant es passen com a arguments
        amb els valors per defecte del notebook 04a (Z_DIM=32, G_DIM=16, K_DIM=8,
        perception_hidden=64, MANAGER_HIDDEN=[64,64], WORKER_HIDDEN=[64,64]).
        """
        if manager_hidden is None:
            manager_hidden = [64, 64]
        if worker_hidden is None:
            worker_hidden = [64, 64]

        ckpt      = torch.load(str(path), map_location=device, weights_only=False)
        obs_dim   = int(ckpt["obs_dim"])
        n_actions = int(ckpt["n_actions"])

        agent = cls(
            obs_dim=obs_dim,
            action_dim=n_actions,
            z_dim=z_dim,
            g_dim=g_dim,
            k_dim=k_dim,
            perception_hidden=perception_hidden,
            manager_hidden=manager_hidden,
            worker_hidden=worker_hidden,
        )
        agent.load_state_dict(ckpt["model_state_dict"])
        agent.eval().to(device)
        return agent
