"""
ActorCriticPolicy — SB3 MlpPolicy equivalent, HRL-ready.

Còpia literal de la classe definida al notebook 03b_baseline_agent_extended.ipynb,
extreta a src/ perquè pugui ser importada per qualsevol notebook d'avaluació o
agent HRL sense haver de definir la classe in-line.

Punt d'entrada principal per a avaluació (notebook 10_evaluation):
    from src.rl.agents.actor_critic_policy import ActorCriticPolicy
    policy = ActorCriticPolicy.load("checkpoints/ppo_pytorch_baseline_with_a_params.pt")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

if TYPE_CHECKING:
    pass


class ActorCriticPolicy(nn.Module):
    """
    SB3-equivalent MlpPolicy amb backbones π / V separats.

    Característiques HRL
    ────────────────────
    • Named layers  → weight sharing / freezing quirúrgic per a options
    • Forward hooks → captura d'activacions per capa (get_activations())
    • Goal slot     → vector de goal injectat al backbone π
                      (activa amb set_goal(g) per a HRL option policies)

    Equivalència SB3 MlpPolicy
    ──────────────────────────
    • policy_net / value_net separats (ambdós llegeixen obs crua)
    • Init ortogonal: √2 backbone, 0.01 action head, 1.0 value head
    • Activació Tanh per defecte
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        net_arch: list,
        activation_fn: type = nn.Tanh,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._goal: torch.Tensor | None = None
        self._activations: dict[str, torch.Tensor] = {}

        def _mlp(input_dim: int, hidden_dims: list) -> nn.Sequential:
            layers, in_d = [], input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_d, h))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h))
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_d = h
            return nn.Sequential(*layers)

        last_dim = net_arch[-1] if net_arch else obs_dim
        self.policy_net = _mlp(obs_dim, net_arch)   # backbone π
        self.value_net  = _mlp(obs_dim, net_arch)   # backbone V
        self.action_net = nn.Linear(last_dim, act_dim)
        self.value_head = nn.Linear(last_dim, 1)

        # Init ortogonal (SB3 defaults) - només per nn.Linear
        for module in list(self.policy_net) + list(self.value_net):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.zeros_(self.action_net.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        # Hooks d'activació per a visibilitat HRL
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(self._make_hook(name))

    # ── HRL helpers ──────────────────────────────────────────────────────────

    def _make_hook(self, name: str):
        def hook(_, __, output):
            self._activations[name] = output.detach()
        return hook

    def set_goal(self, goal: torch.Tensor | None):
        """Injecta un vector de goal d'opció (HRL placeholder – no-op fins que s'activi)."""
        self._goal = goal

    def get_activations(self) -> dict[str, torch.Tensor]:
        """
        Retorna un snapshot de les activacions de l'últim forward.
        Claus: noms de capa (p.ex. 'policy_net.0', 'action_net').
        Ús HRL: el meta-controlador pot inspeccionar el feature space de l'opció.
        """
        return {k: v.clone() for k, v in self._activations.items()}

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (logits, value)."""
        pi_feat = self.policy_net(obs)
        vf_feat = self.value_net(obs)
        return self.action_net(pi_feat), self.value_head(vf_feat).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(values, log_probs, entropy) – usat a l'update PPO."""
        logits, values = self.forward(obs)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return values, log_probs, entropy

    # ── SB3-compatible predict ───────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """
        Interfície SB3-compatible: retorna (action_array, state=None).
        Funciona amb el bucle d'avaluació de NB10 sense cap canvi.
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32,
                device=next(self.parameters()).device,
            )
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            logits, _ = self.forward(obs_t)
            action = (
                logits.argmax(dim=-1) if deterministic
                else Categorical(logits=logits).sample()
            )
        return action.cpu().numpy().squeeze(), None

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Càrrega des de checkpoint ─────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        activation_fn: type = nn.Tanh,
    ) -> "ActorCriticPolicy":
        """
        Carrega una política des d'un checkpoint generat per CustomPPO.save().

        El checkpoint ha de contenir:
            {
              "policy_state_dict": <state_dict>,
              "hyperparams": {"obs_dim": int, "act_dim": int, ...},
            }

        L'arquitectura (net_arch) s'infereix automàticament del state_dict
        llegint les dimensions de sortida de cada capa de policy_net.

        Args:
            path:          Ruta al fitxer .pt
            device:        Dispositiu on carregar el model
            activation_fn: Funció d'activació (ha de coincidir amb l'entrenament)

        Returns:
            ActorCriticPolicy en mode eval, al dispositiu indicat.
        """
        import re

        ckpt = torch.load(str(path), map_location=device, weights_only=False)

        hp = ckpt.get("hyperparams", {})
        sd = ckpt["policy_state_dict"]

        # Inferència sempre des de l'state_dict (font de veritat).
        # Els hyperparams guardats poden no reflectir l'arquitectura real
        # (p.ex. checkpoints meta-learning hereten pesos del baseline).
        param_indices: dict[int, torch.Tensor] = {}
        for key in sd:
            m = re.match(r"policy_net\.(\d+)\.weight$", key)
            if m:
                param_indices[int(m.group(1))] = sd[key]

        linear_idxs    = sorted(i for i, w in param_indices.items() if w.dim() == 2)
        use_layer_norm = any(w.dim() == 1 for w in param_indices.values())
        net_arch       = [param_indices[i].shape[0] for i in linear_idxs]

        # Inferir si hi havia Dropout: amb LN el pas és 3 (L,LN,Act) o 4 (L,LN,Act,Drop);
        # sense LN el pas és 2 (L,Act) o 3 (L,Act,Drop).
        if len(linear_idxs) >= 2:
            step = linear_idxs[1] - linear_idxs[0]
            has_dropout = (step == 4) if use_layer_norm else (step == 3)
        else:
            has_dropout = False
        dropout = 0.1 if has_dropout else 0.0

        # obs_dim i act_dim des dels pesos del primer/últim layer.
        if linear_idxs:
            obs_dim = param_indices[linear_idxs[0]].shape[1]
        else:
            obs_dim = int(hp.get("obs_dim", 71))

        if "action_net.weight" in sd:
            act_dim = sd["action_net.weight"].shape[0]
        else:
            act_dim = int(hp.get("act_dim", 4))

        policy = cls(obs_dim, act_dim, net_arch, activation_fn, use_layer_norm, dropout)
        policy.load_state_dict(sd)
        policy.eval().to(device)

        return policy
