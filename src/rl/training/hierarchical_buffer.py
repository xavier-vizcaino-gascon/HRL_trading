"""HierarchicalRolloutBuffer — buffer dual per a FeUdal HRL.

Worker: emmagatzema experiència pas a pas (n_steps entrades).
Manager: emmagatzema experiència cada c passos (n_mgr_max entrades).
Calcula GAE independent per a cada nivell jeràrquic.
"""

from __future__ import annotations

import math

import numpy as np


class HierarchicalRolloutBuffer:
    """Buffer dual (Worker pas a pas, Manager cada c passos)."""

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        latent_dim: int,
        goal_dim: int,
        c: int,
        device: "torch.device | str",
    ):
        self.n_steps    = n_steps
        self.obs_dim    = obs_dim
        self.latent_dim = latent_dim
        self.goal_dim   = goal_dim
        self.c          = c
        self.device     = device
        self.n_mgr_max  = math.ceil(n_steps / c) + 2
        self.reset()

    def reset(self) -> None:
        self.w_obs   = np.zeros((self.n_steps, self.obs_dim),    dtype=np.float32)
        self.w_z     = np.zeros((self.n_steps, self.latent_dim), dtype=np.float32)
        self.w_goals = np.zeros((self.n_steps, self.goal_dim),   dtype=np.float32)
        self.w_acts  = np.zeros(self.n_steps,                    dtype=np.int64)
        self.w_rews  = np.zeros(self.n_steps,                    dtype=np.float32)
        self.w_vals  = np.zeros(self.n_steps,                    dtype=np.float32)
        self.w_lps   = np.zeros(self.n_steps,                    dtype=np.float32)
        self.w_dones = np.zeros(self.n_steps,                    dtype=np.float32)
        self.w_ptr   = 0
        self.m_z     = np.zeros((self.n_mgr_max, self.latent_dim), dtype=np.float32)
        self.m_graws = np.zeros((self.n_mgr_max, self.goal_dim),   dtype=np.float32)
        self.m_rews  = np.zeros(self.n_mgr_max,                    dtype=np.float32)
        self.m_vals  = np.zeros(self.n_mgr_max,                    dtype=np.float32)
        self.m_lps   = np.zeros(self.n_mgr_max,                    dtype=np.float32)
        self.m_dones = np.zeros(self.n_mgr_max,                    dtype=np.float32)
        self.m_ptr   = 0

    def add_worker(
        self, obs, z, goal, act, rew, val, lp, done
    ) -> None:
        i = self.w_ptr
        self.w_obs[i]   = obs
        self.w_z[i]     = z
        self.w_goals[i] = goal
        self.w_acts[i]  = act
        self.w_rews[i]  = rew
        self.w_vals[i]  = val
        self.w_lps[i]   = lp
        self.w_dones[i] = done
        self.w_ptr     += 1

    def add_manager(self, z, g_raw, rew, val, lp, done) -> None:
        i = self.m_ptr
        self.m_z[i]     = z
        self.m_graws[i] = g_raw
        self.m_rews[i]  = rew
        self.m_vals[i]  = val
        self.m_lps[i]   = lp
        self.m_dones[i] = done
        self.m_ptr     += 1

    def compute_gae_worker(
        self, last_val: float, gamma: float, gae_lambda: float
    ) -> "tuple[np.ndarray, np.ndarray]":
        n = self.w_ptr
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nxt    = last_val if t == n - 1 else self.w_vals[t + 1]
            delta  = self.w_rews[t] + gamma * nxt * (1 - self.w_dones[t]) - self.w_vals[t]
            gae    = delta + gamma * gae_lambda * (1 - self.w_dones[t]) * gae
            adv[t] = gae
        return adv, adv + self.w_vals[:n]

    def compute_gae_manager(
        self, last_val: float, gamma_m: float, gae_lambda: float
    ) -> "tuple[np.ndarray, np.ndarray]":
        n = self.m_ptr
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nxt    = last_val if t == n - 1 else self.m_vals[t + 1]
            delta  = self.m_rews[t] + gamma_m * nxt * (1 - self.m_dones[t]) - self.m_vals[t]
            gae    = delta + gamma_m * gae_lambda * (1 - self.m_dones[t]) * gae
            adv[t] = gae
        return adv, adv + self.m_vals[:n]
