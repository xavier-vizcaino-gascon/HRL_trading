"""
FeudalAgentV0 — HRL FeUdal v0 (05d / 05s baseline, LeakyReLU MLP).

El checkpoint v0 va ser guardat amb la classe localment definida al notebook
(arquitectura idèntica a v1: Sequential LeakyReLU), NO amb FeudalAgent de
feudal_agent.py. Per tant, FeudalAgentV0 hereta de FeudalAgentV1 i load()
auto-detecta les dimensions des del state dict.

Usage:
    from src.rl.agents.feudal_agent_v0 import FeudalAgentV0
    agent = FeudalAgentV0.load("checkpoints_dense/hrl_dense_feudal_agent/hrl_feudal_best.pt")
    action, goal = agent.predict_step(obs, step_ep, c, cur_goal, device)
"""

from .feudal_agent_v1 import FeudalAgentV1


class FeudalAgentV0(FeudalAgentV1):
    """HRL FeUdal v0 — arquitectura LeakyReLU MLP (idèntica a v1)."""
