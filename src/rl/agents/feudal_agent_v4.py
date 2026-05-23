"""
FeudalAgentV4 — HRL FeUdal v4 (05d_4 / 05s_4).

Thin subclass of FeUdalAgent (feudal_agent_paper.py).
load() auto-detects architecture from the checkpoint state dict.

Usage:
    from src.rl.agents.feudal_agent_v4 import FeudalAgentV4
    agent = FeudalAgentV4.load("checkpoints_dense/hrl_dense_feudal_agent_paper/hrl_feudal_best.pt")
    action, goal, *_ = agent.predict_step(obs, step_ep, c, cur_goal, device)
"""

from .feudal_agent_paper import FeUdalAgent


class FeudalAgentV4(FeUdalAgent):
    """HRL FeUdal v4 — re-export of FeUdalAgent (paper architecture)."""
