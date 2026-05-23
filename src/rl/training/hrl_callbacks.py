"""Callbacks SB3-compatibles per a l'entrenament HRL FeUdal.

HrlEvalCallback   — avaluació periòdica + integració Optuna.
FeeCurriculumCallback — curriculum progressiu de comissions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.utils import fee_curriculum_factor

if TYPE_CHECKING:
    import polars as pl


class HrlEvalCallback(BaseCallback):
    """Callback SB3-compatible per avaluació periòdica durant train_hrl.

    Cada eval_freq passos:
      1. Avalua l'agent (FeudalAgent) sobre df_vl
      2. Desa mètriques a self.history
      3. Si el retorn millora, desa el millor state_dict
      4. Si trial no és None: report() + should_prune() (Optuna)

    Args:
        evaluate_fn: callable(agent, df, n_episodes, device, c, seed) -> dict
        df_vl:       DataFrame de validació
        eval_freq:   cada quants timesteps avaluar
        n_eval_eps:  episodis per avaluació
        device:      dispositiu torch
        c:           periode del Manager
        seed:        seed base (incrementat cada avaluació)
        verbose:     0 = silenciós
        trial:       optuna.Trial opcional
    """

    def __init__(
        self,
        evaluate_fn: Callable,
        df_vl: "pl.DataFrame",
        eval_freq: int,
        n_eval_eps: int,
        device: "torch.device | str",
        c: int,
        seed: int = 42,
        verbose: int = 0,
        trial=None,
        writer=None,
    ):
        super().__init__(verbose)
        self.evaluate_fn  = evaluate_fn
        self.df_vl        = df_vl
        self.eval_freq    = eval_freq
        self.n_eval_eps   = n_eval_eps
        self.device       = device
        self.c            = c
        self.seed         = seed
        self.trial        = trial
        self.writer       = writer
        self.history: list[dict] = []
        self._best_return = -np.inf
        self._best_state  = None
        self._eval_count  = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            m = self.evaluate_fn(
                self.model, self.df_vl, self.n_eval_eps,
                self.device, self.c, seed=self.seed + self.n_calls,
            )
            m["timestep"] = self.num_timesteps
            self.history.append(m)
            if m["mean_return"] > self._best_return:
                self._best_return = m["mean_return"]
                self._best_state  = {k: v.clone()
                                     for k, v in self.model.state_dict().items()}
            if self.writer is not None:
                self.writer.add_scalar("val/mean_return", m["mean_return"], self.num_timesteps)
                self.writer.add_scalar("val/sharpe", m["sharpe"], self.num_timesteps)
                self.writer.add_scalar("val/max_drawdown", m["max_drawdown"], self.num_timesteps)
                self.writer.add_scalar("val/position_frac", m["position_frac"], self.num_timesteps)
                if "action_dist_pct" in m:
                    for act_name, pct in m["action_dist_pct"].items():
                        self.writer.add_scalar(f"val_actions/{act_name}_pct", pct, self.num_timesteps)
            if self.trial is not None:
                self.trial.report(m["mean_return"], self._eval_count)
                self._eval_count += 1
                if self.trial.should_prune():
                    return False
            if self.verbose > 0:
                print(
                    f"  ts={self.num_timesteps:>8,} | ret={m['mean_return']:+.2f}% | "
                    f"sharpe={m['sharpe']:+.4f} | dd={m['max_drawdown']:.2f}%"
                )
        return True


class FeeCurriculumCallback(BaseCallback):
    """Curriculum progressiu de comissions durant train_hrl.

    Fases:
      - Exploració   (0–fee_warmup_frac):           fee_factor = 0.0
      - Rampa        (fee_warmup_frac–fee_ramp_end): fee_factor 0 → 1
      - Consolidació (fee_ramp_end+):                fee_factor = 1.0
    """

    def __init__(
        self,
        total_timesteps: int,
        env_ref: list,
        fee_warmup_frac: float = 0.40,
        fee_ramp_end_frac: float = 0.60,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps  = total_timesteps
        self._env_ref         = env_ref
        self.fee_warmup_frac  = fee_warmup_frac
        self.fee_ramp_end_frac = fee_ramp_end_frac

    def _on_step(self) -> bool:
        factor = fee_curriculum_factor(
            self.num_timesteps, self.total_timesteps,
            self.fee_warmup_frac, self.fee_ramp_end_frac,
        )
        if self._env_ref:
            self._env_ref[0].fee_factor = factor
        return True
