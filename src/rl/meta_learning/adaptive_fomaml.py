"""
Sistema de meta-entrenament FOMAML (First-Order MAML) ADAPTATIU.

FOMAML vs Reptile:
- Reptile: θ ← θ + ε·mean(φᵢ - θ)  (interpolació cap als paràmetres adaptats)
- FOMAML: θ ← θ - β∇θ L(φᵢ; D^query)  (gradient sobre loss de query amb φᵢ)

Característiques:
- Inner loops dinàmics: cada tasca s'entrena fins convergir
- Outer updates automàtics: quan s'acumulen N tasques o passa X temps
- Early stopping global: para quan el meta-agent convergeix
- Support/Query split per cada tasca
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Optional, Union
from dataclasses import dataclass

# Reutilitzem els mateixos criteris que Reptile
from src.rl.meta_learning.adaptive_reptile import (
    InnerConvergenceCriteria,
    OuterUpdateTrigger,
    MetaConvergenceCriteria,
    AdaptiveInnerLoop,
)


class AdaptiveFOMAML:
    """Sistema de meta-entrenament FOMAML adaptatiu."""

    def __init__(
        self,
        inner_criteria: InnerConvergenceCriteria,
        outer_trigger: OuterUpdateTrigger,
        meta_criteria: MetaConvergenceCriteria,
        device: torch.device,
        ppo_config: dict,
        meta_lr: float = 1e-4,  # Learning rate per outer update
        support_ratio: float = 0.7,  # 70% support, 30% query
        tensorboard_log: Optional[str] = None,
    ):
        self.inner_loop = AdaptiveInnerLoop(inner_criteria, device, ppo_config)
        self.outer_trigger = outer_trigger
        self.meta_criteria = meta_criteria
        self.device = device
        self.ppo_config = ppo_config
        self.meta_lr = meta_lr
        self.support_ratio = support_ratio
        self.tensorboard_log = tensorboard_log
        if tensorboard_log:
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_log)
        else:
            self.writer = None

    def meta_train(
        self,
        base_policy,
        task_sampler: Callable,  # fn() -> task_df
        env_factory: Callable,   # fn(df) -> env
        val_df,
        verbose: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        checkpoint_every: int = 50,
    ) -> dict:
        """
        Meta-entrenament adaptatiu complet amb FOMAML.

        Args:
            checkpoint_path: Ruta on desar/carregar checkpoints intermedis.
                             Si existeix en iniciar, es repren des d'allà.
            checkpoint_every: Desa checkpoint cada N meta-iters (default 50).

        Returns:
            {
                'meta_policy': ActorCriticPolicy,
                'meta_history': list,
                'inner_diagnostics': list,
            }
        """
        ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None

        meta_agent = copy.deepcopy(base_policy).to(self.device)
        meta_agent.train()

        # Optimizer per l'outer update
        meta_optimizer = torch.optim.Adam(meta_agent.parameters(), lr=self.meta_lr)

        meta_history = []
        inner_diagnostics = []

        best_val_sharpe = -np.inf
        best_state_dict = None
        evals_without_improvement = 0

        meta_iter = 0
        total_tasks_processed = 0

        # ── Resume des de checkpoint si existeix ──────────────────────────
        if ckpt_path is not None and ckpt_path.exists():
            _d = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            meta_agent.load_state_dict(_d['policy_state_dict'])
            meta_agent.train()
            if 'optimizer_state_dict' in _d:
                meta_optimizer.load_state_dict(_d['optimizer_state_dict'])
            meta_history           = _d.get('meta_history', [])
            inner_diagnostics      = _d.get('inner_diagnostics', [])
            meta_iter              = _d.get('meta_iter', 0)
            total_tasks_processed  = _d.get('total_tasks_processed', 0)
            best_val_sharpe        = _d.get('best_val_sharpe', -np.inf)
            best_state_dict        = _d.get('best_state_dict', None)
            evals_without_improvement = _d.get('evals_without_improvement', 0)
            if verbose:
                print(f'[FOMAML] Reprenent des de checkpoint: '
                      f'meta_iter={meta_iter}, best_sharpe={best_val_sharpe:+.4f}')

        while meta_iter < self.meta_criteria.max_meta_iters:
            # ── INNER LOOP BATCH ──────────────────────────────────────────
            adapted_policies = []  # Llista de (φᵢ, query_df) per outer update
            batch_start_time = time.time()

            while len(adapted_policies) < self.outer_trigger.batch_size:
                # Check timeout
                elapsed = time.time() - batch_start_time
                if (self.outer_trigger.max_time_seconds is not None and
                    elapsed > self.outer_trigger.max_time_seconds and
                    len(adapted_policies) >= self.outer_trigger.min_tasks):
                    if verbose:
                        print(f'  Batch timeout: {len(adapted_policies)} tasks in {elapsed:.1f}s')
                    break

                # Sample task i divideix en support/query
                task_df = task_sampler()
                total_tasks_processed += 1

                split_idx = int(len(task_df) * self.support_ratio)
                support_df = task_df[:split_idx]
                query_df = task_df[split_idx:]

                if len(support_df) < self.ppo_config['n_steps'] * 2:
                    if verbose:
                        print(f'  Task {total_tasks_processed}: SKIP (support massa petit)')
                    continue

                # Train inner loop fins convergència sobre SUPPORT
                result = self.inner_loop.train_until_convergence(
                    meta_agent, support_df, env_factory
                )

                # Guarda diagnòstics
                inner_diagnostics.append({
                    'meta_iter': meta_iter,
                    'task_idx': total_tasks_processed,
                    **result,
                })

                # Només usa si convergit correctament
                if result['converged'] and result['state_dict'] is not None:
                    if result['reason'] not in ['gradient_explosion', 'overfitting']:
                        # Crea φᵢ adaptat
                        phi = copy.deepcopy(meta_agent).to(self.device)
                        phi.load_state_dict({k: v.to(self.device)
                                           for k, v in result['state_dict'].items()})
                        phi.train()

                        adapted_policies.append({
                            'phi': phi,
                            'query_df': query_df,
                        })

                        if verbose:
                            print(f"  Task {total_tasks_processed}: {result['reason']} "
                                  f"({result['steps']:,} steps, sharpe={result['final_sharpe']:+.2f})")
                    else:
                        if verbose:
                            print(f"  Task {total_tasks_processed}: SKIP ({result['reason']})")
                else:
                    if verbose:
                        print(f"  Task {total_tasks_processed}: FAIL ({result['reason']})")

            # ── OUTER UPDATE (FOMAML) ────────────────────────────────────
            if len(adapted_policies) >= self.outer_trigger.min_tasks:
                self._fomaml_outer_update(
                    meta_agent, meta_optimizer, adapted_policies, env_factory
                )
                meta_iter += 1

                if verbose:
                    print(f'\nMeta-iter {meta_iter}: Outer update (FOMAML) amb {len(adapted_policies)} tasques')

                # ── META EVALUATION ──────────────────────────────────────
                if meta_iter % self.meta_criteria.eval_every == 0:
                    val_perf = self._evaluate_meta(meta_agent, val_df, env_factory)

                    meta_eval = {
                        'meta_iter': meta_iter,
                        'total_tasks': total_tasks_processed,
                        'n_adapted': len(adapted_policies),
                        'val_sharpe': val_perf['sharpe'],
                        'val_return': val_perf['mean_return'],
                    }
                    meta_history.append(meta_eval)

                    current_val_sharpe = val_perf['sharpe']

                    # TensorBoard logging
                    if self.writer:
                        self._log_to_tensorboard(
                            meta_iter, current_val_sharpe, val_perf['mean_return'],
                            len(adapted_policies), best_val_sharpe, inner_diagnostics
                        )

                    # Early stopping logic
                    if current_val_sharpe > best_val_sharpe + self.meta_criteria.plateau_threshold:
                        best_val_sharpe = current_val_sharpe
                        best_state_dict = {k: v.cpu().clone() for k, v in meta_agent.state_dict().items()}
                        evals_without_improvement = 0
                        status = '✓ NEW BEST'
                    else:
                        evals_without_improvement += 1
                        status = f'plateau {evals_without_improvement}/{self.meta_criteria.plateau_patience}'

                    if verbose:
                        print(f'  Val: sharpe={current_val_sharpe:+.4f} ret={val_perf["mean_return"]:+.2f}% '
                              f'best={best_val_sharpe:+.4f} [{status}]\n')

                    # ── Checkpoint periòdic ──────────────────────────────
                    if (ckpt_path is not None and
                            checkpoint_every > 0 and
                            meta_iter % checkpoint_every == 0):
                        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            'policy_state_dict':      meta_agent.state_dict(),
                            'optimizer_state_dict':   meta_optimizer.state_dict(),
                            'meta_history':           meta_history,
                            'inner_diagnostics':      inner_diagnostics,
                            'meta_iter':              meta_iter,
                            'total_tasks_processed':  total_tasks_processed,
                            'best_val_sharpe':        best_val_sharpe,
                            'best_state_dict':        best_state_dict,
                            'evals_without_improvement': evals_without_improvement,
                        }, ckpt_path)
                        if verbose:
                            print(f'  [ckpt] Desat checkpoint @ iter {meta_iter} → {ckpt_path.name}')

                    # Check stopping
                    if self.meta_criteria.target_val_sharpe is not None:
                        if current_val_sharpe >= self.meta_criteria.target_val_sharpe:
                            print(f'✓ Target val sharpe assolit: {current_val_sharpe:.4f}')
                            break

                    if evals_without_improvement >= self.meta_criteria.plateau_patience:
                        print(f'✗ Meta early stopping: {self.meta_criteria.plateau_patience} evals sense millora')
                        break
            else:
                if verbose:
                    print(f'  SKIP outer update: només {len(adapted_policies)} tasques convergides')

        # Restore best model
        if best_state_dict is not None:
            meta_agent.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})
            print(f'\nRestored best meta-agent: val_sharpe={best_val_sharpe:+.4f}')

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        return {
            'meta_policy': meta_agent,
            'meta_history': meta_history,
            'inner_diagnostics': inner_diagnostics,
            'total_tasks_processed': total_tasks_processed,
            'total_meta_iters': meta_iter,
            'best_val_sharpe': best_val_sharpe,
        }

    def _fomaml_outer_update(
        self,
        meta_agent,
        meta_optimizer,
        adapted_policies: list,
        env_factory: Callable,
    ):
        """
        FOMAML outer update: θ ← θ - β∇θ Σ L(φᵢ; D^query)

        First-order: no backprop through inner loop (tracta φᵢ com constants).
        """
        meta_optimizer.zero_grad()

        total_loss = 0.0
        n_valid = 0

        for item in adapted_policies:
            phi = item['phi']
            query_df = item['query_df']

            # Avalua φᵢ sobre query set
            query_loss = self._compute_query_loss(phi, query_df, env_factory)

            if query_loss is not None:
                total_loss += query_loss
                n_valid += 1

        if n_valid > 0:
            # Mitjana de losses
            avg_loss = total_loss / n_valid

            # Backward sobre meta_agent (no sobre φᵢ, first-order)
            avg_loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(meta_agent.parameters(), self.ppo_config['max_grad_norm'])

            # Update θ
            meta_optimizer.step()

    def _compute_query_loss(self, phi, query_df, env_factory):
        """
        Calcula loss de φ sobre query set.

        Per PPO, usem policy loss + value loss com a proxy del meta-objectiu.
        """
        try:
            env = env_factory(query_df, max_episode_steps=max(1, len(query_df) - 1))

            # Collect rollout amb φ
            n_steps = min(self.ppo_config['n_steps'], len(query_df) - 1)
            rollout = self.inner_loop._collect_rollout(phi, env, n_steps)

            obs, acts = rollout['obs'], rollout['acts']
            advs, rets = rollout['advs'], rollout['rets']

            # Normalitza advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # Compute loss
            vals, lps, entropy = phi.evaluate_actions(obs, acts)

            # Policy loss (sense clipping per simplificar gradient)
            policy_loss = -(lps * advs).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(vals, rets)

            # Total loss
            loss = policy_loss + self.ppo_config['vf_coef'] * value_loss

            return loss

        except Exception as e:
            print(f'Warning: query loss computation failed: {e}')
            return None

    def _evaluate_meta(self, meta_agent, val_df, env_factory):
        """Avalua meta-agent sobre val set."""
        env = env_factory(val_df)
        meta_agent.eval()
        returns = []

        with torch.no_grad():
            for _ in range(self.meta_criteria.eval_episodes):
                obs, info = env.reset()
                eq0 = info['equity']
                done = False
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logits, _ = meta_agent(obs_t)
                    action = logits.argmax(dim=-1).item()
                    obs, _, term, trunc, info = env.step(action)
                    done = term or trunc
                returns.append((info['equity'] / eq0 - 1) * 100)

        meta_agent.train()
        return {
            'mean_return': float(np.mean(returns)),
            'sharpe': float(np.mean(returns) / (np.std(returns) + 1e-8)),
        }

    def _log_to_tensorboard(
        self,
        meta_iter: int,
        current_val_sharpe: float,
        current_val_return: float,
        n_adapted: int,
        best_val_sharpe: float,
        inner_diagnostics: list,
    ):
        """Escriu mètriques a TensorBoard."""
        # Meta-level scalars
        self.writer.add_scalar('meta/val_sharpe', current_val_sharpe, meta_iter)
        self.writer.add_scalar('meta/val_return', current_val_return, meta_iter)
        self.writer.add_scalar('meta/n_adapted_tasks', n_adapted, meta_iter)
        self.writer.add_scalar('meta/best_val_sharpe', best_val_sharpe, meta_iter)

        # Inner-level agregats (només de l'última batch)
        recent_inner = [d for d in inner_diagnostics if d['meta_iter'] == meta_iter]
        if recent_inner:
            converged = [d for d in recent_inner if d['converged']]

            # Convergence rate
            conv_rate = len(converged) / len(recent_inner) if recent_inner else 0.0
            self.writer.add_scalar('inner/convergence_rate', conv_rate, meta_iter)

            if converged:
                # Steps statistics
                steps = [d['steps'] for d in converged]
                self.writer.add_scalar('inner/mean_steps', float(np.mean(steps)), meta_iter)
                self.writer.add_scalar('inner/median_steps', float(np.median(steps)), meta_iter)
                self.writer.add_scalar('inner/min_steps', float(np.min(steps)), meta_iter)
                self.writer.add_scalar('inner/max_steps', float(np.max(steps)), meta_iter)

                # Sharpe statistics
                sharpes = [d['final_sharpe'] for d in converged]
                self.writer.add_scalar('inner/mean_sharpe', float(np.mean(sharpes)), meta_iter)
                self.writer.add_histogram('inner/sharpe_distribution', np.array(sharpes), meta_iter)

            # Reasons distribution
            reasons = {}
            for d in recent_inner:
                r = d['reason']
                reasons[r] = reasons.get(r, 0) + 1

            for reason, count in reasons.items():
                self.writer.add_scalar(f'inner/reason_{reason}', count, meta_iter)


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Adaptive FOMAML system loaded.')
    print('\nUsage:')
    print("""
from adaptive_fomaml import (
    AdaptiveFOMAML,
    InnerConvergenceCriteria,
    OuterUpdateTrigger,
    MetaConvergenceCriteria,
)

# Defineix criteris (iguals que Reptile)
inner_criteria = InnerConvergenceCriteria(
    min_steps=5_000,
    max_steps=30_000,
    target_sharpe=0.8,
    plateau_patience=3,
)

outer_trigger = OuterUpdateTrigger(
    batch_size=5,
    max_time_seconds=300,
    min_tasks=3,
)

meta_criteria = MetaConvergenceCriteria(
    max_meta_iters=500,
    eval_every=5,
    target_val_sharpe=2.0,
    plateau_patience=20,
)

# Crea sistema FOMAML
fomaml = AdaptiveFOMAML(
    inner_criteria=inner_criteria,
    outer_trigger=outer_trigger,
    meta_criteria=meta_criteria,
    device=device,
    ppo_config=ppo_config,
    meta_lr=1e-4,           # Learning rate outer update
    support_ratio=0.7,      # 70% support, 30% query
)

# Task sampler
def sample_task():
    start = rng.integers(0, len(df_meta) - TASK_STEPS)
    return df_meta[start: start + TASK_STEPS]

# Meta-train
result = fomaml.meta_train(
    base_policy=base_policy,
    task_sampler=sample_task,
    env_factory=make_env,
    val_df=df_val,
    verbose=True,
)

meta_policy = result['meta_policy']
    """)
