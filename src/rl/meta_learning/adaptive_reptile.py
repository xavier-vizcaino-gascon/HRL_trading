"""
Sistema de meta-entrenament Reptile ADAPTATIU.

Característiques:
- Inner loops dinàmics: cada tasca s'entrena fins convergir
- Outer updates automàtics: quan s'acumulen N tasques o passa X temps
- Early stopping global: para quan el meta-agent convergeix
- Totalment automatitzat: només defineix criteris de convergència
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


@dataclass
class InnerConvergenceCriteria:
    """Criteris per determinar si un inner loop ha convergit."""
    min_steps: int = 5_000                    # Mínim d'steps abans d'avaluar convergència
    max_steps: int = 30_000                   # Màxim d'steps (timeout)
    eval_every: int = 2_000                   # Avalua convergència cada X steps
    eval_episodes: int = 5                    # Episodis per avaluar sharpe

    # Convergència per performance
    target_sharpe: float = 0.8                # Si sharpe >= target → convergit
    min_sharpe_improvement: float = 0.3       # Millora mínima des d'inici
    plateau_threshold: float = 0.05           # Millora <5% → plateau
    plateau_patience: int = 3                 # N evals sense millora → convergit

    # Convergència per gradients
    min_grad_norm: float = 0.01               # Gradient massa petit → convergit
    max_grad_norm: float = 10.0               # Gradient massa gran → divergència

    # Convergència per paràmetres
    max_param_distance: float = 15.0          # ||φ - θ|| massa gran → overfitting


@dataclass
class OuterUpdateTrigger:
    """Criteris per fer outer update de Reptile."""
    batch_size: int = 5                       # N tasques convergides → update
    max_time_seconds: Optional[float] = 300.0 # O cada 5 min → update
    min_tasks: int = 3                        # Mínim de tasques per update vàlid


@dataclass
class MetaConvergenceCriteria:
    """Criteris per parar el meta-entrenament."""
    max_meta_iters: int = 500                 # Màxim d'outer updates
    eval_every: int = 5                       # Avalua meta-agent cada N updates
    eval_episodes: int = 25                   # Episodis per avaluar val sharpe

    target_val_sharpe: float = 2.0            # Si val sharpe >= target → para
    plateau_threshold: float = 0.1            # Millora <10% → plateau
    plateau_patience: int = 20                # N evals sense millora → para
    min_val_sharpe: float = 0.5               # Mínim acceptable


class AdaptiveInnerLoop:
    """Gestiona un inner loop amb convergència dinàmica."""

    def __init__(
        self,
        criteria: InnerConvergenceCriteria,
        device: torch.device,
        ppo_config: dict,
    ):
        self.criteria = criteria
        self.device = device
        self.ppo_config = ppo_config

    def train_until_convergence(
        self,
        base_policy,
        task_df,
        env_factory: Callable,
    ) -> dict:
        """
        Entrena inner loop fins convergència.

        Returns:
            {
                'converged': bool,
                'reason': str,
                'steps': int,
                'final_sharpe': float,
                'state_dict': dict,
                'diagnostics': dict,
            }
        """
        phi = copy.deepcopy(base_policy).to(self.device)
        phi.train()

        # Estat inicial
        theta_sd = {k: v.clone().cpu() for k, v in base_policy.state_dict().items()}
        optimizer = torch.optim.Adam(
            phi.parameters(),
            lr=self.ppo_config['learning_rate'],
            eps=1e-5
        )

        ep_steps = max(1, len(task_df) - 1)
        env = env_factory(task_df, max_episode_steps=ep_steps)

        # Tracking
        timesteps = 0
        eval_history = []
        last_sharpe = -np.inf
        plateau_count = 0
        initial_sharpe = None

        while timesteps < self.criteria.max_steps:
            # Collect rollout
            steps_this_rollout = min(
                self.ppo_config['n_steps'],
                self.criteria.max_steps - timesteps
            )

            rollout = self._collect_rollout(phi, env, steps_this_rollout)
            grad_stats = self._ppo_update(phi, optimizer, rollout)

            timesteps += steps_this_rollout

            # Avalua convergència periòdicament
            if (timesteps >= self.criteria.min_steps and
                timesteps % self.criteria.eval_every < steps_this_rollout):

                perf = self._evaluate(phi, env)
                param_dist = self._param_distance(phi, theta_sd)

                eval_info = {
                    'step': timesteps,
                    'sharpe': perf['sharpe'],
                    'mean_return': perf['mean_return'],
                    'grad_norm': grad_stats['mean_grad_norm'],
                    'param_distance': param_dist,
                }
                eval_history.append(eval_info)

                if initial_sharpe is None:
                    initial_sharpe = perf['sharpe']

                current_sharpe = perf['sharpe']
                sharpe_improvement = current_sharpe - initial_sharpe

                # Check convergència
                converged, reason = self._check_convergence(
                    current_sharpe=current_sharpe,
                    sharpe_improvement=sharpe_improvement,
                    last_sharpe=last_sharpe,
                    grad_norm=grad_stats['mean_grad_norm'],
                    param_dist=param_dist,
                    plateau_count=plateau_count,
                )

                if converged:
                    return {
                        'converged': True,
                        'reason': reason,
                        'steps': timesteps,
                        'final_sharpe': current_sharpe,
                        'sharpe_improvement': sharpe_improvement,
                        'state_dict': {k: v.cpu() for k, v in phi.state_dict().items()},
                        'diagnostics': {
                            'eval_history': eval_history,
                            'n_evals': len(eval_history),
                        },
                    }

                # Update plateau counter
                if abs(current_sharpe - last_sharpe) < self.criteria.plateau_threshold:
                    plateau_count += 1
                else:
                    plateau_count = 0

                last_sharpe = current_sharpe

        # Timeout
        return {
            'converged': False,
            'reason': 'max_steps_timeout',
            'steps': timesteps,
            'final_sharpe': last_sharpe,
            'sharpe_improvement': last_sharpe - (initial_sharpe or -np.inf),
            'state_dict': None,
            'diagnostics': {'eval_history': eval_history},
        }

    def _check_convergence(
        self,
        current_sharpe: float,
        sharpe_improvement: float,
        last_sharpe: float,
        grad_norm: float,
        param_dist: float,
        plateau_count: int,
    ) -> tuple[bool, str]:
        """Determina si l'inner loop ha convergit."""

        # 1. Target assolit
        if current_sharpe >= self.criteria.target_sharpe:
            return True, 'target_sharpe_reached'

        # 2. Millora suficient + plateau
        if (sharpe_improvement >= self.criteria.min_sharpe_improvement and
            plateau_count >= self.criteria.plateau_patience):
            return True, 'performance_plateau'

        # 3. Gradients massa petits (convergit localment)
        if grad_norm < self.criteria.min_grad_norm:
            return True, 'gradient_vanished'

        # 4. Divergència (gradients explosius)
        if grad_norm > self.criteria.max_grad_norm:
            return True, 'gradient_explosion'

        # 5. Overfitting (param distance massa gran)
        if param_dist > self.criteria.max_param_distance:
            return True, 'overfitting'

        return False, 'training'

    def _collect_rollout(self, policy, env, n_steps):
        """Collect PPO rollout."""
        obs_dim = env.observation_space.shape[0]
        obs_buf = np.zeros((n_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros(n_steps, dtype=np.int64)
        rew_buf = np.zeros(n_steps, dtype=np.float32)
        done_buf = np.zeros(n_steps, dtype=np.float32)
        val_buf = np.zeros(n_steps, dtype=np.float32)
        lp_buf = np.zeros(n_steps, dtype=np.float32)

        policy.eval()
        obs, _ = env.reset()

        with torch.no_grad():
            for t in range(n_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, val = policy(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                lp = dist.log_prob(action)

                obs_buf[t] = obs
                act_buf[t] = action.item()
                val_buf[t] = val.item()
                lp_buf[t] = lp.item()

                obs, rew, term, trunc, _ = env.step(action.item())
                rew_buf[t] = rew
                done_buf[t] = float(term or trunc)

                if term or trunc:
                    obs, _ = env.reset()

            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_val = policy(obs_t)
            last_val = last_val.item()

        # GAE
        adv_buf = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0
        gamma = self.ppo_config['gamma']
        gae_lambda = self.ppo_config['gae_lambda']

        for t in reversed(range(n_steps)):
            next_val = last_val if t == n_steps - 1 else val_buf[t + 1]
            next_non_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + gamma * next_val * next_non_done - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_non_done * last_gae
            adv_buf[t] = last_gae

        ret_buf = adv_buf + val_buf
        policy.train()

        return {
            'obs': torch.tensor(obs_buf, device=self.device),
            'acts': torch.tensor(act_buf, device=self.device),
            'lps': torch.tensor(lp_buf, device=self.device),
            'advs': torch.tensor(adv_buf, device=self.device),
            'rets': torch.tensor(ret_buf, device=self.device),
        }

    def _ppo_update(self, policy, optimizer, rollout):
        """PPO update amb gradient tracking."""
        obs, acts, old_lps = rollout['obs'], rollout['acts'], rollout['lps']
        advs, rets = rollout['advs'], rollout['rets']

        n = obs.shape[0]
        n_epochs = self.ppo_config['n_epochs']
        batch_size = self.ppo_config['batch_size']
        clip_range = self.ppo_config['clip_range']
        ent_coef = self.ppo_config['ent_coef']
        vf_coef = self.ppo_config['vf_coef']
        max_grad_norm = self.ppo_config['max_grad_norm']

        grad_norms = []

        for _ in range(n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, batch_size):
                mb = idx[start: start + batch_size]
                mb_obs = obs[mb]
                mb_acts = acts[mb]
                mb_old_lps = old_lps[mb]
                mb_advs = advs[mb]
                mb_rets = rets[mb]

                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                vals, lps, entropy = policy.evaluate_actions(mb_obs, mb_acts)

                ratio = torch.exp(lps - mb_old_lps)
                surr1 = ratio * mb_advs
                surr2 = ratio.clamp(1 - clip_range, 1 + clip_range) * mb_advs
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = nn.functional.mse_loss(vals, mb_rets)
                ent_loss = -entropy.mean()

                loss = pol_loss + vf_coef * val_loss + ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()

                # Track gradient norm
                total_norm = 0.0
                for p in policy.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms.append(total_norm ** 0.5)

                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        return {'mean_grad_norm': float(np.mean(grad_norms))}

    def _evaluate(self, policy, env):
        """Avaluació ràpida."""
        policy.eval()
        returns = []

        with torch.no_grad():
            for _ in range(self.criteria.eval_episodes):
                obs, info = env.reset()
                eq0 = info['equity']
                done = False
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logits, _ = policy(obs_t)
                    action = logits.argmax(dim=-1).item()
                    obs, _, term, trunc, info = env.step(action)
                    done = term or trunc
                returns.append((info['equity'] / eq0 - 1) * 100)

        policy.train()
        return {
            'mean_return': float(np.mean(returns)),
            'sharpe': float(np.mean(returns) / (np.std(returns) + 1e-8)),
        }

    def _param_distance(self, phi, theta_sd):
        """Calcula ||φ - θ||."""
        phi_sd = {k: v.cpu() for k, v in phi.state_dict().items()}
        total_dist = 0.0
        for key in phi_sd:
            if phi_sd[key].dtype in (torch.float32, torch.float64):
                diff = (phi_sd[key] - theta_sd[key]).float()
                total_dist += (diff ** 2).sum().item()
        return float(np.sqrt(total_dist))


class AdaptiveReptile:
    """Sistema de meta-entrenament Reptile adaptatiu."""

    def __init__(
        self,
        inner_criteria: InnerConvergenceCriteria,
        outer_trigger: OuterUpdateTrigger,
        meta_criteria: MetaConvergenceCriteria,
        device: torch.device,
        ppo_config: dict,
        reptile_eps: float = 0.1,
        tensorboard_log: Optional[str] = None,
    ):
        self.inner_loop = AdaptiveInnerLoop(inner_criteria, device, ppo_config)
        self.outer_trigger = outer_trigger
        self.meta_criteria = meta_criteria
        self.device = device
        self.reptile_eps = reptile_eps
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
        Meta-entrenament adaptatiu complet.

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
            meta_history           = _d.get('meta_history', [])
            inner_diagnostics      = _d.get('inner_diagnostics', [])
            meta_iter              = _d.get('meta_iter', 0)
            total_tasks_processed  = _d.get('total_tasks_processed', 0)
            best_val_sharpe        = _d.get('best_val_sharpe', -np.inf)
            best_state_dict        = _d.get('best_state_dict', None)
            evals_without_improvement = _d.get('evals_without_improvement', 0)
            if verbose:
                print(f'[Reptile] Reprenent des de checkpoint: '
                      f'meta_iter={meta_iter}, best_sharpe={best_val_sharpe:+.4f}')

        while meta_iter < self.meta_criteria.max_meta_iters:
            # ── INNER LOOP BATCH ──────────────────────────────────────────
            adapted_states = []
            batch_start_time = time.time()

            while len(adapted_states) < self.outer_trigger.batch_size:
                # Check timeout
                elapsed = time.time() - batch_start_time
                if (self.outer_trigger.max_time_seconds is not None and
                    elapsed > self.outer_trigger.max_time_seconds and
                    len(adapted_states) >= self.outer_trigger.min_tasks):
                    if verbose:
                        print(f'  Batch timeout: {len(adapted_states)} tasks in {elapsed:.1f}s')
                    break

                # Sample task
                task_df = task_sampler()
                total_tasks_processed += 1

                # Train inner loop fins convergència
                result = self.inner_loop.train_until_convergence(
                    meta_agent, task_df, env_factory
                )

                # Guarda diagnòstics
                inner_diagnostics.append({
                    'meta_iter': meta_iter,
                    'task_idx': total_tasks_processed,
                    **result,
                })

                # Només usa si convergit correctament
                if result['converged'] and result['state_dict'] is not None:
                    # Filtra divergències i overfitting
                    if result['reason'] not in ['gradient_explosion', 'overfitting']:
                        adapted_states.append(result['state_dict'])
                        if verbose:
                            print(f"  Task {total_tasks_processed}: {result['reason']} "
                                  f"({result['steps']:,} steps, sharpe={result['final_sharpe']:+.2f})")
                    else:
                        if verbose:
                            print(f"  Task {total_tasks_processed}: SKIP ({result['reason']})")
                else:
                    if verbose:
                        print(f"  Task {total_tasks_processed}: FAIL ({result['reason']})")

            # ── OUTER UPDATE ──────────────────────────────────────────────
            if len(adapted_states) >= self.outer_trigger.min_tasks:
                self._reptile_outer_update(meta_agent, adapted_states)
                meta_iter += 1

                if verbose:
                    print(f'\nMeta-iter {meta_iter}: Outer update amb {len(adapted_states)} tasques')

                # ── META EVALUATION ──────────────────────────────────────
                if meta_iter % self.meta_criteria.eval_every == 0:
                    val_perf = self._evaluate_meta(meta_agent, val_df, env_factory)

                    meta_eval = {
                        'meta_iter': meta_iter,
                        'total_tasks': total_tasks_processed,
                        'n_adapted': len(adapted_states),
                        'val_sharpe': val_perf['sharpe'],
                        'val_return': val_perf['mean_return'],
                    }
                    meta_history.append(meta_eval)

                    current_val_sharpe = val_perf['sharpe']

                    # TensorBoard logging
                    if self.writer:
                        self._log_to_tensorboard(
                            meta_iter, current_val_sharpe, val_perf['mean_return'],
                            len(adapted_states), best_val_sharpe, inner_diagnostics
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
                    print(f'  SKIP outer update: només {len(adapted_states)} tasques convergides')

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

    def _reptile_outer_update(self, meta_agent, adapted_states):
        """Reptile outer update: θ ← θ + ε · mean(φᵢ − θ)."""
        base_sd = {k: v.cpu() for k, v in meta_agent.state_dict().items()}
        new_sd = {}

        for key in base_sd:
            t = base_sd[key]
            if t.dtype in (torch.float32, torch.float64):
                mean_phi = torch.stack([s[key].float() for s in adapted_states]).mean(0)
                new_sd[key] = t + self.reptile_eps * (mean_phi - t)
            else:
                new_sd[key] = t

        meta_agent.load_state_dict({k: v.to(self.device) for k, v in new_sd.items()})

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
    print('Adaptive Reptile system loaded.')
    print('\nUsage:')
    print("""
from adaptive_reptile import (
    AdaptiveReptile,
    InnerConvergenceCriteria,
    OuterUpdateTrigger,
    MetaConvergenceCriteria,
)

# Defineix criteris
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

# Crea sistema
reptile = AdaptiveReptile(
    inner_criteria=inner_criteria,
    outer_trigger=outer_trigger,
    meta_criteria=meta_criteria,
    device=device,
    ppo_config=ppo_config,
    reptile_eps=0.1,
)

# Task sampler
def sample_task():
    start = rng.integers(0, len(df_meta) - TASK_STEPS)
    return df_meta[start: start + TASK_STEPS]

# Meta-train
result = reptile.meta_train(
    base_policy=base_policy,
    task_sampler=sample_task,
    env_factory=make_env,
    val_df=df_val,
    verbose=True,
)

meta_policy = result['meta_policy']
    """)
