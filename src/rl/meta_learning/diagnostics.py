"""
Diagnòstics per mesurar convergència dels inner loops en Reptile.

Implementa 5 mètriques clau:
1. Loss trajectories (policy, value, entropy)
2. Performance trajectories (sharpe, return)
3. Gradient norms (mitjana i màxim)
4. Parameter distance ||φ - θ||
5. Plateau detection (early stopping intern)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def measure_inner_convergence(
    base_policy,
    task_df,
    env_factory,
    device,
    inner_steps=20_000,
    n_steps=4096,
    n_epochs=7,
    batch_size=64,
    checkpoint_every=2_000,  # Avalua cada 2k steps
    eval_episodes=5,
    lr=2.77e-4,
    gamma=0.9967,
    gae_lambda=0.9015,
    clip_range=0.2024,
    ent_coef=0.0099,
    vf_coef=0.7339,
    max_grad_norm=0.5,
) -> dict:
    """
    Entrena inner loop amb diagnòstics detallats.

    Returns:
        dict amb trajectòries de:
        - losses: {policy_loss, value_loss, entropy}
        - performance: {sharpe, mean_return}
        - gradients: {mean_norm, max_norm}
        - param_distance: ||φ_t - θ||
        - convergence_metrics: {has_converged, plateau_at_step}
    """
    phi = copy.deepcopy(base_policy).to(device)
    phi.train()

    # Guarda estat inicial per calcular distàncies
    theta_sd = {k: v.clone().cpu() for k, v in base_policy.state_dict().items()}

    optimizer = torch.optim.Adam(phi.parameters(), lr=lr, eps=1e-5)
    ep_steps = max(1, len(task_df) - 1)
    env = env_factory(task_df, max_episode_steps=ep_steps)

    # Trajectòries
    checkpoints = []
    timesteps = 0
    last_sharpe = -np.inf
    plateau_count = 0
    convergence_threshold = 0.05  # Si millora < 5% durant 3 checkpoints → convergit

    while timesteps < inner_steps:
        steps_this_rollout = min(n_steps, inner_steps - timesteps)

        # Collect rollout
        rollout = _collect_ppo_rollout_with_metrics(
            phi, env, steps_this_rollout, device, gamma, gae_lambda
        )

        # Update amb tracking de gradients
        update_metrics = _ppo_update_with_gradient_tracking(
            phi, optimizer, rollout, n_epochs, batch_size,
            clip_range, ent_coef, vf_coef, max_grad_norm
        )

        timesteps += steps_this_rollout

        # Checkpoint periòdic
        if timesteps % checkpoint_every < steps_this_rollout or timesteps >= inner_steps:
            # 1. Loss trajectories (ja tenim d'update_metrics)
            losses = {
                'policy_loss': update_metrics['policy_loss'],
                'value_loss': update_metrics['value_loss'],
                'entropy': update_metrics['entropy'],
            }

            # 2. Performance trajectories
            perf = _quick_eval(phi, env, device, n_episodes=eval_episodes)

            # 3. Gradient norms
            grad_norms = {
                'mean_norm': update_metrics['mean_grad_norm'],
                'max_norm': update_metrics['max_grad_norm'],
            }

            # 4. Parameter distance ||φ - θ||
            phi_sd = {k: v.clone().cpu() for k, v in phi.state_dict().items()}
            param_dist = _compute_param_distance(phi_sd, theta_sd)

            checkpoint = {
                'step': timesteps,
                'losses': losses,
                'performance': perf,
                'gradients': grad_norms,
                'param_distance': param_dist,
            }
            checkpoints.append(checkpoint)

            # 5. Plateau detection
            current_sharpe = perf['sharpe']
            if abs(current_sharpe - last_sharpe) < convergence_threshold:
                plateau_count += 1
            else:
                plateau_count = 0

            last_sharpe = current_sharpe

            # Early stop si ha convergit (3 checkpoints consecutius sense millora)
            if plateau_count >= 3:
                convergence_info = {
                    'has_converged': True,
                    'plateau_at_step': timesteps,
                    'reason': 'performance_plateau'
                }
                break
    else:
        convergence_info = {
            'has_converged': False,
            'plateau_at_step': None,
            'reason': 'max_steps_reached'
        }

    # Anàlisi final
    return {
        'checkpoints': checkpoints,
        'convergence': convergence_info,
        'final_state_dict': {k: v.cpu() for k, v in phi.state_dict().items()},
        'summary': _summarize_convergence(checkpoints, convergence_info),
    }


def _collect_ppo_rollout_with_metrics(policy, env, n_steps, device, gamma, gae_lambda):
    """Versió de collect_rollout que també retorna mètriques de l'episodi."""
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
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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

        # Bootstrap
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, last_val = policy(obs_t)
        last_val = last_val.item()

    # GAE
    adv_buf = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        next_val = last_val if t == n_steps - 1 else val_buf[t + 1]
        next_non_done = 1.0 - done_buf[t]
        delta = rew_buf[t] + gamma * next_val * next_non_done - val_buf[t]
        last_gae = delta + gamma * gae_lambda * next_non_done * last_gae
        adv_buf[t] = last_gae
    ret_buf = adv_buf + val_buf

    policy.train()
    return {
        'obs': torch.tensor(obs_buf, device=device),
        'acts': torch.tensor(act_buf, device=device),
        'lps': torch.tensor(lp_buf, device=device),
        'advs': torch.tensor(adv_buf, device=device),
        'rets': torch.tensor(ret_buf, device=device),
    }


def _ppo_update_with_gradient_tracking(
    policy, optimizer, rollout, n_epochs, batch_size,
    clip_range, ent_coef, vf_coef, max_grad_norm
):
    """PPO update que també trackeja gradient norms."""
    obs, acts, old_lps = rollout['obs'], rollout['acts'], rollout['lps']
    advs, rets = rollout['advs'], rollout['rets']

    n = obs.shape[0]
    stats = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'grad_norms': [],
    }

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

            # Trackeja gradient norm ABANS del clipping
            total_norm = 0.0
            for p in policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            stats['grad_norms'].append(total_norm)

            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            stats['policy_loss'].append(pol_loss.item())
            stats['value_loss'].append(val_loss.item())
            stats['entropy'].append(-ent_loss.item())

    return {
        'policy_loss': float(np.mean(stats['policy_loss'])),
        'value_loss': float(np.mean(stats['value_loss'])),
        'entropy': float(np.mean(stats['entropy'])),
        'mean_grad_norm': float(np.mean(stats['grad_norms'])),
        'max_grad_norm': float(np.max(stats['grad_norms'])),
    }


def _quick_eval(policy, env, device, n_episodes=5):
    """Avaluació ràpida per checkpoints interns."""
    policy.eval()
    returns = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env.reset()
            eq0 = info['equity']
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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


def _compute_param_distance(phi_state, theta_state):
    """Calcula ||φ - θ|| (L2 distance entre els paràmetres)."""
    total_dist = 0.0
    n_params = 0

    for key in phi_state:
        if phi_state[key].dtype in (torch.float32, torch.float64):
            diff = (phi_state[key] - theta_state[key]).float()
            total_dist += (diff ** 2).sum().item()
            n_params += diff.numel()

    return {
        'l2_distance': float(np.sqrt(total_dist)),
        'mean_param_change': float(np.sqrt(total_dist / max(n_params, 1))),
    }


def _summarize_convergence(checkpoints, convergence_info):
    """Genera resum de diagnòstic de convergència."""
    if not checkpoints:
        return {'status': 'no_checkpoints'}

    # Trajectòries
    steps = [c['step'] for c in checkpoints]
    sharpes = [c['performance']['sharpe'] for c in checkpoints]
    returns = [c['performance']['mean_return'] for c in checkpoints]
    pol_losses = [c['losses']['policy_loss'] for c in checkpoints]
    grad_norms = [c['gradients']['mean_norm'] for c in checkpoints]
    param_dists = [c['param_distance']['l2_distance'] for c in checkpoints]

    # Anàlisi de tendències
    sharpe_trend = 'improving' if sharpes[-1] > sharpes[0] else 'degrading'
    loss_trend = 'decreasing' if pol_losses[-1] < pol_losses[0] else 'increasing'

    # Gradient stability
    grad_std = np.std(grad_norms)
    grad_stability = 'stable' if grad_std < 0.5 else 'unstable'

    # Parameter movement
    final_param_dist = param_dists[-1]
    param_movement = (
        'large' if final_param_dist > 10.0 else
        'medium' if final_param_dist > 1.0 else
        'small'
    )

    return {
        'status': 'converged' if convergence_info['has_converged'] else 'incomplete',
        'total_steps': steps[-1],
        'n_checkpoints': len(checkpoints),
        'final_sharpe': sharpes[-1],
        'sharpe_improvement': sharpes[-1] - sharpes[0],
        'sharpe_trend': sharpe_trend,
        'final_return': returns[-1],
        'policy_loss_trend': loss_trend,
        'gradient_stability': grad_stability,
        'mean_gradient_norm': float(np.mean(grad_norms)),
        'parameter_movement': param_movement,
        'final_param_distance': final_param_dist,
        'convergence_reason': convergence_info['reason'],
    }


def plot_inner_convergence(diagnostics, save_path=None):
    """Visualitza les mètriques de convergència dels inner loops."""
    import matplotlib.pyplot as plt

    checkpoints = diagnostics['checkpoints']
    if not checkpoints:
        print('No checkpoints to plot')
        return

    steps = [c['step'] for c in checkpoints]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Performance (Sharpe + Return)
    sharpes = [c['performance']['sharpe'] for c in checkpoints]
    returns = [c['performance']['mean_return'] for c in checkpoints]
    axes[0, 0].plot(steps, sharpes, 'o-', color='#2ca02c', lw=2, label='Sharpe')
    axes[0, 0].axhline(0, color='gray', ls='--', alpha=0.5)
    axes[0, 0].set_title('Performance: Sharpe Ratio')
    axes[0, 0].set_xlabel('Inner Steps')
    axes[0, 0].set_ylabel('Sharpe')
    axes[0, 0].grid(alpha=0.3)

    # 2. Returns
    axes[0, 1].plot(steps, returns, 'o-', color='#1f77b4', lw=2)
    axes[0, 1].axhline(0, color='gray', ls='--', alpha=0.5)
    axes[0, 1].set_title('Performance: Mean Return')
    axes[0, 1].set_xlabel('Inner Steps')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].grid(alpha=0.3)

    # 3. Policy Loss
    pol_losses = [c['losses']['policy_loss'] for c in checkpoints]
    axes[0, 2].plot(steps, pol_losses, 'o-', color='#d62728', lw=2)
    axes[0, 2].set_title('Policy Loss')
    axes[0, 2].set_xlabel('Inner Steps')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(alpha=0.3)

    # 4. Value Loss
    val_losses = [c['losses']['value_loss'] for c in checkpoints]
    axes[1, 0].plot(steps, val_losses, 'o-', color='#ff7f0e', lw=2)
    axes[1, 0].set_title('Value Loss')
    axes[1, 0].set_xlabel('Inner Steps')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(alpha=0.3)

    # 5. Gradient Norms
    grad_norms = [c['gradients']['mean_norm'] for c in checkpoints]
    axes[1, 1].plot(steps, grad_norms, 'o-', color='#9467bd', lw=2)
    axes[1, 1].set_title('Mean Gradient Norm')
    axes[1, 1].set_xlabel('Inner Steps')
    axes[1, 1].set_ylabel('Norm')
    axes[1, 1].grid(alpha=0.3)

    # 6. Parameter Distance
    param_dists = [c['param_distance']['l2_distance'] for c in checkpoints]
    axes[1, 2].plot(steps, param_dists, 'o-', color='#8c564b', lw=2)
    axes[1, 2].set_title('Parameter Distance ||φ - θ||')
    axes[1, 2].set_xlabel('Inner Steps')
    axes[1, 2].set_ylabel('L2 Distance')
    axes[1, 2].grid(alpha=0.3)

    plt.suptitle('Inner Loop Convergence Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'Convergence plot saved: {save_path}')

    plt.show()

    # Print summary
    summary = diagnostics['summary']
    print('\n' + '='*60)
    print('INNER LOOP CONVERGENCE SUMMARY')
    print('='*60)
    print(f"Status: {summary['status']}")
    print(f"Total steps: {summary['total_steps']:,}")
    print(f"Final Sharpe: {summary['final_sharpe']:+.4f}")
    print(f"Sharpe improvement: {summary['sharpe_improvement']:+.4f}")
    print(f"Sharpe trend: {summary['sharpe_trend']}")
    print(f"Policy loss trend: {summary['policy_loss_trend']}")
    print(f"Gradient stability: {summary['gradient_stability']}")
    print(f"Parameter movement: {summary['parameter_movement']}")
    print(f"Convergence reason: {summary['convergence_reason']}")
    print('='*60 + '\n')


# Exemple d'ús
if __name__ == '__main__':
    print('Reptile diagnostics module loaded.')
    print('\nUsage:')
    print('  from reptile_diagnostics import measure_inner_convergence, plot_inner_convergence')
    print('  diagnostics = measure_inner_convergence(base_policy, task_df, make_env, device)')
    print('  plot_inner_convergence(diagnostics, save_path="convergence.png")')
