"""
Proximal Policy Optimization (PPO) agent.

Implements PPO algorithm from scratch using PyTorch.
PPO is a stable and sample-efficient on-policy algorithm.

Reference:
    Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray

from src.rl.agents.base import AgentConfig, OnPolicyAgent
from src.rl.models.base import NetworkConfig
from src.rl.models.mlp import MLPNetwork
from src.rl.models.policy_heads import CategoricalPolicyHead, GaussianPolicyHead
from src.rl.models.value_heads import ValueHead
from src.rl.training.replay_buffer import RolloutBuffer


@dataclass
class PPOConfig(AgentConfig):
    """Configuration for PPO agent."""

    # Network architecture
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"

    # PPO hyperparameters
    clip_epsilon: float = 0.2  # Clipping parameter for policy loss
    n_epochs: int = 10  # Number of epochs per update
    batch_size: int = 64  # Mini-batch size
    gae_lambda: float = 0.95  # GAE lambda
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Rollout configuration
    n_steps: int = 2048  # Steps to collect per update

    # Action space
    action_type: str = "discrete"  # 'discrete' or 'continuous'
    action_dim: int = 3  # For discrete: number of actions
    action_low: float = -1.0  # For continuous: lower bound
    action_high: float = 1.0  # For continuous: upper bound


class PPOAgent(OnPolicyAgent):
    """
    Proximal Policy Optimization (PPO) agent.

    PPO uses clipped surrogate objective to prevent large policy updates:
    L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    """

    def __init__(self, observation_dim: int, config: PPOConfig):
        """
        Initialize PPO agent.

        Args:
            observation_dim: Dimension of observation space
            config: PPO configuration
        """
        super().__init__(config)
        self.config: PPOConfig = config
        self.observation_dim = observation_dim

        # Create actor (policy) network
        actor_config = NetworkConfig(
            input_dim=observation_dim,
            output_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            device=config.device,
        )
        self.actor_backbone = MLPNetwork(actor_config).to(self.device)

        # Create policy head based on action type
        if config.action_type == "discrete":
            self.policy_head = CategoricalPolicyHead(config.hidden_dims[-1], config.action_dim).to(
                self.device
            )
        elif config.action_type == "continuous":
            self.policy_head = GaussianPolicyHead(config.hidden_dims[-1], config.action_dim).to(
                self.device
            )
        else:
            raise ValueError(f"Unknown action_type: {config.action_type}")

        # Create critic (value) network
        critic_config = NetworkConfig(
            input_dim=observation_dim,
            output_dim=config.hidden_dims[-1],
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            device=config.device,
        )
        self.critic_backbone = MLPNetwork(critic_config).to(self.device)
        self.value_head = ValueHead(config.hidden_dims[-1]).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.actor_backbone.parameters()) + list(self.policy_head.parameters()),
            lr=config.learning_rate,
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic_backbone.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            capacity=config.n_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

    def select_action(
        self, observation: NDArray[np.float32], deterministic: bool = False
    ) -> tuple[NDArray | int, float, float]:
        """
        Select action from policy.

        Args:
            observation: Current observation
            deterministic: If True, select mean action (no sampling)

        Returns:
            Tuple of (action, log_prob, value)
        """
        self.actor_backbone.eval()
        self.critic_backbone.eval()

        with torch.no_grad():
            obs_tensor = self.to_tensor(observation).unsqueeze(0)  # [1, obs_dim]

            # Get policy distribution
            actor_features = self.actor_backbone(obs_tensor)

            if self.config.action_type == "discrete":
                if deterministic:
                    # Select most probable action
                    logits = self.policy_head(actor_features)
                    action = torch.argmax(logits, dim=-1).item()
                    log_prob = 0.0
                else:
                    # Sample from distribution
                    action_tensor, log_prob_tensor = self.policy_head.sample(actor_features)
                    action = action_tensor.item()
                    log_prob = log_prob_tensor.item()
            else:  # continuous
                if deterministic:
                    # Use mean action
                    mean, _ = self.policy_head(actor_features)
                    action = mean.cpu().numpy()[0]
                    log_prob = 0.0
                else:
                    # Sample from distribution
                    action_tensor, log_prob_tensor = self.policy_head.sample(actor_features)
                    action = action_tensor.cpu().numpy()[0]
                    log_prob = log_prob_tensor.item()

            # Get value estimate
            critic_features = self.critic_backbone(obs_tensor)
            value = self.value_head(critic_features).item()

        self.actor_backbone.train()
        self.critic_backbone.train()

        return action, log_prob, value

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Perform one PPO training step.

        Args:
            batch: Dictionary with rollout data

        Returns:
            Dictionary with training metrics
        """
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get current policy distribution and values
        actor_features = self.actor_backbone(observations)
        critic_features = self.critic_backbone(observations)

        # Evaluate actions under current policy
        if self.config.action_type == "discrete":
            log_probs, entropy = self.policy_head.evaluate(actor_features, actions)
        else:
            log_probs, entropy = self.policy_head.evaluate(actor_features, actions)

        values = self.value_head(critic_features).squeeze(-1)

        # Compute ratio: π_new / π_old
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss (MSE)
        value_loss = nn.functional.mse_loss(values, returns)

        # Compute entropy bonus (encourages exploration)
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        # Optimize
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(
            list(self.actor_backbone.parameters()) + list(self.policy_head.parameters()),
            self.config.max_grad_norm,
        )
        nn.utils.clip_grad_norm_(
            list(self.critic_backbone.parameters()) + list(self.value_head.parameters()),
            self.config.max_grad_norm,
        )

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
            "mean_ratio": ratio.mean().item(),
        }

        return metrics

    def update(self) -> dict[str, float]:
        """
        Update policy using collected rollout data.

        Returns:
            Dictionary with aggregated training metrics
        """
        # Get rollout data
        batch = self.rollout_buffer.get()

        # Track metrics
        all_metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "mean_ratio": [],
        }

        # Multiple epochs over the same data
        for _ in range(self.config.n_epochs):
            # Create mini-batches
            n_samples = len(batch["observations"])
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, n_samples)
                mb_indices = indices[start_idx:end_idx]

                # Create mini-batch
                mini_batch = {key: value[mb_indices] for key, value in batch.items()}

                # Training step
                metrics = self.train_step(mini_batch)

                # Collect metrics
                for key, value in metrics.items():
                    all_metrics[key].append(value)

        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

        # Add to agent metrics
        self.metrics.add_loss(avg_metrics["total_loss"])

        # Clear rollout buffer
        self.rollout_buffer.clear()

        return avg_metrics

    def collect_rollout(self, env, n_steps: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """
        Collect rollout from environment.

        Args:
            env: Environment to collect from
            n_steps: Number of steps to collect

        Returns:
            Tuple of (rollout_data, episode_info)
        """
        rollout_data = []
        episode_rewards = []
        episode_lengths = []

        observation, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

        for _ in range(n_steps):
            # Select action
            action, log_prob, value = self.select_action(observation, deterministic=False)

            # Take step
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store in buffer
            self.rollout_buffer.push(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )

            # Track episode stats
            episode_reward += reward
            episode_length += 1

            # Check if episode ended
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                self.metrics.add_episode(episode_reward, episode_length)

                observation, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                observation = next_observation

        # Compute advantages and returns
        if done:
            last_value = 0.0
        else:
            _, _, last_value = self.select_action(observation, deterministic=False)

        self.rollout_buffer.compute_returns_and_advantages(last_value)

        # Episode info
        episode_info = {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "n_episodes": len(episode_rewards),
        }

        return rollout_data, episode_info

    def save(self, path: str | Path) -> None:
        """Save agent to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "actor_backbone": self.actor_backbone.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "critic_backbone": self.critic_backbone.state_dict(),
                "value_head": self.value_head.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load agent from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor_backbone.load_state_dict(checkpoint["actor_backbone"])
        self.policy_head.load_state_dict(checkpoint["policy_head"])
        self.critic_backbone.load_state_dict(checkpoint["critic_backbone"])
        self.value_head.load_state_dict(checkpoint["value_head"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def train_mode(self) -> None:
        """Set networks to training mode."""
        self.actor_backbone.train()
        self.policy_head.train()
        self.critic_backbone.train()
        self.value_head.train()

    def eval_mode(self) -> None:
        """Set networks to evaluation mode."""
        self.actor_backbone.eval()
        self.policy_head.eval()
        self.critic_backbone.eval()
        self.value_head.eval()
