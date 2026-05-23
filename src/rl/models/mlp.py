"""
Multi-Layer Perceptron (MLP) networks.

Standard feedforward neural networks for RL agents.
Inspired by the architecture from reference notebooks.
"""

import torch
import torch.nn as nn

from src.rl.models.base import BaseNetwork, NetworkConfig, init_weights


class MLPNetwork(BaseNetwork):
    """
    Multi-Layer Perceptron network.

    Simple feedforward architecture:
    input -> hidden_1 -> ... -> hidden_n -> output

    Example from notebooks: 5 -> 256 -> 128 -> 64 -> 3
    """

    def __init__(self, config: NetworkConfig):
        """
        Initialize MLP network.

        Args:
            config: Network configuration
        """
        super().__init__(config)

        layers: list[nn.Module] = []

        # Input layer
        prev_dim = config.input_dim

        # Hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            layers.append(self.get_activation())

            # Dropout
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(init_weights)

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        return self.network(x)


class ResidualMLPNetwork(BaseNetwork):
    """
    MLP with residual connections for better gradient flow.

    Uses skip connections between layers to prevent vanishing gradients
    and enable deeper networks.
    """

    def __init__(self, config: NetworkConfig):
        """
        Initialize Residual MLP network.

        Args:
            config: Network configuration
        """
        super().__init__(config)

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dims[0])

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(config.hidden_dims) - 1):
            block = ResidualBlock(
                config.hidden_dims[i],
                config.hidden_dims[i + 1],
                self.get_activation(),
                config.dropout,
                config.use_layer_norm,
            )
            self.residual_blocks.append(block)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dims[-1], config.output_dim)

        # Initialize weights
        self.apply(init_weights)

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Input projection
        x = self.input_proj(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        """
        Initialize residual block.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else None

        # Skip connection projection if dimensions don't match
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        # First linear + activation
        out = self.linear1(x)
        if self.layer_norm:
            out = self.layer_norm(out)
        out = self.activation(out)

        # Dropout
        if self.dropout:
            out = self.dropout(out)

        # Second linear
        out = self.linear2(out)

        # Skip connection
        if self.skip_proj:
            identity = self.skip_proj(identity)

        # Add residual
        out = out + identity

        # Final activation
        out = self.activation(out)

        return out


class DuelingMLPNetwork(BaseNetwork):
    """
    Dueling MLP architecture for Q-learning.

    Splits into two streams:
    - Value stream V(s): evaluates state value
    - Advantage stream A(s,a): evaluates action advantages

    Combines as: Q(s,a) = V(s) + (A(s,a) - mean(A))

    Based on the Dueling DQN from the reference notebooks.
    """

    def __init__(self, config: NetworkConfig):
        """
        Initialize Dueling MLP network.

        Args:
            config: Network configuration
        """
        super().__init__(config)

        # Common feature extractor
        common_layers: list[nn.Module] = []
        prev_dim = config.input_dim

        # Use first half of hidden layers for common features
        n_common = max(1, len(config.hidden_dims) // 2)
        for i in range(n_common):
            hidden_dim = config.hidden_dims[i]
            common_layers.append(nn.Linear(prev_dim, hidden_dim))
            common_layers.append(self.get_activation())
            if config.dropout > 0:
                common_layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        self.common = nn.Sequential(*common_layers)

        # Value stream (outputs single value)
        value_layers: list[nn.Module] = []
        for i in range(n_common, len(config.hidden_dims)):
            hidden_dim = config.hidden_dims[i]
            value_layers.append(nn.Linear(prev_dim, hidden_dim))
            value_layers.append(self.get_activation())
            prev_dim = hidden_dim

        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_stream = nn.Sequential(*value_layers)

        # Advantage stream (outputs per-action advantages)
        advantage_layers: list[nn.Module] = []
        prev_dim = config.hidden_dims[n_common - 1]
        for i in range(n_common, len(config.hidden_dims)):
            hidden_dim = config.hidden_dims[i]
            advantage_layers.append(nn.Linear(prev_dim, hidden_dim))
            advantage_layers.append(self.get_activation())
            prev_dim = hidden_dim

        advantage_layers.append(nn.Linear(prev_dim, config.output_dim))
        self.advantage_stream = nn.Sequential(*advantage_layers)

        # Initialize weights
        self.apply(init_weights)

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Q-values tensor [batch_size, output_dim]
        """
        # Common features
        features = self.common(x)

        # Value and advantage streams
        value = self.value_stream(features)  # [batch_size, 1]
        advantage = self.advantage_stream(features)  # [batch_size, n_actions]

        # Combine: Q = V + (A - mean(A))
        # Subtract mean advantage for stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
