"""
Base neural network classes for RL models.

Provides foundational network architectures and configurations for all RL agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class NetworkConfig:
    """Configuration for neural networks."""

    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    dropout: float = 0.0
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    device: str = "cpu"


class BaseNetwork(nn.Module, ABC):
    """
    Abstract base class for all neural networks.

    Provides common functionality like initialization, device management,
    and forward pass interface.
    """

    def __init__(self, config: NetworkConfig):
        """
        Initialize base network.

        Args:
            config: Network configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def get_activation(self) -> nn.Module:
        """Get activation function from config."""
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }

        activation = self.config.activation.lower()
        if activation not in activation_map:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(activation_map.keys())}"
            )

        return activation_map[activation]

    def to_device(self, device: str | None = None) -> "BaseNetwork":
        """Move network to device."""
        if device is not None:
            self.device = torch.device(device)
        self.to(self.device)
        return self

    def save(self, path: str) -> None:
        """Save network parameters."""
        torch.save({"state_dict": self.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: str) -> "BaseNetwork":
        """Load network from file."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        network = cls(config)
        network.load_state_dict(checkpoint["state_dict"])
        return network

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        """Freeze all parameters (no gradient updates)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters (enable gradient updates)."""
        for param in self.parameters():
            param.requires_grad = True


def init_weights(module: nn.Module) -> None:
    """
    Initialize network weights using best practices.

    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        # Kaiming/He initialization for convolutional layers
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d):
        # Standard initialization for normalization layers
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
        # Orthogonal initialization for RNN layers
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)
