"""Simple MLP architecture for 2D toy data."""

import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for 2D data.

    This is designed for toy 2D datasets where we want to visualize
    the generative model training process.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        """Initialize MLP.

        Args:
            input_dim: Input dimension (2 for 2D data)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function (relu, silu, gelu)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        # Output projection (will be used by generative model)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_features: If True, return features before output projection

        Returns:
            If return_features: features of shape (batch_size, hidden_dim)
            Otherwise: output of shape (batch_size, input_dim)
        """
        features = self.network(x)

        if return_features:
            return features

        return self.output_proj(features)


class TimeConditionedMLP(nn.Module):
    """MLP with time conditioning for flow matching / diffusion.

    This explicitly conditions on time for the generative model.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        activation: str = "silu",
        time_embedding_dim: int = 32
    ):
        """Initialize time-conditioned MLP.

        Args:
            input_dim: Input dimension (2 for 2D data)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function
            time_embedding_dim: Dimension for time embeddings
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Time embedding layers
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            self.activation,
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        # Build network
        layers = []

        # Input layer (concatenate x and time embedding)
        layers.append(nn.Linear(input_dim + time_embedding_dim, hidden_dim))
        layers.append(self.activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """Forward pass with time conditioning.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t: Time tensor of shape (batch_size, 1) or (batch_size,)
            return_features: If True, return features before output projection

        Returns:
            Output tensor of shape (batch_size, input_dim)
        """
        # Ensure time has shape (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Get time embedding
        t_emb = self.time_mlp(t)

        # Concatenate input and time embedding
        x_t = torch.cat([x, t_emb], dim=-1)

        # Process through network
        features = self.network(x_t)

        if return_features:
            return features

        return self.output_proj(features)
