"""Graph Neural Network for molecular modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MessagePassingLayer(nn.Module):
    """Message passing layer for GNN."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 4,
        activation: str = "silu",
        dropout: float = 0.1,
        aggregation: str = "sum"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def _get_activation(self, name: str):
        activations = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        return activations.get(name, nn.SiLU())

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: (N, hidden_dim)
            edge_index: (2, E)
            edge_features: (E, edge_dim)

        Returns:
            updated_features: (N, hidden_dim)
        """
        src, dst = edge_index
        N = node_features.size(0)

        # Construct messages
        src_features = node_features[src]  # (E, hidden_dim)
        dst_features = node_features[dst]  # (E, hidden_dim)

        # Concatenate source, destination, and edge features
        message_input = torch.cat([src_features, dst_features, edge_features], dim=1)
        messages = self.message_net(message_input)  # (E, hidden_dim)

        # Aggregate messages
        aggregated = torch.zeros(N, self.hidden_dim, device=node_features.device)

        if self.aggregation == "sum":
            aggregated.index_add_(0, dst, messages)
        elif self.aggregation == "mean":
            aggregated.index_add_(0, dst, messages)
            count = torch.zeros(N, device=node_features.device)
            count.index_add_(0, dst, torch.ones(len(dst), device=node_features.device))
            aggregated = aggregated / (count.unsqueeze(1) + 1e-8)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Update node features
        update_input = torch.cat([node_features, aggregated], dim=1)
        updated = self.update_net(update_input)

        # Residual connection and normalization
        output = self.norm(node_features + updated)

        return output


class GNN(nn.Module):
    """Graph Neural Network for molecular modeling.

    Uses message passing to update node features based on graph structure.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_atom_types: int = 10,
        edge_dim: int = 4,
        dropout: float = 0.1,
        activation: str = "silu",
        aggregation: str = "sum",
        readout: str = "sum"
    ):
        """Initialize GNN.

        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of message passing layers
            num_atom_types: Number of atom types to embed
            edge_dim: Edge feature dimension
            dropout: Dropout rate
            activation: Activation function
            aggregation: Message aggregation method
            readout: Graph readout method
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.readout = readout

        # Atom type embedding
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                activation=activation,
                dropout=dropout,
                aggregation=aggregation
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            atom_types: (N,) atom type indices
            positions: (N, 3) 3D coordinates
            edge_index: (2, E) edge indices
            batch_idx: (N,) batch indices for graph-level readout
            return_features: If True, return node features instead of graph features

        Returns:
            If return_features:
                node_features: (N, hidden_dim)
            Else:
                graph_features: (B, hidden_dim)
        """
        # Embed atom types
        node_features = self.atom_embedding(atom_types)  # (N, hidden_dim)

        # Compute edge features from positions
        edge_features = self._compute_edge_features(positions, edge_index)

        # Message passing
        for layer in self.mp_layers:
            node_features = layer(node_features, edge_index, edge_features)

        # Output projection
        node_features = self.output_proj(node_features)

        if return_features:
            return node_features

        # Graph-level readout
        if batch_idx is None:
            # Single graph
            if self.readout == "sum":
                graph_features = node_features.sum(dim=0, keepdim=True)
            elif self.readout == "mean":
                graph_features = node_features.mean(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown readout: {self.readout}")
        else:
            # Batched graphs
            num_graphs = batch_idx.max().item() + 1
            graph_features = torch.zeros(
                num_graphs, self.hidden_dim, device=node_features.device
            )

            if self.readout == "sum":
                graph_features.index_add_(0, batch_idx, node_features)
            elif self.readout == "mean":
                graph_features.index_add_(0, batch_idx, node_features)
                count = torch.zeros(num_graphs, device=node_features.device)
                count.index_add_(
                    0, batch_idx, torch.ones(len(batch_idx), device=node_features.device)
                )
                graph_features = graph_features / count.unsqueeze(1)
            else:
                raise ValueError(f"Unknown readout: {self.readout}")

        return graph_features

    def _compute_edge_features(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute edge features from positions.

        Features include:
        - Distance (1D)
        - Normalized direction vector (3D)

        Args:
            positions: (N, 3)
            edge_index: (2, E)

        Returns:
            edge_features: (E, 4)
        """
        src, dst = edge_index
        diff = positions[src] - positions[dst]  # (E, 3)
        distances = torch.norm(diff, dim=1, keepdim=True)  # (E, 1)
        directions = diff / (distances + 1e-8)  # (E, 3)

        edge_features = torch.cat([distances, directions], dim=1)
        return edge_features

    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
