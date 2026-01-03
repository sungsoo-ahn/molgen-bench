"""Pairformer architecture for molecular modeling.

Based on AlphaFold2's pair representation and triangle updates.
"""

import torch
import torch.nn as nn
from typing import Optional


class TriangleMultiplication(nn.Module):
    """Triangle multiplication update from AlphaFold2."""

    def __init__(self, pair_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pair_dim = pair_dim

        self.linear_a = nn.Linear(pair_dim, pair_dim)
        self.linear_b = nn.Linear(pair_dim, pair_dim)
        self.linear_g = nn.Linear(pair_dim, pair_dim)
        self.linear_out = nn.Linear(pair_dim, pair_dim)

        self.norm = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
        """Triangle multiplication update.

        Args:
            pair_rep: (B, N, N, pair_dim)

        Returns:
            updated: (B, N, N, pair_dim)
        """
        B, N, _, D = pair_rep.shape

        # Project
        a = self.linear_a(pair_rep)  # (B, N, N, D)
        b = self.linear_b(pair_rep)
        g = torch.sigmoid(self.linear_g(pair_rep))

        # Triangle multiplication (outgoing)
        # p_ik = sum_j p_ij * p_jk
        a = a.transpose(-2, -3)  # (B, N, N, D)
        update = torch.einsum('bijc,bjkc->bikc', a, b)  # (B, N, N, D)

        # Gate and output
        update = g * update
        update = self.linear_out(update)
        update = self.dropout(update)

        return self.norm(pair_rep + update)


class Pairformer(nn.Module):
    """Pairformer architecture with pair representations.

    This is a skeleton based on AlphaFold2's architecture.
    Full implementation should include:
    - Triangle attention
    - Proper pair-to-node and node-to-pair updates
    - Invariant geometric features
    """

    def __init__(
        self,
        node_dim: int = 256,
        pair_dim: int = 128,
        num_layers: int = 4,
        num_atom_types: int = 10,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_triangle_multiplication: bool = True
    ):
        """Initialize Pairformer.

        Args:
            node_dim: Node representation dimension
            pair_dim: Pair representation dimension
            num_layers: Number of pairformer layers
            num_atom_types: Number of atom types
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_triangle_multiplication: Whether to use triangle multiplication
        """
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.num_layers = num_layers

        # Atom embedding
        self.atom_embedding = nn.Embedding(num_atom_types, node_dim)

        # Pair initialization
        self.pair_init = nn.Linear(1, pair_dim)  # Distance-based initialization

        # Pairformer layers
        self.layers = nn.ModuleList([
            PairformerLayer(
                node_dim=node_dim,
                pair_dim=pair_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_triangle_multiplication=use_triangle_multiplication
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(node_dim, node_dim)

    def forward(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            atom_types: (N,) atom types
            positions: (N, 3) positions
            batch_idx: Optional batch indices
            return_features: If True, return node features

        Returns:
            features
        """
        N = atom_types.size(0)

        # Node features
        node_features = self.atom_embedding(atom_types)  # (N, node_dim)

        # Pair features (distance-based)
        distances = torch.cdist(positions, positions).unsqueeze(-1)  # (N, N, 1)
        pair_features = self.pair_init(distances)  # (N, N, pair_dim)

        # Add batch dimension
        node_features = node_features.unsqueeze(0)  # (1, N, node_dim)
        pair_features = pair_features.unsqueeze(0)  # (1, N, N, pair_dim)

        # Pairformer layers
        for layer in self.layers:
            node_features, pair_features = layer(node_features, pair_features)

        # Output
        node_features = self.output_proj(node_features)
        node_features = node_features.squeeze(0)  # (N, node_dim)

        if return_features:
            return node_features

        # Graph-level pooling
        if batch_idx is None:
            graph_features = node_features.mean(dim=0, keepdim=True)
        else:
            num_graphs = batch_idx.max().item() + 1
            graph_features = torch.zeros(
                num_graphs, self.node_dim, device=node_features.device
            )
            graph_features.index_add_(0, batch_idx, node_features)
            count = torch.zeros(num_graphs, device=node_features.device)
            count.index_add_(
                0, batch_idx, torch.ones(len(batch_idx), device=node_features.device)
            )
            graph_features = graph_features / count.unsqueeze(1)

        return graph_features

    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PairformerLayer(nn.Module):
    """Single Pairformer layer."""

    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        num_heads: int,
        dropout: float,
        use_triangle_multiplication: bool
    ):
        super().__init__()

        # Node self-attention
        self.node_attention = nn.MultiheadAttention(
            node_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.node_norm = nn.LayerNorm(node_dim)

        # Pair update
        if use_triangle_multiplication:
            self.pair_update = TriangleMultiplication(pair_dim, dropout)
        else:
            self.pair_update = nn.Identity()

        # Node transition
        self.node_transition = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout)
        )
        self.node_transition_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        pair_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            node_features: (B, N, node_dim)
            pair_features: (B, N, N, pair_dim)

        Returns:
            updated_node_features, updated_pair_features
        """
        # Node self-attention
        attn_out, _ = self.node_attention(
            node_features, node_features, node_features
        )
        node_features = self.node_norm(node_features + attn_out)

        # Pair update
        pair_features = self.pair_update(pair_features)

        # Node transition
        transition_out = self.node_transition(node_features)
        node_features = self.node_transition_norm(node_features + transition_out)

        return node_features, pair_features
