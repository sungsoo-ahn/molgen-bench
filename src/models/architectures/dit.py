"""DiT (Diffusion Transformer) architecture for molecular modeling.

Based on MinCatFlow's transformer implementation with adaptive layer normalization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embeddings with MLP projection.

    Converts scalar timesteps into vector representations using
    sinusoidal positional encoding followed by a learnable MLP.
    """

    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        """Initialize TimestepEmbedder.

        Args:
            hidden_dim: Output embedding dimension
            freq_dim: Frequency embedding dimension (must be even)
        """
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: (B,) or (N,) timestep values in [0, 1]

        Returns:
            embedding: (B, hidden_dim) or (N, hidden_dim)
        """
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class Mlp(nn.Module):
    """Vision Transformer-style MLP with GELU activation.

    Two-layer MLP with optional layer normalization and dropout.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: Optional[nn.Module] = None,
        dropout: float = 0.0
    ):
        """Initialize MLP.

        Args:
            in_features: Input feature dimension
            hidden_features: Hidden dimension (default: 4x in_features)
            out_features: Output dimension (default: in_features)
            act_layer: Activation function
            norm_layer: Optional normalization layer
            dropout: Dropout rate
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(approximate='tanh') if act_layer == nn.GELU else act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (*, in_features) input tensor

        Returns:
            output: (*, out_features) output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    """Diffusion Transformer Block with adaptive layer normalization (adaLN-Zero).

    Uses time conditioning to modulate layer normalization, enabling
    the model to adapt its behavior based on the diffusion timestep.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        """Initialize DiTBlock.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension multiplier
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )

        # Adaptive layer norm modulation
        # Outputs: shift1, scale1, gate1, shift2, scale2, gate2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        # Initialize modulation to zero for stable training
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with time conditioning.

        Args:
            x: (B, N, hidden_dim) input features
            c: (B, hidden_dim) conditioning (timestep embedding)
            mask: Optional (B, N) padding mask (True = masked)

        Returns:
            output: (B, N, hidden_dim) output features
        """
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)  # (B, 6 * hidden_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)

        # Expand for broadcasting: (B, hidden_dim) -> (B, 1, hidden_dim)
        shift1 = shift1.unsqueeze(1)
        scale1 = scale1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        # Self-attention with adaLN
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale1) + shift1

        # Convert mask to key_padding_mask format if provided
        key_padding_mask = mask if mask is not None else None

        attn_out, _ = self.attn(
            x_modulated, x_modulated, x_modulated,
            key_padding_mask=key_padding_mask
        )
        x = x + gate1 * attn_out

        # MLP with adaLN
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale2) + shift2
        mlp_out = self.mlp(x_modulated)
        x = x + gate2 * mlp_out

        return x


class DiT(nn.Module):
    """Diffusion Transformer for molecular modeling.

    Stacks DiTBlock layers with timestep conditioning for
    flow matching / diffusion-based molecular generation.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_atom_types: int = 100,
        dropout: float = 0.1,
        max_atoms: int = 256,
        use_checkpoint: bool = False
    ):
        """Initialize DiT.

        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension multiplier
            num_atom_types: Number of atom types for embedding
            dropout: Dropout rate
            max_atoms: Maximum number of atoms (for positional encoding)
            use_checkpoint: Use gradient checkpointing to save memory
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

        # Atom embedding
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)

        # Position embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_atoms, hidden_dim) * 0.02)

        # Coordinate projection
        self.coord_proj = nn.Linear(3, hidden_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedder(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            atom_types: (N,) or (B, N) atom type indices
            positions: (N, 3) or (B, N, 3) 3D coordinates
            t: Optional (B,) timestep values in [0, 1]
            batch_idx: Optional (N,) batch indices for unbatched input
            return_features: If True, return node features

        Returns:
            features: (B, N, hidden_dim) if return_features else (B, hidden_dim)
        """
        # Handle batching
        if atom_types.dim() == 1:
            # Unbatched input - need to batch it
            if batch_idx is not None:
                # Multiple graphs batched together
                x, mask, batch_sizes = self._batch_from_indices(
                    atom_types, positions, batch_idx
                )
            else:
                # Single graph
                atom_types = atom_types.unsqueeze(0)
                positions = positions.unsqueeze(0)
                x = None
                mask = None
        else:
            x = None
            mask = None

        if x is None:
            B, N = atom_types.shape
            device = atom_types.device

            # Embed atoms
            x = self.atom_embedding(atom_types)  # (B, N, hidden_dim)

            # Add positional encoding
            x = x + self.pos_embedding[:, :N, :]

            # Add coordinate information
            x = x + self.coord_proj(positions)

            mask = None

        B = x.shape[0]
        device = x.device

        # Get timestep embedding
        if t is None:
            t = torch.zeros(B, device=device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(B)

        c = self.time_embed(t)  # (B, hidden_dim)

        # Apply transformer blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, c, mask, use_reentrant=False)
            else:
                x = block(x, c, mask)

        # Final normalization
        x = self.final_norm(x)

        # Output projection
        x = self.output_proj(x)

        if return_features:
            # Return node features
            if batch_idx is not None and atom_types.dim() == 1:
                # Unbatch back to original format
                return self._unbatch_features(x, mask, batch_idx)
            return x.squeeze(0) if x.shape[0] == 1 and batch_idx is None else x

        # Pool to graph level (mean pooling)
        if mask is not None:
            # Masked mean
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return x.squeeze(0) if x.shape[0] == 1 and batch_idx is None else x

    def _batch_from_indices(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        batch_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Convert batched indices format to padded batch format.

        Args:
            atom_types: (N,) atom types
            positions: (N, 3) positions
            batch_idx: (N,) batch indices

        Returns:
            x: (B, max_N, hidden_dim) embedded features
            mask: (B, max_N) padding mask (True = padded)
            batch_sizes: List of sizes per graph
        """
        device = atom_types.device
        num_graphs = batch_idx.max().item() + 1

        # Get sizes per graph
        batch_sizes = []
        for i in range(num_graphs):
            batch_sizes.append((batch_idx == i).sum().item())
        max_size = max(batch_sizes)

        # Create padded tensors
        padded_types = torch.zeros(num_graphs, max_size, dtype=atom_types.dtype, device=device)
        padded_pos = torch.zeros(num_graphs, max_size, 3, device=device)
        mask = torch.ones(num_graphs, max_size, dtype=torch.bool, device=device)

        # Fill in values
        for i in range(num_graphs):
            graph_mask = batch_idx == i
            size = batch_sizes[i]
            padded_types[i, :size] = atom_types[graph_mask]
            padded_pos[i, :size] = positions[graph_mask]
            mask[i, :size] = False

        # Embed
        x = self.atom_embedding(padded_types)
        x = x + self.pos_embedding[:, :max_size, :]
        x = x + self.coord_proj(padded_pos)

        return x, mask, batch_sizes

    def _unbatch_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Convert padded batch features back to indices format.

        Args:
            x: (B, max_N, hidden_dim) features
            mask: (B, max_N) padding mask
            batch_idx: (N,) original batch indices

        Returns:
            features: (N, hidden_dim) unbatched features
        """
        device = x.device
        N = batch_idx.shape[0]
        features = torch.zeros(N, x.shape[-1], device=device)

        num_graphs = x.shape[0]
        offset = 0
        for i in range(num_graphs):
            size = (~mask[i]).sum().item()
            features[offset:offset + size] = x[i, :size]
            offset += size

        return features

    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
