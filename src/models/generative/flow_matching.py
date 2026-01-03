"""Flow Matching for molecular generation.

Implements conditional flow matching for 3D molecular generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class FlowMatching(nn.Module):
    """Flow Matching generative model.

    Learns to model the velocity field of a probability flow from
    a simple prior (e.g., Gaussian) to the data distribution.

    Supports both DiT-style backbones (with internal time conditioning)
    and standard backbones (with external time embedding).
    """

    def __init__(
        self,
        backbone: nn.Module,
        time_steps: int = 1000,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
        schedule: str = "linear",
        loss_type: str = "mse"
    ):
        """Initialize Flow Matching model.

        Args:
            backbone: Neural network that predicts velocity field
            time_steps: Number of discretization steps (for schedule)
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            schedule: Noise schedule ("linear", "cosine")
            loss_type: Loss function type ("mse", "huber")
        """
        super().__init__()
        self.backbone = backbone
        self.time_steps = time_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.schedule = schedule
        self.loss_type = loss_type

        # Check if backbone supports internal time conditioning (DiT-style)
        self._dit_style = self._check_dit_style()

        # Time embedding for non-DiT backbones
        if not self._dit_style:
            self.time_embedding = TimeEmbedding(backbone.hidden_dim)

        # Velocity head will be created lazily
        self._velocity_head = None

    def _check_dit_style(self) -> bool:
        """Check if backbone handles time conditioning internally."""
        import inspect
        sig = inspect.signature(self.backbone.forward)
        return 't' in sig.parameters

    def get_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get interpolation schedule.

        For flow matching, we use linear interpolation:
        x_t = (1 - t) * x_0 + t * x_1
        where x_0 ~ N(0, I) and x_1 is data.

        Args:
            t: (B,) time values in [0, 1]

        Returns:
            alpha_t: Interpolation coefficient for data (t)
            sigma_t: Noise level (1 - t)
        """
        if self.schedule == "linear":
            # Standard linear interpolation for flow matching
            alpha_t = t
            sigma_t = 1 - t

        elif self.schedule == "cosine":
            # Cosine schedule
            alpha_t = 1 - torch.cos(t * math.pi / 2)
            sigma_t = torch.cos(t * math.pi / 2)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return alpha_t, sigma_t

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute flow matching loss.

        Args:
            positions: (N, 3) target positions
            atom_types: (N,) atom types
            edge_index: Optional (2, E) edge indices (for GNN-style backbones)
            batch_idx: (N,) batch indices

        Returns:
            loss: Scalar loss
            info: Dict with additional information
        """
        device = positions.device

        # Sample time uniformly in [0, 1]
        if batch_idx is None:
            B = 1
            t = torch.rand(1, device=device)
        else:
            B = batch_idx.max().item() + 1
            t = torch.rand(B, device=device)

        # Sample noise from prior
        noise = torch.randn_like(positions)

        # Get schedule
        alpha_t, sigma_t = self.get_schedule(t)

        # Expand for per-atom interpolation
        if batch_idx is not None:
            alpha_t_expanded = alpha_t[batch_idx].unsqueeze(-1)
            sigma_t_expanded = sigma_t[batch_idx].unsqueeze(-1)
            t_expanded = t[batch_idx]
        else:
            alpha_t_expanded = alpha_t.unsqueeze(-1)
            sigma_t_expanded = sigma_t.unsqueeze(-1)
            t_expanded = t.expand(positions.size(0))

        # Interpolate: x_t = sigma_t * x_0 + alpha_t * x_1
        # where x_0 ~ N(0, I) is noise, x_1 is data
        noisy_positions = sigma_t_expanded * noise + alpha_t_expanded * positions

        # Target velocity for conditional flow matching
        # v_t = x_1 - x_0 (constant velocity along straight path)
        target_velocity = positions - noise

        # Predict velocity
        if self._dit_style:
            # DiT handles time conditioning internally
            node_features = self.backbone(
                atom_types=atom_types,
                positions=noisy_positions,
                t=t,
                batch_idx=batch_idx,
                return_features=True
            )
        else:
            # Add time embedding externally
            time_emb = self.time_embedding(t_expanded)
            node_features = self.backbone(
                atom_types=atom_types,
                positions=noisy_positions,
                edge_index=edge_index,
                batch_idx=batch_idx,
                return_features=True
            )
            node_features = node_features + time_emb

        predicted_velocity = self.velocity_head(node_features)

        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted_velocity, target_velocity)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_velocity, target_velocity)
        elif self.loss_type == "l1":
            loss = F.l1_loss(predicted_velocity, target_velocity)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        info = {
            "loss": loss.item(),
            "mean_t": t.mean().item(),
        }

        return loss, info

    @torch.no_grad()
    def sample(
        self,
        atom_types: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """Sample molecular structures using Euler integration.

        Args:
            atom_types: (N,) atom types
            edge_index: Optional (2, E) edge indices
            batch_idx: (N,) batch indices
            num_steps: Number of sampling steps (default: 50)
            return_trajectory: If True, return full trajectory

        Returns:
            positions: (N, 3) sampled positions
            Or if return_trajectory: List of positions at each step
        """
        N = atom_types.size(0)
        device = atom_types.device

        # Determine number of graphs
        if batch_idx is None:
            B = 1
        else:
            B = batch_idx.max().item() + 1

        # Start from noise (prior)
        positions = torch.randn(N, 3, device=device)

        trajectory = [positions.clone()] if return_trajectory else None

        # ODE integration (Euler method)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)

            # Predict velocity
            if self._dit_style:
                node_features = self.backbone(
                    atom_types=atom_types,
                    positions=positions,
                    t=t,
                    batch_idx=batch_idx,
                    return_features=True
                )
            else:
                if batch_idx is not None:
                    t_expanded = t[batch_idx]
                else:
                    t_expanded = t.expand(N)
                time_emb = self.time_embedding(t_expanded)
                node_features = self.backbone(
                    atom_types=atom_types,
                    positions=positions,
                    edge_index=edge_index,
                    batch_idx=batch_idx,
                    return_features=True
                )
                node_features = node_features + time_emb

            velocity = self.velocity_head(node_features)

            # Update positions
            positions = positions + velocity * dt

            if return_trajectory:
                trajectory.append(positions.clone())

        if return_trajectory:
            return trajectory
        return positions

    @property
    def velocity_head(self) -> nn.Module:
        """Velocity prediction head (created lazily)."""
        if self._velocity_head is None:
            self._velocity_head = nn.Sequential(
                nn.Linear(self.backbone.hidden_dim, self.backbone.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.backbone.hidden_dim, 3)
            )
            # Move to same device as backbone
            device = next(self.backbone.parameters()).device
            self._velocity_head = self._velocity_head.to(device)
        return self._velocity_head

    @velocity_head.setter
    def velocity_head(self, value: nn.Module):
        """Set velocity head."""
        self._velocity_head = value


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding with MLP projection.

    Used for backbones that don't handle time conditioning internally.
    """

    def __init__(self, dim: int, freq_dim: int = 256):
        """Initialize TimeEmbedding.

        Args:
            dim: Output embedding dimension
            freq_dim: Frequency embedding dimension
        """
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed time values.

        Args:
            t: (B,) or (N,) time values in [0, 1]

        Returns:
            embedding: (B, dim) or (N, dim)
        """
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return self.mlp(emb)
