"""Flow Matching for molecular generation.

Implements conditional flow matching for 3D molecular generation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


class FlowMatching(nn.Module):
    """Flow Matching generative model.

    Learns to model the velocity field of a probability flow from
    a simple prior (e.g., Gaussian) to the data distribution.
    """

    def __init__(
        self,
        backbone: nn.Module,
        time_steps: int = 1000,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0,
        schedule: str = "cosine",
        loss_type: str = "mse"
    ):
        """Initialize Flow Matching model.

        Args:
            backbone: Neural network that predicts velocity field
            time_steps: Number of discretization steps
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            schedule: Noise schedule ("linear", "cosine")
            loss_type: Loss function type
        """
        super().__init__()
        self.backbone = backbone
        self.time_steps = time_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.schedule = schedule
        self.loss_type = loss_type

        # Time embedding
        self.time_embedding = TimeEmbedding(backbone.hidden_dim)

    def get_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get interpolation schedule.

        Args:
            t: (B,) time values in [0, 1]

        Returns:
            alpha_t: Interpolation coefficient for data
            sigma_t: Noise level
        """
        if self.schedule == "linear":
            alpha_t = t
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)

        elif self.schedule == "cosine":
            # Cosine schedule as used in guided diffusion
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2) * self.sigma_max

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return alpha_t, sigma_t

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute flow matching loss.

        Args:
            positions: (N, 3) target positions
            atom_types: (N,) atom types
            edge_index: (2, E) edge indices
            batch_idx: (N,) batch indices

        Returns:
            loss: Scalar loss
            info: Dict with additional information
        """
        N = positions.size(0)
        device = positions.device

        # Sample time
        if batch_idx is None:
            B = 1
            t = torch.rand(1, device=device)
        else:
            B = batch_idx.max().item() + 1
            t = torch.rand(B, device=device)
            # Expand to match each atom
            t_expanded = t[batch_idx]

        # Sample noise
        noise = torch.randn_like(positions)

        # Get schedule
        alpha_t, sigma_t = self.get_schedule(t)

        if batch_idx is not None:
            alpha_t = alpha_t[batch_idx].unsqueeze(-1)
            sigma_t = sigma_t[batch_idx].unsqueeze(-1)
        else:
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        # Interpolate: x_t = alpha_t * x_1 + sigma_t * x_0
        # where x_0 ~ N(0, I) and x_1 is data
        noisy_positions = alpha_t * positions + sigma_t * noise

        # Target velocity field (conditional flow matching)
        # v_t = d/dt [alpha_t * x_1 + sigma_t * x_0]
        #     = alpha'_t * x_1 + sigma'_t * x_0
        # For linear: alpha'_t = 1, sigma'_t = -(sigma_max - sigma_min)
        # Simplified target: v_t = (x_1 - x_t) / (1 - t)
        if batch_idx is not None:
            t_for_target = t[batch_idx].unsqueeze(-1)
        else:
            t_for_target = t.unsqueeze(-1)

        target_velocity = (positions - noisy_positions) / (1 - t_for_target + 1e-5)

        # Predict velocity with backbone
        # Add time embedding to node features
        time_emb = self.time_embedding(t if batch_idx is None else t[batch_idx])

        # Get node features from backbone
        node_features = self.backbone(
            atom_types=atom_types,
            positions=noisy_positions,
            edge_index=edge_index,
            batch_idx=batch_idx,
            return_features=True
        )

        # Add time embedding
        node_features = node_features + time_emb

        # Predict velocity (simple MLP on top of features)
        predicted_velocity = self.velocity_head(node_features)

        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted_velocity, target_velocity)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_velocity, target_velocity)
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
        edge_index: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """Sample molecular structures.

        Args:
            atom_types: (N,) atom types
            edge_index: (2, E) edge indices
            batch_idx: (N,) batch indices
            num_steps: Number of sampling steps
            return_trajectory: If True, return full trajectory

        Returns:
            positions: (N, 3) sampled positions
            Or if return_trajectory: List of positions
        """
        N = atom_types.size(0)
        device = atom_types.device

        # Start from noise
        positions = torch.randn(N, 3, device=device) * self.sigma_max

        trajectory = [positions.clone()] if return_trajectory else None

        # ODE integration (Euler method)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((1 if batch_idx is None else batch_idx.max().item() + 1,),
                          t_val, device=device)

            # Get time embedding
            time_emb = self.time_embedding(t if batch_idx is None else t[batch_idx])

            # Predict velocity
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
    def velocity_head(self):
        """Create velocity prediction head if not exists."""
        if not hasattr(self, "_velocity_head"):
            self._velocity_head = nn.Sequential(
                nn.Linear(self.backbone.hidden_dim, self.backbone.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.backbone.hidden_dim, 3)
            ).to(next(self.backbone.parameters()).device)
        return self._velocity_head


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed time values.

        Args:
            t: (B,) or (N,) time values in [0, 1]

        Returns:
            embedding: (B, dim) or (N, dim)
        """
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if emb.size(-1) < self.dim:
            # Pad if odd dimension
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return self.proj(emb)


import torch.nn.functional as F
