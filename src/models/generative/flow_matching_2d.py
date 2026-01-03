"""Flow matching for 2D data (simplified version for point clouds and distributions)."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class FlowMatching2D(nn.Module):
    """Flow Matching for 2D point clouds and distributions.

    Simplified version that works directly with 2D coordinates without
    molecular-specific features like atom types and edges.
    """

    def __init__(
        self,
        backbone: nn.Module,
        time_steps: int = 1000,
        sigma_min: float = 0.001,
        sigma_max: float = 1.0
    ):
        """Initialize 2D Flow Matching model.

        Args:
            backbone: Neural network backbone (MLP or other)
            time_steps: Number of discretization steps
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        super().__init__()
        self.backbone = backbone
        self.time_steps = time_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(
        self,
        positions: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for training.

        Args:
            positions: Target positions of shape (num_points, 2)
            batch_idx: Batch indices of shape (num_points,)

        Returns:
            loss: Scalar loss value
            info: Dict with additional information
        """
        batch_size = positions.shape[0]

        # Sample random time
        t = torch.rand(batch_size, device=positions.device)

        # Sample noise
        noise = torch.randn_like(positions)

        # Interpolate between noise and data: x_t = t * x_1 + (1 - t) * x_0
        # where x_0 is noise and x_1 is data
        t_expanded = t.unsqueeze(-1)  # (batch_size, 1)
        noisy_positions = t_expanded * positions + (1 - t_expanded) * noise

        # Target velocity: dx/dt = x_1 - x_0
        target_velocity = positions - noise

        # Predict velocity using backbone
        predicted_velocity = self.backbone(noisy_positions, return_features=False)

        # Compute loss (MSE between predicted and target velocity)
        loss = torch.mean((predicted_velocity - target_velocity) ** 2)

        info = {
            "velocity_mse": loss.item(),
            "mean_time": t.mean().item()
        }

        return loss, info

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        num_steps: int = 100
    ) -> torch.Tensor:
        """Generate samples by integrating the flow.

        Args:
            noise: Initial noise of shape (num_points, 2)
            num_steps: Number of integration steps

        Returns:
            Generated positions of shape (num_points, 2)
        """
        positions = noise.clone()
        dt = 1.0 / num_steps

        for step in range(num_steps):
            # Get velocity from model
            velocity = self.backbone(positions, return_features=False)

            # Euler integration step
            positions = positions + velocity * dt

        return positions
