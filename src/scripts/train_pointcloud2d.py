"""Training script for 2D point cloud datasets with visualization.

This script trains generative models on 2D point clouds (shapes like circles, stars, etc.)
and visualizes how the model learns to generate these shapes from noise.
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb

from src.data import PointCloud2DDataset, collate_pointcloud2d
from src.models.architectures import GNN, MLP
from src.models.generative import FlowMatching2D


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def visualize_point_clouds(
    real_samples,
    generated_samples,
    epoch,
    output_dir,
    num_vis=9
):
    """Visualize real vs generated point clouds.

    Args:
        real_samples: List of real point cloud samples (each is np.ndarray)
        generated_samples: List of generated point cloud samples
        epoch: Current epoch number
        output_dir: Directory to save plots
        num_vis: Number of samples to visualize
    """
    num_vis = min(num_vis, len(real_samples), len(generated_samples))
    fig, axes = plt.subplots(3, num_vis, figsize=(2.5 * num_vis, 7.5))

    if num_vis == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_vis):
        # Real data
        real_pts = real_samples[i]
        axes[0, i].scatter(real_pts[:, 0], real_pts[:, 1], s=10, alpha=0.6)
        axes[0, i].set_xlim(-2, 2)
        axes[0, i].set_ylim(-2, 2)
        axes[0, i].set_aspect('equal')
        axes[0, i].grid(True, alpha=0.3)
        if i == 0:
            axes[0, i].set_ylabel('Real', fontsize=12)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Generated data
        gen_pts = generated_samples[i]
        axes[1, i].scatter(gen_pts[:, 0], gen_pts[:, 1], s=10, alpha=0.6, c='red')
        axes[1, i].set_xlim(-2, 2)
        axes[1, i].set_ylim(-2, 2)
        axes[1, i].set_aspect('equal')
        axes[1, i].grid(True, alpha=0.3)
        if i == 0:
            axes[1, i].set_ylabel('Generated', fontsize=12)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # Overlay
        axes[2, i].scatter(real_pts[:, 0], real_pts[:, 1], s=10, alpha=0.4, c='blue', label='Real')
        axes[2, i].scatter(gen_pts[:, 0], gen_pts[:, 1], s=10, alpha=0.4, c='red', label='Gen')
        axes[2, i].set_xlim(-2, 2)
        axes[2, i].set_ylim(-2, 2)
        axes[2, i].set_aspect('equal')
        axes[2, i].grid(True, alpha=0.3)
        if i == 0:
            axes[2, i].set_ylabel('Overlay', fontsize=12)
            axes[2, i].legend(fontsize=8, loc='upper right')
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])

    plt.suptitle(f'Epoch {epoch}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_samples(model, num_samples, num_points_range, device):
    """Generate point cloud samples from the model.

    Args:
        model: Trained generative model
        num_samples: Number of samples to generate
        num_points_range: (min, max) number of points per sample
        device: Device to run on

    Returns:
        List of numpy arrays, each of shape (num_points, 2)
    """
    model.eval()
    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Random number of points
            num_points = np.random.randint(num_points_range[0], num_points_range[1] + 1)

            # Sample from noise
            noise = torch.randn(num_points, 2).to(device)

            # Generate through flow
            # For 2D point clouds, we need to handle the batch format differently
            # Create a dummy batch with just this sample
            batch_idx = torch.zeros(num_points, dtype=torch.long).to(device)

            # Sample using the model
            # We need to integrate the flow from t=0 to t=1
            positions = noise
            num_steps = 100
            dt = 1.0 / num_steps

            for step in range(num_steps):
                t = step * dt
                t_tensor = torch.full((num_points,), t).to(device)

                # Get velocity from model
                # For flow matching with 2D data, we pass positions directly
                velocity = model.backbone(positions)

                # Euler step
                positions = positions + velocity * dt

            samples.append(positions.cpu().numpy())

    model.train()
    return samples


def main(args):
    """Main training function."""
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Initialize WandB
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", True):
        wandb.init(
            project=wandb_config.get("project", "molgen-bench"),
            entity=wandb_config.get("entity", None),
            name=wandb_config.get("name", None),
            config=config,
            tags=wandb_config.get("tags", []),
        )
        # Log code
        wandb.run.log_code(root=".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
        print("WandB initialized and code logged")
    else:
        print("WandB disabled")

    # Set device
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = PointCloud2DDataset(
        shape_type=config["data"]["shape_type"],
        num_samples=config["data"]["num_samples"],
        num_points_range=tuple(config["data"]["num_points_range"]),
        noise=config["data"]["noise"],
        scale=config["data"]["scale"]
    )
    print(f"Dataset: {config['data']['shape_type']}, {len(dataset)} samples")
    print(f"Points per sample: {config['data']['num_points_range'][0]}-{config['data']['num_points_range'][1]}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_pointcloud2d,
        num_workers=config["data"]["num_workers"]
    )

    # Create model
    print("\nCreating model...")
    arch_type = config["architecture"]["type"]

    if arch_type == "mlp":
        backbone = MLP(
            input_dim=2,
            hidden_dim=config["architecture"]["hidden_dim"],
            num_layers=config["architecture"]["num_layers"],
            dropout=config["architecture"]["dropout"],
            activation=config["architecture"]["activation"]
        )
    elif arch_type == "gnn":
        backbone = GNN(
            hidden_dim=config["architecture"]["hidden_dim"],
            num_layers=config["architecture"]["num_layers"],
            dropout=config["architecture"]["dropout"],
            activation=config["architecture"].get("activation", "silu")
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")

    model = FlowMatching2D(
        backbone=backbone,
        time_steps=config["generative"]["time_steps"]
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {arch_type}, {num_params:,} parameters")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Training loop
    print("\nStarting training...")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = batch.to(device)

            # For flow matching with 2D point clouds
            # FlowMatching2D just needs positions and batch indices
            positions = batch.positions
            batch_idx = batch.batch_idx

            # Forward pass
            loss, info = model(positions, batch_idx)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config["training"]["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["gradient_clip"]
                )

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")

        # Log metrics to WandB
        if wandb.run is not None:
            wandb.log({"epoch": epoch, "train/loss": avg_loss})

        # Visualize every N epochs
        if epoch % config["training"]["visualize_every"] == 0 or epoch == config["training"]["epochs"] - 1:
            print(f"Generating visualization for epoch {epoch}...")

            # Get some real samples
            real_samples = [dataset[i]["positions"] for i in range(9)]

            # Generate samples
            generated_samples = generate_samples(
                model,
                num_samples=9,
                num_points_range=tuple(config["data"]["num_points_range"]),
                device=device
            )

            # Visualize
            visualize_point_clouds(real_samples, generated_samples, epoch, output_dir)

            # Log visualization to WandB
            if wandb.run is not None:
                img_path = f"{output_dir}/epoch_{epoch:04d}.png"
                wandb.log({"visualizations/point_clouds": wandb.Image(img_path)})

    # Save model
    model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")

    # Log model to WandB
    if wandb.run is not None:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        print("Model logged to WandB")

    print("\nTraining complete!")
    print(f"Visualizations saved to {output_dir}")

    # Finish WandB run
    if wandb.run is not None:
        wandb.finish()
        print("WandB run finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D point cloud generative model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pointcloud2d.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
