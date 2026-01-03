"""Training script for 2D toy datasets with visualization."""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb

from src.data import Toy2DDataset, collate_toy2d
from src.models.architectures import MLP
from src.models.generative import FlowMatching


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def visualize_training(
    model,
    data,
    device,
    epoch,
    output_dir,
    num_samples=1000
):
    """Visualize the model's learned distribution.

    Args:
        model: Trained generative model
        data: Real data tensor of shape (N, 2)
        device: Device to run on
        epoch: Current epoch number
        output_dir: Directory to save plots
        num_samples: Number of samples to generate
    """
    model.eval()

    with torch.no_grad():
        # Generate samples from noise
        noise = torch.randn(num_samples, 2).to(device)
        generated = model.sample(noise, num_steps=100)

    # Convert to numpy
    data_np = data.cpu().numpy()
    generated_np = generated.cpu().numpy()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot real data
    axes[0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.3, s=1)
    axes[0].set_title("Real Data")
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Plot generated data
    axes[1].scatter(generated_np[:, 0], generated_np[:, 1], alpha=0.3, s=1, c='red')
    axes[1].set_title(f"Generated Data (Epoch {epoch})")
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    # Plot both overlaid
    axes[2].scatter(data_np[:, 0], data_np[:, 1], alpha=0.2, s=1, label='Real', c='blue')
    axes[2].scatter(generated_np[:, 0], generated_np[:, 1], alpha=0.2, s=1, label='Generated', c='red')
    axes[2].set_title("Overlay")
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()

    model.train()


def visualize_flow_field(
    model,
    device,
    epoch,
    output_dir,
    grid_size=20,
    t=0.5
):
    """Visualize the vector field at a specific time.

    Args:
        model: Flow matching model
        device: Device to run on
        epoch: Current epoch number
        output_dir: Directory to save plots
        grid_size: Number of grid points along each dimension
        t: Time point to visualize (0 to 1)
    """
    model.eval()

    # Create grid
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)

    # Convert to tensor
    positions_tensor = torch.from_numpy(positions).float().to(device)
    t_tensor = torch.ones(len(positions), 1).to(device) * t

    with torch.no_grad():
        # Get velocity field
        velocities = model.backbone(positions_tensor)
        velocities_np = velocities.cpu().numpy()

    # Reshape for quiver plot
    U = velocities_np[:, 0].reshape(grid_size, grid_size)
    V = velocities_np[:, 1].reshape(grid_size, grid_size)

    # Plot vector field
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, alpha=0.6)
    plt.title(f"Flow Field at t={t:.2f} (Epoch {epoch})")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(f"{output_dir}/flow_field_epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()

    model.train()


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
    dataset = Toy2DDataset(
        dataset_type=config["data"]["dataset_type"],
        num_samples=config["data"]["num_samples"],
        noise=config["data"]["noise"]
    )
    print(f"Dataset: {config['data']['dataset_type']}, {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_toy2d
    )

    # Create model
    print("\nCreating model...")
    backbone = MLP(
        input_dim=2,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        activation=config["model"]["activation"]
    )

    model = FlowMatching(
        backbone=backbone,
        time_steps=config["model"]["time_steps"]
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0)
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

            # Forward pass (Flow Matching for 2D)
            # Hack: create dummy edge_index for flow matching
            # FlowMatching expects (positions, atom_types, edge_index)
            # For 2D toy data, we just pass positions as both positions and features
            loss, info = model(batch, batch, edge_index=None)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
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
        if epoch % config["training"]["visualize_every"] == 0:
            print(f"Generating visualization for epoch {epoch}...")
            all_data = dataset.get_all_data().to(device)
            visualize_training(model, all_data, device, epoch, output_dir)

            # Log visualization to WandB
            if wandb.run is not None:
                img_path = f"{output_dir}/epoch_{epoch:04d}.png"
                wandb.log({"visualizations/samples": wandb.Image(img_path)})

            # Also visualize flow field
            if config["training"].get("visualize_flow_field", True):
                visualize_flow_field(model, device, epoch, output_dir)

                # Log flow field to WandB
                if wandb.run is not None:
                    flow_img_path = f"{output_dir}/flow_field_epoch_{epoch:04d}.png"
                    wandb.log({"visualizations/flow_field": wandb.Image(flow_img_path)})

    # Final visualization
    print("\nGenerating final visualization...")
    all_data = dataset.get_all_data().to(device)
    visualize_training(model, all_data, device, config["training"]["epochs"], output_dir)

    # Log final visualization to WandB
    if wandb.run is not None:
        final_img_path = f"{output_dir}/epoch_{config['training']['epochs']:04d}.png"
        wandb.log({"visualizations/final": wandb.Image(final_img_path)})

    # Save model
    model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

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
    parser = argparse.ArgumentParser(description="Train 2D toy generative model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy2d.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
