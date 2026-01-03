"""Training script for 3D toy molecular dataset.

This script trains generative models on 3D geometric shapes with molecular features
(positions + atom types) for quick testing and validation before using real datasets.
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb

from src.data import ToyMolecular3DDataset, collate_molecular_data
from src.data.utils import build_fully_connected_edges
from src.models.architectures import GNN
from src.models.generative import FlowMatching


# Atom type mapping: [1, 6, 7, 8, 9] -> [0, 1, 2, 3, 4] (same as QM9)
ATOM_TYPES = [1, 6, 7, 8, 9]
ATOM_TYPE_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
ATOM_COLORS = {0: 'white', 1: 'gray', 2: 'blue', 3: 'red', 4: 'green'}  # H, C, N, O, F


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def visualize_molecules_3d(
    real_samples,
    generated_samples,
    epoch,
    output_dir,
    num_vis=6
):
    """Visualize real vs generated 3D molecules.

    Args:
        real_samples: List of (positions, atom_types) tuples
        generated_samples: List of (positions, atom_types) tuples
        epoch: Current epoch number
        output_dir: Directory to save plots
        num_vis: Number of samples to visualize
    """
    num_vis = min(num_vis, len(real_samples), len(generated_samples))
    fig = plt.figure(figsize=(4 * num_vis, 8))

    for i in range(num_vis):
        # Real molecule
        ax1 = fig.add_subplot(2, num_vis, i + 1, projection='3d')
        real_pos, real_types = real_samples[i]
        colors = [ATOM_COLORS.get(t, 'black') for t in real_types]
        ax1.scatter(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2],
                    c=colors, s=50, edgecolors='black', alpha=0.8)
        ax1.set_title(f'Real {i+1}')
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_zlim(-1.5, 1.5)

        # Generated molecule
        ax2 = fig.add_subplot(2, num_vis, num_vis + i + 1, projection='3d')
        gen_pos, gen_types = generated_samples[i]
        colors = [ATOM_COLORS.get(t, 'black') for t in gen_types]
        ax2.scatter(gen_pos[:, 0], gen_pos[:, 1], gen_pos[:, 2],
                    c=colors, s=50, edgecolors='black', alpha=0.8)
        ax2.set_title(f'Generated {i+1}')
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_zlim(-1.5, 1.5)

    plt.suptitle(f'Epoch {epoch}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_samples(model, num_samples, num_atoms_range, device):
    """Generate molecular samples from the model.

    Args:
        model: Trained generative model
        num_samples: Number of samples to generate
        num_atoms_range: (min, max) number of atoms per sample
        device: Device to run on

    Returns:
        List of (positions, atom_types) tuples
    """
    model.eval()
    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Random number of atoms
            num_atoms = np.random.randint(num_atoms_range[0], num_atoms_range[1] + 1)

            # Random atom types
            atom_probs = [0.5, 0.3, 0.1, 0.08, 0.02]
            atom_types_raw = np.random.choice(ATOM_TYPES, size=num_atoms, p=atom_probs)
            atom_types = torch.tensor([ATOM_TYPE_MAP[a] for a in atom_types_raw],
                                       dtype=torch.long, device=device)

            # Build edges for this molecule
            batch_idx = torch.zeros(num_atoms, dtype=torch.long, device=device)
            num_atoms_tensor = torch.tensor([num_atoms], dtype=torch.long, device=device)
            edge_index = build_fully_connected_edges(batch_idx, num_atoms_tensor)

            # Generate through flow integration (model starts from noise internally)
            positions = model.sample(
                atom_types=atom_types,
                edge_index=edge_index,
                batch_idx=batch_idx,
                num_steps=100
            )

            samples.append((
                positions.cpu().numpy(),
                atom_types.cpu().numpy()
            ))

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
    (output_dir / "figures").mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = ToyMolecular3DDataset(
        shape_type=config["data"]["shape_type"],
        num_samples=config["data"]["num_samples"],
        num_atoms_range=tuple(config["data"]["num_atoms_range"]),
        noise=config["data"]["noise"],
        scale=config["data"]["scale"]
    )
    print(f"Dataset: {config['data']['shape_type']}, {len(dataset)} samples")
    print(f"Atoms per sample: {config['data']['num_atoms_range'][0]}-{config['data']['num_atoms_range'][1]}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_molecular_data,
        num_workers=config["data"]["num_workers"]
    )

    # Create model
    print("\nCreating model...")
    num_atom_types = 5  # Same as QM9: H, C, N, O, F

    backbone = GNN(
        num_atom_types=num_atom_types,
        hidden_dim=config["architecture"]["hidden_dim"],
        num_layers=config["architecture"]["num_layers"],
        dropout=config["architecture"]["dropout"],
        activation=config["architecture"].get("activation", "silu"),
        readout=config["architecture"].get("readout", "sum")
    )

    model = FlowMatching(
        backbone=backbone,
        time_steps=config["generative"].get("time_steps", 100),
        sigma_min=config["generative"].get("sigma_min", 0.001),
        sigma_max=config["generative"].get("sigma_max", 1.0),
        schedule=config["generative"].get("schedule", "cosine"),
        loss_type=config["generative"].get("loss_type", "mse")
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: GNN + FlowMatching, {num_params:,} parameters")

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
            # Move to device
            positions = batch.positions.to(device)
            atom_types = batch.atom_types.to(device)
            batch_idx = batch.batch_idx.to(device)

            # Map atom types to indices 0-4
            mapped_atom_types = torch.zeros_like(atom_types)
            for orig, new in ATOM_TYPE_MAP.items():
                mapped_atom_types[atom_types == orig] = new

            # Build fully connected edges
            num_atoms = torch.bincount(batch_idx)
            edge_index = build_fully_connected_edges(batch_idx, num_atoms)

            # Forward pass
            loss, info = model(positions, mapped_atom_types, edge_index, batch_idx)

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
        visualize_every = config["training"].get("visualize_every", 10)
        if epoch % visualize_every == 0 or epoch == config["training"]["epochs"] - 1:
            print(f"Generating visualization for epoch {epoch}...")

            # Get some real samples
            real_samples = []
            for i in range(6):
                sample = dataset[i]
                mapped_types = np.array([ATOM_TYPE_MAP[a.item()] for a in sample["atom_types"]])
                real_samples.append((sample["positions"].numpy(), mapped_types))

            # Generate samples
            generated_samples = generate_samples(
                model,
                num_samples=6,
                num_atoms_range=tuple(config["data"]["num_atoms_range"]),
                device=device
            )

            # Visualize
            visualize_molecules_3d(
                real_samples,
                generated_samples,
                epoch,
                output_dir / "figures"
            )

            # Log visualization to WandB
            if wandb.run is not None:
                img_path = f"{output_dir}/figures/epoch_{epoch:04d}.png"
                wandb.log({"visualizations/molecules": wandb.Image(img_path)})

    # Save model
    model_path = output_dir / "final_model.pt"
    torch.save({
        'epoch': config["training"]["epochs"] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_loss,
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # Log model to WandB
    if wandb.run is not None:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        print("Model logged to WandB")

    print("\nTraining complete!")
    print(f"Visualizations saved to {output_dir}/figures")

    # Finish WandB run
    if wandb.run is not None:
        wandb.finish()
        print("WandB run finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D toy molecular generative model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_molecular.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
