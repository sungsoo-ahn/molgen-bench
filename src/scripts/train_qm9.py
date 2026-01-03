"""Training script for QM9 molecular dataset.

This script trains generative models on the QM9 dataset of small organic molecules.
"""

import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb

from src.data import QM9Dataset, collate_molecular_data
from src.data.utils import build_fully_connected_edges
from src.models.architectures import GNN
from src.models.generative import FlowMatching


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


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

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = QM9Dataset(
        data_dir=config["data"]["data_dir"],
        split="train",
        download=True
    )
    val_dataset = QM9Dataset(
        data_dir=config["data"]["data_dir"],
        split="val",
        download=True
    )
    print(f"Train dataset: {len(train_dataset)} molecules")
    print(f"Val dataset: {len(val_dataset)} molecules")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_molecular_data,
        num_workers=config["data"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_molecular_data,
        num_workers=config["data"]["num_workers"]
    )

    # Create model
    print("\nCreating model...")
    # For QM9, we have 5 atom types: H(1), C(6), N(7), O(8), F(9)
    # Map them to indices 0-4
    num_atom_types = 5

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
        time_steps=config["generative"].get("time_steps", 1000),
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

    # Create learning rate scheduler if specified
    scheduler = None
    if config["training"].get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

    # Atom type mapping for QM9: [1, 6, 7, 8, 9] -> [0, 1, 2, 3, 4]
    atom_type_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(config["training"]["epochs"]):
        # Training
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            # Move to device
            positions = batch.positions.to(device)
            atom_types = batch.atom_types.to(device)
            batch_idx = batch.batch_idx.to(device)

            # Map atom types to indices 0-4
            mapped_atom_types = torch.zeros_like(atom_types)
            for orig, new in atom_type_map.items():
                mapped_atom_types[atom_types == orig] = new

            # Build fully connected edges for each molecule
            # Count atoms per molecule
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

            total_train_loss += loss.item()
            num_train_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / num_train_batches

        # Validation
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [val]")
            for batch in pbar:
                positions = batch.positions.to(device)
                atom_types = batch.atom_types.to(device)
                batch_idx = batch.batch_idx.to(device)

                # Map atom types
                mapped_atom_types = torch.zeros_like(atom_types)
                for orig, new in atom_type_map.items():
                    mapped_atom_types[atom_types == orig] = new

                # Build edges
                num_atoms = torch.bincount(batch_idx)
                edge_index = build_fully_connected_edges(batch_idx, num_atoms)

                loss, info = model(positions, mapped_atom_types, edge_index, batch_idx)

                total_val_loss += loss.item()
                num_val_batches += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / num_val_batches

        print(f"Epoch {epoch}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}")

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Log metrics to WandB
        if wandb.run is not None:
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
            }
            if scheduler is not None:
                log_dict["train/lr"] = optimizer.param_groups[0]['lr']
            wandb.log(log_dict)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_path)
            print(f"Saved best model to {model_path}")

    # Save final model
    model_path = output_dir / "final_model.pt"
    torch.save({
        'epoch': config["training"]["epochs"] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, model_path)
    print(f"\nSaved final model to {model_path}")

    # Log model to WandB
    if wandb.run is not None:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(str(output_dir / "best_model.pt"))
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        print("Models logged to WandB")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Finish WandB run
    if wandb.run is not None:
        wandb.finish()
        print("WandB run finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QM9 generative model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qm9_quick.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
