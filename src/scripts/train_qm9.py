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
from src.models.architectures import DiT
from src.models.generative import FlowMatching
from src.evaluation.sample_quality import compute_atom_stability, compute_molecule_stability


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def get_lr_with_warmup(step: int, warmup_steps: int, base_lr: float) -> float:
    """Calculate learning rate with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


@torch.no_grad()
def generate_molecules(
    model,
    num_samples: int,
    dataset: QM9Dataset,
    device: torch.device,
    num_steps: int = 100
):
    """Generate molecules by sampling from the model.

    Uses atom type distributions and molecule sizes from the dataset.

    Args:
        model: FlowMatching model
        num_samples: Number of molecules to generate
        dataset: QM9Dataset to sample atom compositions from
        device: Device to run generation on
        num_steps: Number of ODE integration steps

    Returns:
        List of dicts with 'positions' and 'atom_types'
    """
    model.eval()
    generated = []

    # Sample molecule compositions from the dataset
    indices = np.random.choice(len(dataset), num_samples, replace=True)

    for idx in tqdm(indices, desc="Generating molecules"):
        # Get atom types from a real molecule
        mol_data = dataset[idx]
        atom_types = mol_data["atom_types"].to(device)

        # Map to indices 0-4 if needed (QM9 uses atomic numbers)
        atom_type_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        mapped_types = torch.zeros_like(atom_types)
        for orig, new in atom_type_map.items():
            mapped_types[atom_types == orig] = new

        # Generate positions
        batch_idx = torch.zeros(len(mapped_types), dtype=torch.long, device=device)
        positions = model.sample(
            atom_types=mapped_types,
            batch_idx=batch_idx,
            num_steps=num_steps
        )

        generated.append({
            "positions": positions.cpu(),
            "atom_types": mapped_types.cpu()  # Use indices for stability check
        })

    return generated


def evaluate_generation(
    model,
    dataset: QM9Dataset,
    device: torch.device,
    num_samples: int = 100,
    num_steps: int = 100
):
    """Generate molecules and compute stability metrics.

    Args:
        model: FlowMatching model
        dataset: QM9Dataset
        device: Device
        num_samples: Number of molecules to generate
        num_steps: Number of ODE integration steps

    Returns:
        Dict with metrics
    """
    generated = generate_molecules(model, num_samples, dataset, device, num_steps)

    # Compute stability metrics (use_atomic_numbers=False since we use indices)
    atom_stab = compute_atom_stability(generated, use_atomic_numbers=False)
    mol_stab = compute_molecule_stability(generated, use_atomic_numbers=False)

    return {
        "atom_stability": atom_stab["atom_stability"],
        "molecule_stability": mol_stab["molecule_stability"],
        "num_generated": num_samples
    }


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
    num_atom_types = 5  # H, C, N, O, F

    backbone = DiT(
        num_atom_types=num_atom_types,
        hidden_dim=config["architecture"]["hidden_dim"],
        num_layers=config["architecture"]["num_layers"],
        num_heads=config["architecture"].get("num_heads", 8),
        mlp_ratio=config["architecture"].get("mlp_ratio", 4.0),
        dropout=config["architecture"]["dropout"],
        max_atoms=config["architecture"].get("max_atoms", 64),
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
    print(f"Model: DiT + FlowMatching, {num_params:,} parameters")

    # Create optimizer
    base_lr = config["training"]["learning_rate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999)
    )

    # Initialize EMA
    use_ema = config["training"].get("use_ema", True)
    ema_decay = config["training"].get("ema_decay", 0.9999)
    ema = EMA(model, decay=ema_decay) if use_ema else None
    if use_ema:
        print(f"EMA enabled with decay={ema_decay}")

    # Warmup settings
    warmup_steps = config["training"].get("warmup_steps", 0)
    global_step = 0

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

            # Forward pass (DiT doesn't need edge_index)
            loss, info = model(positions, mapped_atom_types, batch_idx=batch_idx)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config["training"]["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["gradient_clip"]
                )

            # Apply warmup learning rate
            if warmup_steps > 0:
                lr = get_lr_with_warmup(global_step, warmup_steps, base_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            optimizer.step()

            # Update EMA
            if ema is not None:
                ema.update(model)

            global_step += 1
            total_train_loss += loss.item()
            num_train_batches += 1

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        avg_train_loss = total_train_loss / num_train_batches

        # Validation (use EMA weights if available)
        if ema is not None:
            ema.apply_shadow(model)

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

                loss, info = model(positions, mapped_atom_types, batch_idx=batch_idx)

                total_val_loss += loss.item()
                num_val_batches += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Restore original weights after validation
        if ema is not None:
            ema.restore(model)

        avg_val_loss = total_val_loss / num_val_batches

        print(f"Epoch {epoch}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}")

        # Evaluate generation quality periodically
        eval_every = config["training"].get("eval_every", 10)
        if epoch % eval_every == 0 or epoch == config["training"]["epochs"] - 1:
            print(f"Evaluating generation quality...")
            # Use EMA weights for evaluation
            if ema is not None:
                ema.apply_shadow(model)

            gen_metrics = evaluate_generation(
                model,
                val_dataset,
                device,
                num_samples=config["training"].get("eval_samples", 100),
                num_steps=config["training"].get("eval_steps", 100)
            )

            if ema is not None:
                ema.restore(model)

            print(f"  Atom stability: {gen_metrics['atom_stability']:.3f}")
            print(f"  Molecule stability: {gen_metrics['molecule_stability']:.3f}")

            # Log generation metrics to WandB
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "gen/atom_stability": gen_metrics["atom_stability"],
                    "gen/molecule_stability": gen_metrics["molecule_stability"],
                })

        # Log metrics to WandB
        if wandb.run is not None:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "train/lr": current_lr,
                "train/global_step": global_step
            })

        # Save best model (save EMA weights)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # Save EMA weights as best model
            if ema is not None:
                ema.apply_shadow(model)

            model_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_path)
            print(f"Saved best model to {model_path}")

            if ema is not None:
                ema.restore(model)

    # Save final model
    model_path = output_dir / "final_model.pt"
    save_dict = {
        'epoch': config["training"]["epochs"] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'global_step': global_step,
    }
    if ema is not None:
        save_dict['ema_shadow'] = ema.shadow
    torch.save(save_dict, model_path)
    print(f"\nSaved final model to {model_path}")

    # Save EMA model separately
    if ema is not None:
        ema.apply_shadow(model)
        ema_model_path = output_dir / "final_model_ema.pt"
        torch.save({
            'epoch': config["training"]["epochs"] - 1,
            'model_state_dict': model.state_dict(),
            'val_loss': avg_val_loss,
        }, ema_model_path)
        ema.restore(model)
        print(f"Saved EMA model to {ema_model_path}")

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
        default="configs/qm9.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
