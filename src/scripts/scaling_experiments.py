"""Scaling law experiments.

Run experiments varying:
- Model size (number of layers, hidden dimensions)
- Dataset size (subsample training data)
- Training compute (different numbers of epochs)
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path

from src.data import QM9Dataset, collate_molecular_data
from src.models.architectures import GNN
from src.models.generative import FlowMatching
from src.training import Trainer


def run_scaling_experiment(
    base_config: dict,
    model_sizes: list,
    dataset_fractions: list,
    device: str = "cuda"
):
    """Run scaling experiments.

    Args:
        base_config: Base configuration dict
        model_sizes: List of (hidden_dim, num_layers) tuples
        dataset_fractions: List of dataset fractions to use
        device: Device to use
    """
    # Load full dataset
    train_dataset = QM9Dataset(
        data_dir=base_config["data"]["data_dir"],
        split="train",
        download=True
    )

    val_dataset = QM9Dataset(
        data_dir=base_config["data"]["data_dir"],
        split="val",
        download=True
    )

    results = []

    # Experiment 1: Vary model size
    print("\n" + "="*60)
    print("Experiment 1: Varying Model Size")
    print("="*60)

    for hidden_dim, num_layers in model_sizes:
        print(f"\nTraining with hidden_dim={hidden_dim}, num_layers={num_layers}")

        # Create model
        backbone = GNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1
        )
        model = FlowMatching(backbone=backbone)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model size: {num_params:,} parameters")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=base_config["data"]["batch_size"],
            shuffle=True,
            collate_fn=collate_molecular_data
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=base_config["data"]["batch_size"],
            shuffle=False,
            collate_fn=collate_molecular_data
        )

        # Train
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            log_interval=100,
            eval_interval=500,
            checkpoint_dir=f"outputs/scaling_model_{hidden_dim}_{num_layers}",
            track_scaling=True
        )

        trainer.train(num_epochs=5)  # Short training for scaling experiments

        # Record results
        results.append({
            "experiment": "model_size",
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_params": num_params,
            "final_val_loss": trainer.best_val_loss
        })

    # Experiment 2: Vary dataset size
    print("\n" + "="*60)
    print("Experiment 2: Varying Dataset Size")
    print("="*60)

    for fraction in dataset_fractions:
        print(f"\nTraining with {fraction*100:.0f}% of dataset")

        # Subsample dataset
        n_samples = int(len(train_dataset) * fraction)
        indices = np.random.choice(len(train_dataset), n_samples, replace=False)
        subset = Subset(train_dataset, indices)

        print(f"Dataset size: {len(subset):,} samples")

        # Create model (fixed size)
        backbone = GNN(hidden_dim=256, num_layers=6, dropout=0.1)
        model = FlowMatching(backbone=backbone)

        # Create data loaders
        train_loader = DataLoader(
            subset,
            batch_size=base_config["data"]["batch_size"],
            shuffle=True,
            collate_fn=collate_molecular_data
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=base_config["data"]["batch_size"],
            shuffle=False,
            collate_fn=collate_molecular_data
        )

        # Train
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            log_interval=100,
            eval_interval=500,
            checkpoint_dir=f"outputs/scaling_data_{fraction}",
            track_scaling=True
        )

        trainer.train(num_epochs=10)

        # Record results
        results.append({
            "experiment": "dataset_size",
            "fraction": fraction,
            "dataset_size": len(subset),
            "final_val_loss": trainer.best_val_loss
        })

    # Print summary
    print("\n" + "="*60)
    print("Scaling Experiments Summary")
    print("="*60)

    print("\nModel Size Scaling:")
    for r in results:
        if r["experiment"] == "model_size":
            print(f"  {r['num_params']:>10,} params: val_loss = {r['final_val_loss']:.4f}")

    print("\nDataset Size Scaling:")
    for r in results:
        if r["experiment"] == "dataset_size":
            print(f"  {r['dataset_size']:>7,} samples: val_loss = {r['final_val_loss']:.4f}")


def main(args):
    """Main function."""
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define scaling experiment parameters
    model_sizes = [
        (128, 3),   # Small
        (256, 6),   # Medium
        (512, 12),  # Large
    ]

    dataset_fractions = [0.1, 0.3, 0.5, 0.7, 1.0]

    # Run experiments
    run_scaling_experiment(
        base_config=config,
        model_sizes=model_sizes,
        dataset_fractions=dataset_fractions,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scaling law experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
