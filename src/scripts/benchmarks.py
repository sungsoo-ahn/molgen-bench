"""Main benchmark script for training and evaluating molecular generative models."""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.data import QM9Dataset, MP20Dataset, collate_molecular_data
from src.models.architectures import DiT
from src.models.generative import FlowMatching, Diffusion, StochasticInterpolant
from src.training import Trainer
from src.evaluation import compute_all_metrics, print_metrics_summary


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config: dict, split: str):
    """Create dataset from config."""
    dataset_name = config["data"]["dataset"]
    data_dir = config["data"]["data_dir"]

    if dataset_name == "qm9":
        dataset = QM9Dataset(data_dir=data_dir, split=split, download=True)
    elif dataset_name == "mp20":
        dataset = MP20Dataset(data_dir=data_dir, split=split, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def create_architecture(config: dict):
    """Create backbone architecture from config."""
    arch_config = config["architecture"]
    arch_type = arch_config["type"]

    if arch_type == "dit":
        model = DiT(
            hidden_dim=arch_config.get("hidden_dim", 256),
            num_layers=arch_config.get("num_layers", 6),
            num_heads=arch_config.get("num_heads", 4),
            mlp_ratio=arch_config.get("mlp_ratio", 4.0),
            num_atom_types=arch_config.get("num_atom_types", 100),
            dropout=arch_config.get("dropout", 0.1),
            max_atoms=arch_config.get("max_atoms", 256),
            use_checkpoint=arch_config.get("use_checkpoint", False),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_type}. Only 'dit' is supported.")

    return model


def create_generative_model(backbone, config: dict):
    """Create generative model from config."""
    gen_config = config["generative"]
    gen_type = gen_config["type"]

    if gen_type == "flow_matching":
        model = FlowMatching(
            backbone=backbone,
            time_steps=gen_config.get("time_steps", 1000),
            sigma_min=gen_config.get("sigma_min", 0.001),
            sigma_max=gen_config.get("sigma_max", 1.0),
            schedule=gen_config.get("noise_schedule", "linear"),
            loss_type=gen_config.get("loss_type", "mse"),
        )
    elif gen_type == "diffusion":
        model = Diffusion(
            backbone=backbone,
            time_steps=gen_config.get("time_steps", 1000),
            beta_schedule=gen_config.get("beta_schedule", "cosine"),
        )
    elif gen_type == "stochastic_interpolant":
        model = StochasticInterpolant(
            backbone=backbone,
            time_steps=gen_config.get("time_steps", 1000),
            interpolant_type=gen_config.get("interpolant_type", "polynomial"),
        )
    else:
        raise ValueError(f"Unknown generative model: {gen_type}")

    return model


def main(args):
    """Main training and evaluation loop."""
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_dataset(config, "train")
    val_dataset = create_dataset(config, "val")
    test_dataset = create_dataset(config, "test")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_molecular_data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_molecular_data
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_molecular_data
    )

    # Create models
    print("\nCreating models...")
    backbone = create_architecture(config)
    model = create_generative_model(backbone, config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["optimizer"]["betas"]
    )

    # Create scheduler
    if config["scheduler"]["type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"] * len(train_loader),
            eta_min=config["scheduler"]["min_lr"]
        )
    else:
        scheduler = None

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=config.get("mixed_precision", True),
        gradient_clip=config["training"]["gradient_clip"],
        log_interval=config["training"]["log_interval"],
        eval_interval=config["training"]["eval_interval"],
        checkpoint_dir=f"{config['logging']['output_dir']}/{config['logging']['experiment_name']}",
        track_scaling=config["training"]["track_scaling"]
    )

    # Train
    if not args.eval_only:
        print("\nStarting training...")
        trainer.train(num_epochs=config["training"]["epochs"])

    # Evaluate
    if args.evaluate:
        print("\nEvaluating on test set...")
        # TODO: Generate samples and compute metrics
        # For now, just evaluate loss
        test_metrics = trainer.evaluate()
        print(f"Test loss: {test_metrics['loss']:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train molecular generative models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Evaluate after training"
    )

    args = parser.parse_args()
    main(args)
