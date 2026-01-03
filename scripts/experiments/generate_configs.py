#!/usr/bin/env python3
"""Generate experiment configurations for overnight runs.

This script creates various experiment configs comparing:
- Model sizes (small, medium, large)
- Optimizers (adamw, muon)
- Mixed precision (on, off)
- Learning rates
- Schedules
"""

import yaml
from pathlib import Path
from itertools import product

# Base configuration
BASE_CONFIG = {
    "wandb": {
        "enabled": True,
        "project": "molgen-bench",
        "entity": None,
    },
    "data": {
        "dataset_type": "qm9",
        "data_dir": "./data/downloaded/qm9",
        "batch_size": 64,
        "num_workers": 4,
    },
    "generative": {
        "type": "flow_matching",
        "time_steps": 1000,
        "sigma_min": 0.001,
        "sigma_max": 1.0,
        "schedule": "cosine",
        "loss_type": "mse",
    },
    "training": {
        "epochs": 100,
        "weight_decay": 0.00001,
        "gradient_clip": 1.0,
        "warmup_steps": 1000,
        "ema_decay": 0.9999,
        "use_ema": True,
        "eval_every": 10,
        "eval_samples": 100,
        "eval_steps": 100,
        "log_interval": 100,
        "log_detailed_metrics": True,
    },
    "device": "cuda",
}

# Experiment dimensions
MODEL_SIZES = {
    "small": {"hidden_dim": 128, "num_layers": 4, "num_heads": 4, "mlp_ratio": 4.0},
    "medium": {"hidden_dim": 256, "num_layers": 6, "num_heads": 8, "mlp_ratio": 4.0},
    "large": {"hidden_dim": 384, "num_layers": 8, "num_heads": 12, "mlp_ratio": 4.0},
}

OPTIMIZERS = {
    "adamw": {
        "optimizer": "adamw",
        "learning_rate": 3e-4,
        "muon_lr": 0.02,  # Not used but kept for consistency
        "muon_momentum": 0.95,
    },
    "muon": {
        "optimizer": "muon",
        "learning_rate": 3e-4,  # For AdamW part of Muon
        "muon_lr": 0.02,
        "muon_momentum": 0.95,
    },
}

PRECISION = {
    "fp32": {"mixed_precision": False},
    "fp16": {"mixed_precision": True},
}

# Learning rate variants for sensitivity analysis
LR_VARIANTS = {
    "lr_low": {"learning_rate": 1e-4, "muon_lr": 0.01},
    "lr_default": {"learning_rate": 3e-4, "muon_lr": 0.02},
    "lr_high": {"learning_rate": 1e-3, "muon_lr": 0.05},
}

# Schedule variants
SCHEDULE_VARIANTS = {
    "linear": {"schedule": "linear"},
    "cosine": {"schedule": "cosine"},
    "sigmoid": {"schedule": "sigmoid"},
    "quadratic": {"schedule": "quadratic"},
}

# Positional encoding variants
POS_ENCODING_VARIANTS = {
    "learnable": {"pos_encoding": "learnable"},
    "sinusoidal": {"pos_encoding": "sinusoidal"},
    "none": {"pos_encoding": "none"},
}

# Coordinate encoding variants
COORD_ENCODING_VARIANTS = {
    "linear": {"coord_encoding": "linear"},
    "fourier": {"coord_encoding": "fourier"},
}

# Normalization variants
NORM_VARIANTS = {
    "layernorm": {"norm_type": "layernorm"},
    "rmsnorm": {"norm_type": "rmsnorm"},
}

# Loss type variants
LOSS_VARIANTS = {
    "mse": {"loss_type": "mse"},
    "huber": {"loss_type": "huber"},
}


def create_config(
    name: str,
    model_size: str,
    optimizer: str,
    precision: str,
    lr_variant: str = "lr_default",
    schedule: str = "cosine",
    pos_encoding: str = "learnable",
    coord_encoding: str = "linear",
    norm_type: str = "layernorm",
    loss_type: str = "mse",
    group: str = "overnight_v1",
) -> dict:
    """Create a single experiment configuration."""
    config = yaml.safe_load(yaml.dump(BASE_CONFIG))  # Deep copy

    # Set output directory and WandB name
    config["output_dir"] = f"data/experiments/{group}/{name}"
    config["wandb"]["name"] = name
    config["wandb"]["group"] = group  # Group related experiments together
    config["wandb"]["job_type"] = "train"
    config["wandb"]["tags"] = [
        "qm9", "flow_matching", "dit", "experiment",
        f"size_{model_size}", f"opt_{optimizer}", f"prec_{precision}",
        group
    ]

    # Model architecture
    config["architecture"] = {
        "type": "dit",
        "dropout": 0.0,
        "max_atoms": 64,
        "pos_encoding": pos_encoding,
        "coord_encoding": coord_encoding,
        "norm_type": norm_type,
        **MODEL_SIZES[model_size],
    }

    # Generative settings
    config["generative"]["schedule"] = schedule
    config["generative"]["loss_type"] = loss_type

    # Training settings
    config["training"].update(OPTIMIZERS[optimizer])
    config["training"].update(LR_VARIANTS[lr_variant])

    # Precision
    config["mixed_precision"] = PRECISION[precision]["mixed_precision"]

    # Adjust batch size for larger models
    if model_size == "large":
        config["data"]["batch_size"] = 32
    elif model_size == "medium":
        config["data"]["batch_size"] = 64
    else:
        config["data"]["batch_size"] = 128

    return config


def generate_main_experiments(output_dir: Path, group: str = "overnight_v1"):
    """Generate main experiment grid: model_size x optimizer x precision."""
    experiments = []

    for model_size, optimizer, precision in product(
        ["small", "medium", "large"],
        ["adamw", "muon"],
        ["fp32", "fp16"]
    ):
        name = f"{model_size}_{optimizer}_{precision}"
        config = create_config(
            name=name,
            model_size=model_size,
            optimizer=optimizer,
            precision=precision,
            group=group,
        )

        config_path = output_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        experiments.append({
            "name": name,
            "config": str(config_path),
            "model_size": model_size,
            "optimizer": optimizer,
            "precision": precision,
        })

        print(f"Created: {config_path}")

    return experiments


def generate_lr_ablation(output_dir: Path, group: str = "overnight_v1"):
    """Generate learning rate ablation experiments."""
    experiments = []

    # Test different LRs with medium model and both optimizers
    for optimizer, lr_variant in product(["adamw", "muon"], ["lr_low", "lr_default", "lr_high"]):
        name = f"medium_{optimizer}_{lr_variant}"
        config = create_config(
            name=name,
            model_size="medium",
            optimizer=optimizer,
            precision="fp32",
            lr_variant=lr_variant,
            group=group,
        )

        config_path = output_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        experiments.append({
            "name": name,
            "config": str(config_path),
        })

        print(f"Created: {config_path}")

    return experiments


def generate_schedule_ablation(output_dir: Path, group: str = "overnight_v1"):
    """Generate schedule ablation experiments."""
    experiments = []

    for schedule in ["linear", "cosine", "sigmoid", "quadratic"]:
        name = f"medium_adamw_sched_{schedule}"
        config = create_config(
            name=name,
            model_size="medium",
            optimizer="adamw",
            precision="fp32",
            schedule=schedule,
            group=group,
        )

        config_path = output_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        experiments.append({
            "name": name,
            "config": str(config_path),
        })

        print(f"Created: {config_path}")

    return experiments


def generate_encoding_ablation(output_dir: Path, group: str = "overnight_v1"):
    """Generate positional/coordinate encoding ablation experiments."""
    experiments = []

    # Positional encoding ablation
    for pos_enc in ["learnable", "sinusoidal", "none"]:
        name = f"medium_adamw_pos_{pos_enc}"
        config = create_config(
            name=name,
            model_size="medium",
            optimizer="adamw",
            precision="fp32",
            pos_encoding=pos_enc,
            group=group,
        )

        config_path = output_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        experiments.append({"name": name, "config": str(config_path)})
        print(f"Created: {config_path}")

    # Coordinate encoding ablation
    for coord_enc in ["linear", "fourier"]:
        name = f"medium_adamw_coord_{coord_enc}"
        config = create_config(
            name=name,
            model_size="medium",
            optimizer="adamw",
            precision="fp32",
            coord_encoding=coord_enc,
            group=group,
        )

        config_path = output_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        experiments.append({"name": name, "config": str(config_path)})
        print(f"Created: {config_path}")

    return experiments


def generate_norm_loss_ablation(output_dir: Path, group: str = "overnight_v1"):
    """Generate normalization and loss type ablation experiments."""
    experiments = []

    # Normalization ablation
    for norm in ["layernorm", "rmsnorm"]:
        name = f"medium_adamw_norm_{norm}"
        config = create_config(
            name=name,
            model_size="medium",
            optimizer="adamw",
            precision="fp32",
            norm_type=norm,
            group=group,
        )

        config_path = output_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        experiments.append({"name": name, "config": str(config_path)})
        print(f"Created: {config_path}")

    # Loss type ablation
    for loss in ["mse", "huber"]:
        name = f"medium_adamw_loss_{loss}"
        config = create_config(
            name=name,
            model_size="medium",
            optimizer="adamw",
            precision="fp32",
            loss_type=loss,
            group=group,
        )

        config_path = output_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        experiments.append({"name": name, "config": str(config_path)})
        print(f"Created: {config_path}")

    return experiments


def main():
    output_dir = Path("configs/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    group = "overnight_v1"

    print(f"\n{'='*60}")
    print("Generating Main Experiments (model_size x optimizer x precision)")
    print(f"{'='*60}")
    main_experiments = generate_main_experiments(output_dir, group)

    print(f"\n{'='*60}")
    print("Generating Learning Rate Ablation")
    print(f"{'='*60}")
    lr_experiments = generate_lr_ablation(output_dir, group)

    print(f"\n{'='*60}")
    print("Generating Schedule Ablation")
    print(f"{'='*60}")
    schedule_experiments = generate_schedule_ablation(output_dir, group)

    print(f"\n{'='*60}")
    print("Generating Encoding Ablation (pos_encoding, coord_encoding)")
    print(f"{'='*60}")
    encoding_experiments = generate_encoding_ablation(output_dir, group)

    print(f"\n{'='*60}")
    print("Generating Normalization & Loss Ablation")
    print(f"{'='*60}")
    norm_loss_experiments = generate_norm_loss_ablation(output_dir, group)

    # Generate experiment list for runner
    all_experiments = (
        main_experiments +
        lr_experiments +
        schedule_experiments +
        encoding_experiments +
        norm_loss_experiments
    )
    experiments_list_path = output_dir / "experiment_list.yaml"
    with open(experiments_list_path, "w") as f:
        yaml.dump({
            "group": group,
            "experiments": [e["config"] for e in all_experiments],
        }, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_experiments)} experiment configs")
    print(f"Experiment list saved to: {experiments_list_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
