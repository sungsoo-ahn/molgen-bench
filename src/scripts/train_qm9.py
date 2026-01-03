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
from collections import defaultdict
import wandb

from src.data import QM9Dataset, collate_molecular_data
from src.models.architectures import DiT
from src.models.generative import FlowMatching
from src.evaluation.sample_quality import compute_atom_stability, compute_molecule_stability
from src.utils.optimizers import create_optimizer, CombinedOptimizer


def compute_gradient_metrics(model: torch.nn.Module) -> dict:
    """Compute gradient statistics for model parameters.

    Args:
        model: The model with computed gradients

    Returns:
        dict with gradient metrics
    """
    metrics = {}

    # Total gradient norm
    total_norm = 0.0
    num_params_with_grad = 0

    # Per-component gradient norms
    backbone_norm = 0.0
    velocity_head_norm = 0.0

    # Gradient statistics
    all_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params_with_grad += 1

            # Collect for statistics
            all_grads.append(param.grad.data.flatten())

            # Component-wise
            if 'backbone' in name:
                backbone_norm += param_norm ** 2
            elif 'velocity_head' in name:
                velocity_head_norm += param_norm ** 2

    total_norm = total_norm ** 0.5
    backbone_norm = backbone_norm ** 0.5
    velocity_head_norm = velocity_head_norm ** 0.5

    metrics["grad/total_norm"] = total_norm
    metrics["grad/backbone_norm"] = backbone_norm
    metrics["grad/velocity_head_norm"] = velocity_head_norm

    # Gradient statistics
    if all_grads:
        all_grads = torch.cat(all_grads)
        metrics["grad/mean"] = all_grads.mean().item()
        metrics["grad/std"] = all_grads.std().item()
        metrics["grad/abs_max"] = all_grads.abs().max().item()
        metrics["grad/num_zeros"] = (all_grads == 0).sum().item()
        metrics["grad/sparsity"] = (all_grads == 0).float().mean().item()

    return metrics


def compute_weight_metrics(model: torch.nn.Module) -> dict:
    """Compute weight statistics for model parameters.

    Args:
        model: The model

    Returns:
        dict with weight metrics
    """
    metrics = {}

    # Per-layer weight norms
    total_norm = 0.0
    backbone_norm = 0.0
    velocity_head_norm = 0.0

    # Weight statistics
    all_weights = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2).item()
            total_norm += param_norm ** 2
            all_weights.append(param.data.flatten())

            if 'backbone' in name:
                backbone_norm += param_norm ** 2
            elif 'velocity_head' in name:
                velocity_head_norm += param_norm ** 2

    total_norm = total_norm ** 0.5
    backbone_norm = backbone_norm ** 0.5
    velocity_head_norm = velocity_head_norm ** 0.5

    metrics["weight/total_norm"] = total_norm
    metrics["weight/backbone_norm"] = backbone_norm
    metrics["weight/velocity_head_norm"] = velocity_head_norm

    # Weight statistics
    if all_weights:
        all_weights = torch.cat(all_weights)
        metrics["weight/mean"] = all_weights.mean().item()
        metrics["weight/std"] = all_weights.std().item()
        metrics["weight/abs_max"] = all_weights.abs().max().item()

    return metrics


def compute_ema_diff_metrics(model: torch.nn.Module, ema) -> dict:
    """Compute difference between model weights and EMA weights.

    Args:
        model: The model
        ema: EMA object

    Returns:
        dict with EMA difference metrics
    """
    if ema is None:
        return {}

    metrics = {}
    total_diff = 0.0
    num_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad and name in ema.shadow:
            diff = (param.data - ema.shadow[name]).norm(2).item()
            total_diff += diff ** 2
            num_params += 1

    metrics["ema/weight_diff_norm"] = (total_diff ** 0.5) if num_params > 0 else 0.0

    return metrics


def compute_data_metrics(batch, atom_type_map: dict) -> dict:
    """Compute statistics about the data batch.

    Args:
        batch: Data batch
        atom_type_map: Mapping from atomic numbers to indices

    Returns:
        dict with data metrics
    """
    metrics = {}

    positions = batch.positions
    atom_types = batch.atom_types
    batch_idx = batch.batch_idx

    # Position statistics
    metrics["data/pos_mean"] = positions.mean().item()
    metrics["data/pos_std"] = positions.std().item()
    metrics["data/pos_abs_max"] = positions.abs().max().item()

    # Per-dimension statistics
    metrics["data/pos_x_std"] = positions[:, 0].std().item()
    metrics["data/pos_y_std"] = positions[:, 1].std().item()
    metrics["data/pos_z_std"] = positions[:, 2].std().item()

    # Molecule size statistics
    num_graphs = batch_idx.max().item() + 1
    mol_sizes = torch.bincount(batch_idx)
    metrics["data/mol_size_mean"] = mol_sizes.float().mean().item()
    metrics["data/mol_size_std"] = mol_sizes.float().std().item()
    metrics["data/mol_size_min"] = mol_sizes.min().item()
    metrics["data/mol_size_max"] = mol_sizes.max().item()
    metrics["data/num_molecules"] = num_graphs

    # Atom type distribution
    atom_counts = torch.bincount(atom_types, minlength=10)
    index_to_name = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
    total_atoms = len(atom_types)
    for atomic_num, idx in atom_type_map.items():
        name = index_to_name.get(atomic_num, str(atomic_num))
        count = atom_counts[atomic_num].item()
        metrics[f"data/frac_{name}"] = count / total_atoms if total_atoms > 0 else 0

    return metrics


def compute_model_internals(model: torch.nn.Module) -> dict:
    """Compute internal model statistics (adaLN, attention, etc).

    Args:
        model: The model (expected to be FlowMatching with DiT backbone)

    Returns:
        dict with model internal metrics
    """
    metrics = {}

    backbone = model.backbone

    # Check if DiT backbone
    if not hasattr(backbone, 'blocks'):
        return metrics

    # Analyze adaLN modulation weights
    adaLN_norms = []
    for i, block in enumerate(backbone.blocks):
        if hasattr(block, 'adaLN_modulation'):
            # Get the linear layer weight
            linear = block.adaLN_modulation[1]  # nn.Sequential: [SiLU, Linear]
            weight_norm = linear.weight.data.norm(2).item()
            bias_norm = linear.bias.data.norm(2).item() if linear.bias is not None else 0
            adaLN_norms.append(weight_norm)
            metrics[f"model/block{i}_adaLN_weight_norm"] = weight_norm
            metrics[f"model/block{i}_adaLN_bias_norm"] = bias_norm

    if adaLN_norms:
        metrics["model/adaLN_weight_norm_mean"] = np.mean(adaLN_norms)
        metrics["model/adaLN_weight_norm_std"] = np.std(adaLN_norms)

    # Attention weight norms
    attn_norms = []
    for i, block in enumerate(backbone.blocks):
        if hasattr(block, 'attn'):
            in_proj_norm = block.attn.in_proj_weight.data.norm(2).item()
            out_proj_norm = block.attn.out_proj.weight.data.norm(2).item()
            attn_norms.append(in_proj_norm)
            metrics[f"model/block{i}_attn_in_proj_norm"] = in_proj_norm
            metrics[f"model/block{i}_attn_out_proj_norm"] = out_proj_norm

    if attn_norms:
        metrics["model/attn_in_proj_norm_mean"] = np.mean(attn_norms)

    # MLP weight norms
    mlp_norms = []
    for i, block in enumerate(backbone.blocks):
        if hasattr(block, 'mlp'):
            fc1_norm = block.mlp.fc1.weight.data.norm(2).item()
            fc2_norm = block.mlp.fc2.weight.data.norm(2).item()
            mlp_norms.append(fc1_norm)
            metrics[f"model/block{i}_mlp_fc1_norm"] = fc1_norm
            metrics[f"model/block{i}_mlp_fc2_norm"] = fc2_norm

    if mlp_norms:
        metrics["model/mlp_fc1_norm_mean"] = np.mean(mlp_norms)

    # Timestep embedding weight norm
    if hasattr(backbone, 'time_embed'):
        time_mlp = backbone.time_embed.mlp
        metrics["model/time_embed_fc1_norm"] = time_mlp[0].weight.data.norm(2).item()
        metrics["model/time_embed_fc2_norm"] = time_mlp[2].weight.data.norm(2).item()

    # Atom embedding statistics
    if hasattr(backbone, 'atom_embedding'):
        emb_weight = backbone.atom_embedding.weight.data
        metrics["model/atom_emb_norm"] = emb_weight.norm(2).item()
        metrics["model/atom_emb_std"] = emb_weight.std().item()

    return metrics


class MetricsAccumulator:
    """Accumulator for averaging metrics over batches."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def add(self, metrics_dict: dict):
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.metrics[key].append(value)

    def get_averages(self) -> dict:
        return {key: np.mean(values) for key, values in self.metrics.items()}

    def reset(self):
        self.metrics = defaultdict(list)


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
            group=wandb_config.get("group", None),  # For grouping related experiments
            job_type=wandb_config.get("job_type", "train"),
            config=config,
            tags=wandb_config.get("tags", []),
        )
        wandb.run.log_code(root=".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
        print(f"WandB initialized: {wandb.run.name}")
        if wandb_config.get("group"):
            print(f"  Group: {wandb_config.get('group')}")
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
        pos_encoding=config["architecture"].get("pos_encoding", "learnable"),
        coord_encoding=config["architecture"].get("coord_encoding", "linear"),
        norm_type=config["architecture"].get("norm_type", "layernorm"),
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
    optimizer_type = config["training"].get("optimizer", "adamw")
    muon_lr = config["training"].get("muon_lr", 0.02)
    muon_momentum = config["training"].get("muon_momentum", 0.95)

    optimizer = create_optimizer(
        model,
        optimizer_type=optimizer_type,
        learning_rate=base_lr,
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999),
        muon_momentum=muon_momentum,
        muon_lr=muon_lr,
    )

    # Log optimizer info
    if isinstance(optimizer, CombinedOptimizer):
        print(f"Optimizer: {optimizer_type} (combined: {optimizer._names})")
        for name, opt in optimizer.optimizers:
            num_params = sum(p.numel() for g in opt.param_groups for p in g['params'])
            print(f"  {name}: {num_params:,} parameters, lr={opt.param_groups[0]['lr']}")
    else:
        print(f"Optimizer: {optimizer_type}, lr={base_lr}")

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

    # Logging configuration
    log_interval = config["training"].get("log_interval", 100)
    log_detailed_metrics = config["training"].get("log_detailed_metrics", True)

    for epoch in range(config["training"]["epochs"]):
        # Training
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        # Metrics accumulator for detailed logging
        train_metrics = MetricsAccumulator()

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

            # Compute gradient metrics before clipping
            if log_detailed_metrics and global_step % log_interval == 0:
                grad_metrics_pre_clip = compute_gradient_metrics(model)
                train_metrics.add({f"{k}_pre_clip": v for k, v in grad_metrics_pre_clip.items()})

            # Gradient clipping
            grad_norm_clipped = None
            if config["training"]["gradient_clip"] > 0:
                grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["gradient_clip"]
                )

            # Compute gradient metrics after clipping
            if log_detailed_metrics and global_step % log_interval == 0:
                grad_metrics_post_clip = compute_gradient_metrics(model)
                train_metrics.add(grad_metrics_post_clip)

                # Data metrics
                data_metrics = compute_data_metrics(batch, atom_type_map)
                train_metrics.add(data_metrics)

                # Loss component metrics from info
                for key, value in info.items():
                    train_metrics.add({f"loss/{key}": value})

            # Apply warmup learning rate
            if warmup_steps > 0 and global_step < warmup_steps:
                warmup_factor = (global_step + 1) / warmup_steps
                if isinstance(optimizer, CombinedOptimizer):
                    # Scale each optimizer's base LR
                    for name, opt in optimizer.optimizers:
                        for param_group in opt.param_groups:
                            base = muon_lr if name == 'muon' else base_lr
                            param_group['lr'] = base * warmup_factor
                else:
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

            if isinstance(optimizer, CombinedOptimizer):
                current_lr = optimizer.param_groups[0]['lr']  # First group's LR
                lr_dict = optimizer.get_lr()
            else:
                current_lr = optimizer.param_groups[0]['lr']
                lr_dict = {"adamw": current_lr}
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
                "cos_sim": f"{info.get('cosine_similarity', 0):.3f}"
            })

            # Detailed per-step logging
            if wandb.run is not None and global_step % log_interval == 0:
                step_metrics = {
                    "train/step_loss": loss.item(),
                    "train/lr": current_lr,
                    "train/global_step": global_step,
                }

                # Log per-optimizer learning rates
                for opt_name, opt_lr in lr_dict.items():
                    step_metrics[f"train/lr_{opt_name}"] = opt_lr

                # Add flow matching specific metrics
                step_metrics["train/loss_early"] = info.get("loss_early", 0)
                step_metrics["train/loss_mid"] = info.get("loss_mid", 0)
                step_metrics["train/loss_late"] = info.get("loss_late", 0)
                step_metrics["train/cosine_similarity"] = info.get("cosine_similarity", 0)
                step_metrics["train/pred_std"] = info.get("pred_std", 0)
                step_metrics["train/target_std"] = info.get("target_std", 0)
                step_metrics["train/error_mean"] = info.get("error_mean", 0)

                # Gradient norm
                if grad_norm_clipped is not None:
                    step_metrics["train/grad_norm_before_clip"] = grad_metrics_pre_clip.get("grad/total_norm", 0)
                    step_metrics["train/grad_norm_after_clip"] = grad_norm_clipped.item() if torch.is_tensor(grad_norm_clipped) else grad_norm_clipped

                wandb.log(step_metrics, step=global_step)

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

        # Compute epoch-level detailed metrics
        if log_detailed_metrics:
            # Weight metrics
            weight_metrics = compute_weight_metrics(model)

            # EMA difference metrics
            ema_metrics = compute_ema_diff_metrics(model, ema)

            # Model internal metrics (adaLN, attention, etc)
            model_internal_metrics = compute_model_internals(model)

            # Get averaged training metrics from accumulator
            avg_train_metrics = train_metrics.get_averages()

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
            if isinstance(optimizer, CombinedOptimizer):
                current_lr = optimizer.param_groups[0]['lr']
                lr_dict = optimizer.get_lr()
            else:
                current_lr = optimizer.param_groups[0]['lr']
                lr_dict = {"adamw": current_lr}

            epoch_metrics = {
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "train/lr": current_lr,
                "train/global_step": global_step,
            }

            # Log per-optimizer learning rates
            for opt_name, opt_lr in lr_dict.items():
                epoch_metrics[f"train/lr_{opt_name}"] = opt_lr

            # Add detailed metrics if enabled
            if log_detailed_metrics:
                # Weight metrics
                for key, value in weight_metrics.items():
                    epoch_metrics[key] = value

                # EMA metrics
                for key, value in ema_metrics.items():
                    epoch_metrics[key] = value

                # Model internal metrics
                for key, value in model_internal_metrics.items():
                    epoch_metrics[key] = value

                # Averaged training metrics (loss components, gradients, data stats)
                for key, value in avg_train_metrics.items():
                    # Prefix with 'train_avg/' if not already prefixed
                    if not key.startswith(('train/', 'val/', 'gen/', 'weight/', 'ema/', 'model/', 'grad/', 'data/', 'loss/')):
                        epoch_metrics[f"train_avg/{key}"] = value
                    else:
                        epoch_metrics[key] = value

            wandb.log(epoch_metrics)

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
