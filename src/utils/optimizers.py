"""Optimizer utilities for training.

Supports AdamW and Muon optimizers with proper parameter grouping.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_patterns: List[str] = None
) -> List[Dict[str, Any]]:
    """Get parameter groups with weight decay filtering.

    Args:
        model: The model
        weight_decay: Weight decay value
        no_decay_patterns: Patterns for parameters that shouldn't have weight decay

    Returns:
        List of parameter group dicts
    """
    if no_decay_patterns is None:
        no_decay_patterns = ['bias', 'LayerNorm', 'layer_norm', 'embedding']

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(pattern in name for pattern in no_decay_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def get_muon_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
) -> Tuple[List[Dict], List[Dict]]:
    """Get separate parameter groups for Muon and AdamW.

    Muon is designed for hidden layer weights (2D+ tensors).
    AdamW handles embeddings, biases, layer norms, and 1D parameters.

    Args:
        model: The model
        weight_decay: Weight decay value
        muon_lr: Learning rate for Muon optimizer
        adamw_lr: Learning rate for AdamW optimizer

    Returns:
        (muon_params, adamw_params) - Lists of parameter group dicts
    """
    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []

    # Patterns that should use AdamW without decay
    no_decay_patterns = ['bias', 'LayerNorm', 'layer_norm', 'norm']
    # Patterns that should use AdamW (with decay)
    adamw_patterns = ['embedding', 'embed', 'pos_embedding', 'time_embed']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if it should use AdamW
        use_adamw = any(pattern in name for pattern in adamw_patterns)
        no_decay = any(pattern in name for pattern in no_decay_patterns)

        # Muon works best on 2D+ weight matrices in hidden layers
        is_hidden_weight = param.dim() >= 2 and not use_adamw and not no_decay

        if is_hidden_weight:
            muon_params.append(param)
        elif no_decay:
            adamw_no_decay_params.append(param)
        else:
            adamw_decay_params.append(param)

    muon_groups = [{'params': muon_params, 'lr': muon_lr}] if muon_params else []
    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append({'params': adamw_decay_params, 'weight_decay': weight_decay, 'lr': adamw_lr})
    if adamw_no_decay_params:
        adamw_groups.append({'params': adamw_no_decay_params, 'weight_decay': 0.0, 'lr': adamw_lr})

    return muon_groups, adamw_groups


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    muon_momentum: float = 0.95,
    muon_lr: float = 0.02,
) -> torch.optim.Optimizer:
    """Create optimizer based on configuration.

    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer ("adamw", "muon", "adam", "sgd")
        learning_rate: Learning rate (for AdamW/Adam/SGD)
        weight_decay: Weight decay
        betas: Beta values for Adam/AdamW
        muon_momentum: Momentum for Muon
        muon_lr: Learning rate for Muon (typically higher than AdamW)

    Returns:
        Configured optimizer
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw":
        param_groups = get_parameter_groups(model, weight_decay)
        return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)

    elif optimizer_type == "adam":
        param_groups = get_parameter_groups(model, weight_decay=0.0)
        return torch.optim.Adam(param_groups, lr=learning_rate, betas=betas)

    elif optimizer_type == "sgd":
        param_groups = get_parameter_groups(model, weight_decay)
        return torch.optim.SGD(param_groups, lr=learning_rate, momentum=0.9)

    elif optimizer_type == "muon":
        # Muon for hidden weights, AdamW for embeddings/biases/norms
        muon_groups, adamw_groups = get_muon_parameter_groups(
            model,
            weight_decay=weight_decay,
            muon_lr=muon_lr,
            adamw_lr=learning_rate
        )

        optimizers = []

        if muon_groups:
            muon_opt = torch.optim.Muon(
                muon_groups,
                lr=muon_lr,
                momentum=muon_momentum,
                nesterov=True,
                weight_decay=weight_decay,
            )
            optimizers.append(('muon', muon_opt))

        if adamw_groups:
            adamw_opt = torch.optim.AdamW(adamw_groups, betas=betas)
            optimizers.append(('adamw', adamw_opt))

        # Return a combined optimizer wrapper
        return CombinedOptimizer(optimizers)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class CombinedOptimizer:
    """Wrapper to use multiple optimizers together (e.g., Muon + AdamW)."""

    def __init__(self, optimizers: List[Tuple[str, torch.optim.Optimizer]]):
        """Initialize with list of (name, optimizer) tuples."""
        self.optimizers = optimizers
        self._names = [name for name, _ in optimizers]

    def zero_grad(self, set_to_none: bool = True):
        for _, opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for _, opt in self.optimizers:
            opt.step()

    def state_dict(self) -> Dict[str, Any]:
        return {name: opt.state_dict() for name, opt in self.optimizers}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name, opt in self.optimizers:
            if name in state_dict:
                opt.load_state_dict(state_dict[name])

    @property
    def param_groups(self) -> List[Dict]:
        """Return all param groups from all optimizers."""
        groups = []
        for _, opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def get_lr(self) -> Dict[str, float]:
        """Get learning rates for each optimizer."""
        lrs = {}
        for name, opt in self.optimizers:
            lrs[name] = opt.param_groups[0]['lr']
        return lrs

    def set_lr(self, lr: float, optimizer_name: str = None):
        """Set learning rate for specified or all optimizers."""
        for name, opt in self.optimizers:
            if optimizer_name is None or name == optimizer_name:
                for group in opt.param_groups:
                    group['lr'] = lr
