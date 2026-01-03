"""Scaling law tracking and analysis.

Tracks model performance as a function of:
- Model size (number of parameters)
- Dataset size (number of training examples)
- Compute (FLOPs or training steps)

Implements analysis methods from:
- "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
- "Chinchilla" scaling laws (Hoffmann et al., 2022)
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path


class ScalingLawTracker:
    """Track and analyze scaling behavior of models."""

    def __init__(self, save_dir: str = "./scaling_logs"):
        """Initialize tracker.

        Args:
            save_dir: Directory to save scaling data
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for scaling data
        self.data = {
            "model_size": [],  # Number of parameters
            "dataset_size": [],  # Number of training examples
            "training_steps": [],  # Training iterations
            "train_loss": [],  # Training loss
            "val_loss": [],  # Validation loss
            "flops_per_batch": [],  # FLOPs per batch
            "wall_time": [],  # Wall-clock time
            "gpu_memory": [],  # GPU memory usage
        }

    def log(
        self,
        step: int,
        train_loss: float,
        val_loss: float,
        model_size: int,
        dataset_size: int,
        flops_per_batch: Optional[float] = None,
        wall_time: Optional[float] = None,
        gpu_memory: Optional[float] = None
    ):
        """Log a data point.

        Args:
            step: Training step
            train_loss: Training loss
            val_loss: Validation loss
            model_size: Number of parameters
            dataset_size: Number of training examples used
            flops_per_batch: FLOPs per batch (optional)
            wall_time: Wall-clock time in seconds
            gpu_memory: GPU memory in GB
        """
        self.data["training_steps"].append(step)
        self.data["train_loss"].append(train_loss)
        self.data["val_loss"].append(val_loss)
        self.data["model_size"].append(model_size)
        self.data["dataset_size"].append(dataset_size)
        self.data["flops_per_batch"].append(flops_per_batch)
        self.data["wall_time"].append(wall_time)
        self.data["gpu_memory"].append(gpu_memory)

    def save(self, filename: str = "scaling_data.json"):
        """Save scaling data to file."""
        filepath = self.save_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Saved scaling data to {filepath}")

    def load(self, filename: str = "scaling_data.json"):
        """Load scaling data from file."""
        filepath = self.save_dir / filename
        with open(filepath, "r") as f:
            self.data = json.load(f)
        print(f"Loaded scaling data from {filepath}")

    def fit_power_law(
        self,
        x_key: str,
        y_key: str = "val_loss",
        min_points: int = 3
    ) -> Dict[str, float]:
        """Fit power law: L(X) = a * X^b + c

        Args:
            x_key: Independent variable (e.g., "model_size", "dataset_size")
            y_key: Dependent variable (e.g., "val_loss")
            min_points: Minimum number of points required

        Returns:
            dict with fitted parameters and R^2
        """
        x = np.array(self.data[x_key])
        y = np.array(self.data[y_key])

        # Remove None values
        mask = (x != None) & (y != None)
        x = x[mask].astype(float)
        y = y[mask].astype(float)

        if len(x) < min_points:
            return {"error": f"Not enough data points (need {min_points}, got {len(x)})"}

        # Fit in log space: log(L) = log(a) + b * log(X)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_x = np.log(x + epsilon)
        log_y = np.log(y + epsilon)

        # Linear regression in log space
        A = np.vstack([log_x, np.ones(len(log_x))]).T
        result = np.linalg.lstsq(A, log_y, rcond=None)
        b, log_a = result[0]
        a = np.exp(log_a)

        # Compute R^2
        y_pred = a * x ** b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "a": float(a),
            "b": float(b),
            "r2": float(r2),
            "equation": f"L({x_key}) = {a:.4e} * {x_key}^{b:.4f}",
            "n_points": len(x)
        }

    def estimate_compute_optimal(
        self,
        compute_budget: float,
        model_size_scaling: Dict[str, float],
        dataset_size_scaling: Dict[str, float]
    ) -> Dict[str, float]:
        """Estimate compute-optimal model and dataset size.

        Based on Chinchilla scaling laws: for a given compute budget,
        model size and dataset size should scale equally.

        Args:
            compute_budget: Available compute in FLOPs
            model_size_scaling: Power law fit for model size
            dataset_size_scaling: Power law fit for dataset size

        Returns:
            dict with optimal model size and dataset size
        """
        # This is a simplified estimate
        # Full implementation would solve the optimization problem

        # Rough heuristic: N and D should scale as sqrt(C)
        sqrt_compute = np.sqrt(compute_budget)

        optimal_params = sqrt_compute * 1e-6  # Rough scaling
        optimal_tokens = sqrt_compute * 1e-3

        return {
            "compute_budget": compute_budget,
            "optimal_model_size": optimal_params,
            "optimal_dataset_size": optimal_tokens,
            "note": "This is a rough estimate. Run ablations to find true optimum."
        }

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}

        for key in ["train_loss", "val_loss"]:
            values = [v for v in self.data[key] if v is not None]
            if values:
                summary[key] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "final": float(values[-1])
                }

        summary["total_steps"] = len(self.data["training_steps"])

        # Try to fit scaling laws
        if len(self.data["model_size"]) > 0:
            summary["model_size_scaling"] = self.fit_power_law("model_size", "val_loss")

        if len(self.data["dataset_size"]) > 0:
            summary["dataset_size_scaling"] = self.fit_power_law("dataset_size", "val_loss")

        return summary
