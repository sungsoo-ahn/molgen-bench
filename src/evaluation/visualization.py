"""Visualization utilities for molecular generation."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import seaborn as sns


def plot_property_distributions(
    generated_molecules: List[Dict],
    reference_molecules: List[Dict],
    property_key: str = "num_atoms",
    save_path: Optional[str] = None
):
    """Plot distribution comparison.

    Args:
        generated_molecules: Generated molecules
        reference_molecules: Reference molecules
        property_key: Property to plot
        save_path: Path to save figure
    """
    # Extract property values
    gen_values = []
    ref_values = []

    for mol in generated_molecules:
        if property_key in mol.get("properties", {}):
            gen_values.append(float(mol["properties"][property_key]))
        elif property_key == "num_atoms":
            gen_values.append(len(mol["atom_types"]))

    for mol in reference_molecules:
        if property_key in mol.get("properties", {}):
            ref_values.append(float(mol["properties"][property_key]))
        elif property_key == "num_atoms":
            ref_values.append(len(mol["atom_types"]))

    # Plot
    plt.figure(figsize=(10, 6))

    plt.hist(ref_values, bins=30, alpha=0.5, label="Reference", density=True)
    plt.hist(gen_values, bins=30, alpha=0.5, label="Generated", density=True)

    plt.xlabel(property_key)
    plt.ylabel("Density")
    plt.title(f"Distribution of {property_key}")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_scaling_laws(
    scaling_data: Dict,
    save_dir: Optional[str] = None
):
    """Plot scaling law curves.

    Args:
        scaling_data: Dict with scaling data
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss vs model size
    if len(scaling_data["model_size"]) > 0:
        model_sizes = np.array([x for x in scaling_data["model_size"] if x is not None])
        val_losses = np.array([scaling_data["val_loss"][i]
                              for i, x in enumerate(scaling_data["model_size"]) if x is not None])

        if len(model_sizes) > 0:
            axes[0].loglog(model_sizes, val_losses, 'o-', alpha=0.7)
            axes[0].set_xlabel("Model Size (parameters)")
            axes[0].set_ylabel("Validation Loss")
            axes[0].set_title("Scaling Law: Loss vs Model Size")
            axes[0].grid(alpha=0.3)

    # Loss vs training steps
    if len(scaling_data["training_steps"]) > 0:
        steps = np.array(scaling_data["training_steps"])
        train_losses = np.array(scaling_data["train_loss"])
        val_losses = np.array(scaling_data["val_loss"])

        axes[1].semilogy(steps, train_losses, label="Train", alpha=0.7)
        axes[1].semilogy(steps, val_losses, label="Validation", alpha=0.7)
        axes[1].set_xlabel("Training Steps")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Curves")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = f"{save_dir}/scaling_laws.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved scaling law plot to {save_path}")
    else:
        plt.show()

    plt.close()
