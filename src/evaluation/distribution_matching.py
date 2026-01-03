"""Distribution matching metrics.

Measures how well the generated distribution matches the data distribution.

Implements:
- Wasserstein distance
- Fréchet ChemNet Distance (FCD) - placeholder
- Coverage metrics
- Property distribution comparisons
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel


def compute_wasserstein_distance(
    generated_molecules: List[Dict],
    reference_molecules: List[Dict],
    property_keys: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute Wasserstein distance between generated and reference distributions.

    Args:
        generated_molecules: Generated molecules with properties
        reference_molecules: Reference molecules with properties
        property_keys: List of property keys to compare

    Returns:
        dict with Wasserstein distances for each property
    """
    if property_keys is None:
        # Default properties
        property_keys = ["num_atoms", "num_bonds"]

    results = {}

    for prop_key in property_keys:
        # Extract property values
        gen_values = []
        ref_values = []

        for mol in generated_molecules:
            if prop_key in mol.get("properties", {}):
                gen_values.append(float(mol["properties"][prop_key]))
            elif prop_key == "num_atoms":
                gen_values.append(len(mol["atom_types"]))

        for mol in reference_molecules:
            if prop_key in mol.get("properties", {}):
                ref_values.append(float(mol["properties"][prop_key]))
            elif prop_key == "num_atoms":
                ref_values.append(len(mol["atom_types"]))

        if len(gen_values) > 0 and len(ref_values) > 0:
            wd = wasserstein_distance(gen_values, ref_values)
            results[f"wasserstein_{prop_key}"] = wd
        else:
            results[f"wasserstein_{prop_key}"] = None

    return results


def compute_property_distributions(
    generated_molecules: List[Dict],
    reference_molecules: List[Dict],
    property_keys: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Compare property distributions between generated and reference.

    Args:
        generated_molecules: Generated molecules
        reference_molecules: Reference molecules
        property_keys: Properties to compare

    Returns:
        dict with distribution statistics
    """
    if property_keys is None:
        property_keys = ["num_atoms"]

    results = {}

    for prop_key in property_keys:
        # Extract values
        gen_values = []
        ref_values = []

        for mol in generated_molecules:
            if prop_key in mol.get("properties", {}):
                val = mol["properties"][prop_key]
                if torch.is_tensor(val):
                    val = val.item()
                gen_values.append(float(val))
            elif prop_key == "num_atoms":
                gen_values.append(len(mol["atom_types"]))

        for mol in reference_molecules:
            if prop_key in mol.get("properties", {}):
                val = mol["properties"][prop_key]
                if torch.is_tensor(val):
                    val = val.item()
                ref_values.append(float(val))
            elif prop_key == "num_atoms":
                ref_values.append(len(mol["atom_types"]))

        if len(gen_values) > 0 and len(ref_values) > 0:
            gen_values = np.array(gen_values)
            ref_values = np.array(ref_values)

            results[prop_key] = {
                "generated": {
                    "mean": float(np.mean(gen_values)),
                    "std": float(np.std(gen_values)),
                    "min": float(np.min(gen_values)),
                    "max": float(np.max(gen_values)),
                },
                "reference": {
                    "mean": float(np.mean(ref_values)),
                    "std": float(np.std(ref_values)),
                    "min": float(np.min(ref_values)),
                    "max": float(np.max(ref_values)),
                },
                "mean_absolute_error": float(np.abs(np.mean(gen_values) - np.mean(ref_values))),
                "std_absolute_error": float(np.abs(np.std(gen_values) - np.std(ref_values)))
            }

    return results


def compute_coverage(
    generated_molecules: List[Dict],
    reference_molecules: List[Dict],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute coverage metric.

    Coverage = fraction of reference molecules that are 'close' to at least
    one generated molecule.

    Args:
        generated_molecules: Generated molecules
        reference_molecules: Reference molecules
        threshold: Distance threshold for considering molecules 'close'

    Returns:
        dict with coverage metrics
    """
    # This is a simplified placeholder
    # Full implementation would use molecular fingerprints and Tanimoto similarity

    num_covered = 0

    for ref_mol in reference_molecules:
        ref_atoms = ref_mol["atom_types"]

        # Check if any generated molecule is similar
        is_covered = False

        for gen_mol in generated_molecules:
            gen_atoms = gen_mol["atom_types"]

            # Simple similarity: same number and types of atoms
            if len(ref_atoms) == len(gen_atoms):
                ref_sorted = sorted(ref_atoms.cpu().numpy() if torch.is_tensor(ref_atoms) else ref_atoms)
                gen_sorted = sorted(gen_atoms.cpu().numpy() if torch.is_tensor(gen_atoms) else gen_atoms)

                if np.array_equal(ref_sorted, gen_sorted):
                    is_covered = True
                    break

        if is_covered:
            num_covered += 1

    coverage = num_covered / len(reference_molecules) if len(reference_molecules) > 0 else 0.0

    return {
        "coverage": coverage,
        "num_covered": num_covered,
        "num_reference": len(reference_molecules)
    }


def compute_fcd(
    generated_molecules: List[Dict],
    reference_molecules: List[Dict]
) -> float:
    """Compute Fréchet ChemNet Distance.

    This is a placeholder. Full implementation requires:
    1. ChemNet feature extractor
    2. Computing activations for generated and reference molecules
    3. Computing Fréchet distance between activation distributions

    Args:
        generated_molecules: Generated molecules
        reference_molecules: Reference molecules

    Returns:
        FCD score
    """
    # Placeholder
    # In practice:
    # 1. Convert molecules to SMILES or molecular graphs
    # 2. Extract features using pre-trained ChemNet
    # 3. Compute mean and covariance of features
    # 4. Compute Fréchet distance

    return 0.0  # Placeholder
