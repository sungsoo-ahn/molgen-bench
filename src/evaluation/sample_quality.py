"""Sample quality metrics for generated molecules.

Implements:
- Validity: Fraction of chemically valid molecules
- Uniqueness: Fraction of unique molecules
- Novelty: Fraction of molecules not in training set
- Molecular properties: QED, SA, logP, etc.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from collections import Counter


def compute_validity(
    generated_molecules: List[Dict],
    check_valence: bool = True,
    check_kekulize: bool = True
) -> Dict[str, float]:
    """Compute validity of generated molecules.

    Note: This is a simplified version. Full implementation would use RDKit
    for proper chemical validity checking.

    Args:
        generated_molecules: List of dicts with 'positions' and 'atom_types'
        check_valence: Whether to check valence rules
        check_kekulize: Whether to check Kekulization

    Returns:
        dict with validity metrics
    """
    num_molecules = len(generated_molecules)
    if num_molecules == 0:
        return {"validity": 0.0, "num_valid": 0, "num_total": 0}

    # Simplified validity check
    # In practice, you would:
    # 1. Convert positions + atom_types to RDKit molecule
    # 2. Check chemical validity
    # 3. Check valence rules
    # 4. Try to kekulize

    num_valid = 0

    for mol_data in generated_molecules:
        positions = mol_data["positions"]
        atom_types = mol_data["atom_types"]

        # Basic checks
        is_valid = True

        # Check if molecule is not empty
        if len(atom_types) == 0:
            is_valid = False

        # Check if positions are reasonable (not NaN, not too large)
        if torch.isnan(positions).any() or torch.isinf(positions).any():
            is_valid = False

        if torch.abs(positions).max() > 1000:  # Arbitrary large threshold
            is_valid = False

        # TODO: Add RDKit-based validity checking
        # mol = positions_to_rdkit_mol(positions, atom_types)
        # if mol is None:
        #     is_valid = False
        # elif check_valence and not check_mol_valence(mol):
        #     is_valid = False
        # elif check_kekulize:
        #     try:
        #         Chem.Kekulize(mol)
        #     except:
        #         is_valid = False

        if is_valid:
            num_valid += 1

    validity = num_valid / num_molecules

    return {
        "validity": validity,
        "num_valid": num_valid,
        "num_total": num_molecules
    }


def compute_uniqueness(
    generated_molecules: List[Dict],
    use_positions: bool = True,
    position_tolerance: float = 0.1
) -> Dict[str, float]:
    """Compute uniqueness of generated molecules.

    Args:
        generated_molecules: List of molecule dicts
        use_positions: Whether to consider 3D positions for uniqueness
        position_tolerance: Tolerance for position matching

    Returns:
        dict with uniqueness metrics
    """
    num_molecules = len(generated_molecules)
    if num_molecules == 0:
        return {"uniqueness": 0.0, "num_unique": 0, "num_total": 0}

    # Convert molecules to hashable representations
    mol_hashes = []

    for mol_data in generated_molecules:
        atom_types = mol_data["atom_types"]

        if use_positions:
            positions = mol_data["positions"]
            # Round positions to tolerance
            rounded_pos = (positions / position_tolerance).round()
            # Create hash from atom types and positions
            mol_hash = hash((
                tuple(atom_types.cpu().numpy() if torch.is_tensor(atom_types) else atom_types),
                tuple(rounded_pos.flatten().cpu().numpy() if torch.is_tensor(rounded_pos) else rounded_pos.flatten())
            ))
        else:
            # Just use atom types (graph structure)
            mol_hash = hash(
                tuple(sorted(atom_types.cpu().numpy() if torch.is_tensor(atom_types) else atom_types))
            )

        mol_hashes.append(mol_hash)

    # Count unique hashes
    num_unique = len(set(mol_hashes))
    uniqueness = num_unique / num_molecules

    # Compute frequency distribution
    hash_counts = Counter(mol_hashes)
    max_duplicates = max(hash_counts.values())

    return {
        "uniqueness": uniqueness,
        "num_unique": num_unique,
        "num_total": num_molecules,
        "max_duplicates": max_duplicates
    }


def compute_novelty(
    generated_molecules: List[Dict],
    training_molecules: List[Dict],
    position_tolerance: float = 0.1
) -> Dict[str, float]:
    """Compute novelty of generated molecules.

    Novelty = fraction of generated molecules not in training set.

    Args:
        generated_molecules: Generated molecule dicts
        training_molecules: Training set molecule dicts
        position_tolerance: Tolerance for position matching

    Returns:
        dict with novelty metrics
    """
    if len(generated_molecules) == 0:
        return {"novelty": 0.0, "num_novel": 0, "num_total": 0}

    # Create set of training molecule hashes
    training_hashes = set()

    for mol_data in training_molecules:
        atom_types = mol_data["atom_types"]
        positions = mol_data["positions"]

        rounded_pos = (positions / position_tolerance).round()
        mol_hash = hash((
            tuple(atom_types.cpu().numpy() if torch.is_tensor(atom_types) else atom_types),
            tuple(rounded_pos.flatten().cpu().numpy() if torch.is_tensor(rounded_pos) else rounded_pos.flatten())
        ))
        training_hashes.add(mol_hash)

    # Check how many generated molecules are novel
    num_novel = 0

    for mol_data in generated_molecules:
        atom_types = mol_data["atom_types"]
        positions = mol_data["positions"]

        rounded_pos = (positions / position_tolerance).round()
        mol_hash = hash((
            tuple(atom_types.cpu().numpy() if torch.is_tensor(atom_types) else atom_types),
            tuple(rounded_pos.flatten().cpu().numpy() if torch.is_tensor(rounded_pos) else rounded_pos.flatten())
        ))

        if mol_hash not in training_hashes:
            num_novel += 1

    novelty = num_novel / len(generated_molecules)

    return {
        "novelty": novelty,
        "num_novel": num_novel,
        "num_total": len(generated_molecules)
    }


def compute_molecular_properties(
    generated_molecules: List[Dict]
) -> Dict[str, Dict[str, float]]:
    """Compute molecular properties (QED, SA, logP, etc.).

    This is a placeholder. Full implementation requires RDKit.

    Args:
        generated_molecules: List of molecule dicts

    Returns:
        dict with property statistics
    """
    # Placeholder - in practice, compute with RDKit:
    # from rdkit import Chem
    # from rdkit.Chem import QED, Descriptors, Crippen

    properties = {
        "qed": [],
        "sa_score": [],
        "logp": [],
        "molecular_weight": [],
    }

    for mol_data in generated_molecules:
        # TODO: Convert to RDKit molecule and compute properties
        # mol = positions_to_rdkit_mol(mol_data['positions'], mol_data['atom_types'])
        # if mol is not None:
        #     properties['qed'].append(QED.qed(mol))
        #     properties['sa_score'].append(calculate_sa_score(mol))
        #     properties['logp'].append(Crippen.MolLogP(mol))
        #     properties['molecular_weight'].append(Descriptors.MolWt(mol))

        # Placeholder values
        properties["qed"].append(0.5)
        properties["sa_score"].append(3.0)
        properties["logp"].append(0.0)
        properties["molecular_weight"].append(150.0)

    # Compute statistics
    stats = {}
    for prop_name, values in properties.items():
        if len(values) > 0:
            stats[prop_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }

    return stats
