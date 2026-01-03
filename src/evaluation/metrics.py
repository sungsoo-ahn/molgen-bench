"""Main evaluation interface."""

from typing import List, Dict, Optional
from .sample_quality import (
    compute_validity,
    compute_uniqueness,
    compute_novelty,
    compute_molecular_properties,
    compute_atom_stability,
    compute_molecule_stability,
    compute_validity_rdkit,
    compute_uniqueness_smiles,
    compute_novelty_smiles,
    mol_to_smiles,
    RDKIT_AVAILABLE,
    OPENBABEL_AVAILABLE,
)
from .distribution_matching import (
    compute_wasserstein_distance,
    compute_property_distributions,
    compute_coverage,
    compute_fcd
)


def compute_all_metrics(
    generated_molecules: List[Dict],
    reference_molecules: Optional[List[Dict]] = None,
    training_molecules: Optional[List[Dict]] = None,
    metrics: Optional[List[str]] = None
) -> Dict:
    """Compute all evaluation metrics.

    Args:
        generated_molecules: List of generated molecule dicts
        reference_molecules: List of reference molecule dicts (e.g., test set)
        training_molecules: List of training molecule dicts (for novelty)
        metrics: List of metric names to compute (None = all)

    Returns:
        dict with all computed metrics
    """
    if metrics is None:
        metrics = [
            "atom_stability",
            "molecule_stability",
            "validity",
            "uniqueness",
            "novelty",
            "properties",
            "wasserstein",
            "distribution",
            "coverage"
        ]

    results = {}

    # Sample quality metrics
    if "atom_stability" in metrics:
        results["atom_stability"] = compute_atom_stability(generated_molecules)

    if "molecule_stability" in metrics:
        results["molecule_stability"] = compute_molecule_stability(generated_molecules)

    if "validity" in metrics:
        results["validity"] = compute_validity(generated_molecules)

    if "uniqueness" in metrics:
        results["uniqueness"] = compute_uniqueness(generated_molecules)

    if "novelty" in metrics and training_molecules is not None:
        results["novelty"] = compute_novelty(generated_molecules, training_molecules)

    if "properties" in metrics:
        results["properties"] = compute_molecular_properties(generated_molecules)

    # Distribution matching metrics
    if reference_molecules is not None:
        if "wasserstein" in metrics:
            results["wasserstein"] = compute_wasserstein_distance(
                generated_molecules, reference_molecules
            )

        if "distribution" in metrics:
            results["property_distributions"] = compute_property_distributions(
                generated_molecules, reference_molecules
            )

        if "coverage" in metrics:
            results["coverage"] = compute_coverage(
                generated_molecules, reference_molecules
            )

        if "fcd" in metrics:
            results["fcd"] = compute_fcd(generated_molecules, reference_molecules)

    return results


def print_metrics_summary(metrics: Dict):
    """Print a formatted summary of metrics.

    Args:
        metrics: Dict of computed metrics
    """
    print("\n" + "="*60)
    print("Evaluation Metrics Summary")
    print("="*60)

    # Atom Stability
    if "atom_stability" in metrics:
        a = metrics["atom_stability"]
        print(f"\nAtom Stability: {a['atom_stability']:.3f} ({a['num_stable_atoms']}/{a['num_total_atoms']})")

    # Molecule Stability
    if "molecule_stability" in metrics:
        m = metrics["molecule_stability"]
        print(f"Molecule Stability: {m['molecule_stability']:.3f} ({m['num_stable_molecules']}/{m['num_total_molecules']})")

    # Validity
    if "validity" in metrics:
        v = metrics["validity"]
        print(f"Validity: {v['validity']:.3f} ({v['num_valid']}/{v['num_total']})")

    # Uniqueness
    if "uniqueness" in metrics:
        u = metrics["uniqueness"]
        print(f"Uniqueness: {u['uniqueness']:.3f} ({u['num_unique']}/{u['num_total']})")

    # Novelty
    if "novelty" in metrics:
        n = metrics["novelty"]
        print(f"Novelty: {n['novelty']:.3f} ({n['num_novel']}/{n['num_total']})")

    # Properties
    if "properties" in metrics:
        print("\nMolecular Properties:")
        for prop_name, stats in metrics["properties"].items():
            print(f"  {prop_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    # Wasserstein
    if "wasserstein" in metrics:
        print("\nWasserstein Distances:")
        for key, value in metrics["wasserstein"].items():
            if value is not None:
                print(f"  {key}: {value:.3f}")

    # Coverage
    if "coverage" in metrics:
        c = metrics["coverage"]
        print(f"\nCoverage: {c['coverage']:.3f} ({c['num_covered']}/{c['num_reference']})")

    print("="*60 + "\n")
