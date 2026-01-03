"""Sample quality metrics for generated molecules.

Implements:
- Atom stability: Fraction of atoms with correct valence
- Molecule stability: Fraction of molecules where all atoms are stable
- Validity: Fraction of chemically valid molecules (via RDKit)
- Uniqueness: Fraction of unique molecules (via SMILES)
- Novelty: Fraction of molecules not in training set
- Molecular properties: QED, SA, logP, etc.

Based on evaluation metrics from:
- EDM (Hoogeboom et al.)
- ADiT (Facebook Research)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
import tempfile
import os

# Try to import RDKit for proper validity checking
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Try to import openbabel for xyz to mol conversion
try:
    from openbabel import openbabel
    openbabel.obErrorLog.StopLogging()
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False


# Bond length thresholds for QM9 (in Angstroms)
# Based on typical covalent bond lengths + tolerance
BOND_THRESHOLDS = {
    (1, 1): 1.0,   # H-H
    (1, 6): 1.3,   # H-C
    (1, 7): 1.2,   # H-N
    (1, 8): 1.2,   # H-O
    (1, 9): 1.1,   # H-F
    (6, 6): 1.8,   # C-C (single/double/triple)
    (6, 7): 1.7,   # C-N
    (6, 8): 1.7,   # C-O
    (6, 9): 1.6,   # C-F
    (7, 7): 1.6,   # N-N
    (7, 8): 1.6,   # N-O
    (8, 8): 1.6,   # O-O
}

# Allowed valences for each atom type in QM9
# atom_type: [allowed_valences]
ALLOWED_VALENCES = {
    1: [1],           # H
    6: [4],           # C
    7: [3],           # N (can also be 4 with positive charge, but simplified)
    8: [2],           # O
    9: [1],           # F
}

# Mapping from index to atomic number
INDEX_TO_ATOM = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}  # H, C, N, O, F
ATOM_TO_INDEX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

# Element symbols
ATOMIC_NUM_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}


def get_bond_threshold(atom1: int, atom2: int) -> float:
    """Get bond length threshold for a pair of atoms."""
    key = tuple(sorted([atom1, atom2]))
    return BOND_THRESHOLDS.get(key, 2.0)  # Default threshold


def compute_adjacency_matrix(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    use_atomic_numbers: bool = True
) -> torch.Tensor:
    """Compute adjacency matrix based on distance thresholds.

    Args:
        positions: (N, 3) atom positions
        atom_types: (N,) atom types (either indices 0-4 or atomic numbers 1,6,7,8,9)
        use_atomic_numbers: If True, atom_types are atomic numbers; else indices

    Returns:
        (N, N) adjacency matrix (1 = bonded, 0 = not bonded)
    """
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions)
    if isinstance(atom_types, np.ndarray):
        atom_types = torch.from_numpy(atom_types)

    N = positions.shape[0]
    adj = torch.zeros(N, N, dtype=torch.long)

    # Convert to atomic numbers if needed
    if not use_atomic_numbers:
        atomic_nums = torch.tensor([INDEX_TO_ATOM[int(t)] for t in atom_types])
    else:
        atomic_nums = atom_types

    # Compute pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dist = torch.norm(positions[i] - positions[j]).item()
            threshold = get_bond_threshold(int(atomic_nums[i]), int(atomic_nums[j]))

            if dist < threshold:
                adj[i, j] = 1
                adj[j, i] = 1

    return adj


def compute_atom_valences(adj: torch.Tensor) -> torch.Tensor:
    """Compute valence (number of bonds) for each atom.

    Args:
        adj: (N, N) adjacency matrix

    Returns:
        (N,) valence for each atom
    """
    return adj.sum(dim=1)


def check_atom_stability(
    atom_types: torch.Tensor,
    valences: torch.Tensor,
    use_atomic_numbers: bool = True
) -> Tuple[torch.Tensor, float]:
    """Check if each atom has a valid valence.

    Args:
        atom_types: (N,) atom types
        valences: (N,) computed valences
        use_atomic_numbers: If True, atom_types are atomic numbers

    Returns:
        (N,) boolean tensor of stable atoms, stability fraction
    """
    if isinstance(atom_types, np.ndarray):
        atom_types = torch.from_numpy(atom_types)
    if isinstance(valences, np.ndarray):
        valences = torch.from_numpy(valences)

    N = len(atom_types)
    stable = torch.zeros(N, dtype=torch.bool)

    for i in range(N):
        if use_atomic_numbers:
            atom_num = int(atom_types[i])
        else:
            atom_num = INDEX_TO_ATOM[int(atom_types[i])]

        allowed = ALLOWED_VALENCES.get(atom_num, [])
        valence = int(valences[i])

        if valence in allowed:
            stable[i] = True

    stability = stable.float().mean().item()
    return stable, stability


def compute_atom_stability(
    generated_molecules: List[Dict],
    use_atomic_numbers: bool = False
) -> Dict[str, float]:
    """Compute atom stability metrics for generated molecules.

    Args:
        generated_molecules: List of dicts with 'positions' and 'atom_types'
        use_atomic_numbers: If True, atom_types are atomic numbers (1,6,7,8,9)

    Returns:
        dict with atom stability metrics
    """
    if len(generated_molecules) == 0:
        return {"atom_stability": 0.0, "num_stable_atoms": 0, "num_total_atoms": 0}

    total_atoms = 0
    stable_atoms = 0

    for mol_data in generated_molecules:
        positions = mol_data["positions"]
        atom_types = mol_data["atom_types"]

        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions)
        if isinstance(atom_types, np.ndarray):
            atom_types = torch.from_numpy(atom_types)

        # Check for invalid positions
        if torch.isnan(positions).any() or torch.isinf(positions).any():
            total_atoms += len(atom_types)
            continue

        adj = compute_adjacency_matrix(positions, atom_types, use_atomic_numbers)
        valences = compute_atom_valences(adj)
        stable, _ = check_atom_stability(atom_types, valences, use_atomic_numbers)

        total_atoms += len(atom_types)
        stable_atoms += stable.sum().item()

    atom_stability = stable_atoms / total_atoms if total_atoms > 0 else 0.0

    return {
        "atom_stability": atom_stability,
        "num_stable_atoms": int(stable_atoms),
        "num_total_atoms": int(total_atoms)
    }


def compute_molecule_stability(
    generated_molecules: List[Dict],
    use_atomic_numbers: bool = False
) -> Dict[str, float]:
    """Compute molecule stability (all atoms stable).

    Args:
        generated_molecules: List of dicts with 'positions' and 'atom_types'
        use_atomic_numbers: If True, atom_types are atomic numbers

    Returns:
        dict with molecule stability metrics
    """
    if len(generated_molecules) == 0:
        return {"molecule_stability": 0.0, "num_stable_molecules": 0, "num_total_molecules": 0}

    num_stable = 0

    for mol_data in generated_molecules:
        positions = mol_data["positions"]
        atom_types = mol_data["atom_types"]

        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions)
        if isinstance(atom_types, np.ndarray):
            atom_types = torch.from_numpy(atom_types)

        # Check for invalid positions
        if torch.isnan(positions).any() or torch.isinf(positions).any():
            continue

        adj = compute_adjacency_matrix(positions, atom_types, use_atomic_numbers)
        valences = compute_atom_valences(adj)
        stable, _ = check_atom_stability(atom_types, valences, use_atomic_numbers)

        # Molecule is stable if ALL atoms are stable
        if stable.all():
            num_stable += 1

    molecule_stability = num_stable / len(generated_molecules)

    return {
        "molecule_stability": molecule_stability,
        "num_stable_molecules": num_stable,
        "num_total_molecules": len(generated_molecules)
    }


def mol_to_xyz_string(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    use_atomic_numbers: bool = False
) -> str:
    """Convert molecule to XYZ format string.

    Args:
        positions: (N, 3) atom positions in Angstroms
        atom_types: (N,) atom types
        use_atomic_numbers: If True, atom_types are atomic numbers

    Returns:
        XYZ format string
    """
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.cpu().numpy()

    n_atoms = len(atom_types)
    lines = [str(n_atoms), "Generated molecule"]

    for i in range(n_atoms):
        if use_atomic_numbers:
            atom_num = int(atom_types[i])
        else:
            atom_num = INDEX_TO_ATOM[int(atom_types[i])]

        symbol = ATOMIC_NUM_TO_SYMBOL.get(atom_num, 'X')
        x, y, z = positions[i]
        lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")

    return "\n".join(lines)


def xyz_to_rdkit_mol(xyz_string: str) -> Optional[Chem.Mol]:
    """Convert XYZ string to RDKit molecule using OpenBabel.

    Args:
        xyz_string: XYZ format string

    Returns:
        RDKit Mol object or None if conversion failed
    """
    if not OPENBABEL_AVAILABLE or not RDKIT_AVAILABLE:
        return None

    try:
        # Write XYZ to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write(xyz_string)
            xyz_path = f.name

        # Convert XYZ to MOL using OpenBabel
        mol_path = xyz_path.replace('.xyz', '.mol')

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "mol")

        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, xyz_path)
        obConversion.WriteFile(mol, mol_path)

        # Read MOL file with RDKit
        rdkit_mol = Chem.MolFromMolFile(mol_path, removeHs=False)

        # Clean up temp files
        os.unlink(xyz_path)
        if os.path.exists(mol_path):
            os.unlink(mol_path)

        return rdkit_mol

    except Exception as e:
        return None


def mol_to_smiles(mol_data: Dict, use_atomic_numbers: bool = False) -> Optional[str]:
    """Convert molecule dict to SMILES string.

    Args:
        mol_data: Dict with 'positions' and 'atom_types'
        use_atomic_numbers: If True, atom_types are atomic numbers

    Returns:
        SMILES string or None if conversion failed
    """
    if not RDKIT_AVAILABLE:
        return None

    positions = mol_data["positions"]
    atom_types = mol_data["atom_types"]

    # Check for invalid positions
    if isinstance(positions, torch.Tensor):
        if torch.isnan(positions).any() or torch.isinf(positions).any():
            return None

    xyz_string = mol_to_xyz_string(positions, atom_types, use_atomic_numbers)
    rdkit_mol = xyz_to_rdkit_mol(xyz_string)

    if rdkit_mol is None:
        return None

    try:
        # Get largest fragment (in case of disconnected components)
        frags = Chem.rdmolops.GetMolFrags(rdkit_mol, asMols=True)
        if len(frags) == 0:
            return None
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=True)
        return smiles
    except Exception:
        return None


def compute_validity_rdkit(
    generated_molecules: List[Dict],
    use_atomic_numbers: bool = False
) -> Dict[str, float]:
    """Compute validity using RDKit (requires openbabel).

    A molecule is valid if it can be converted to a valid SMILES.

    Args:
        generated_molecules: List of dicts with 'positions' and 'atom_types'
        use_atomic_numbers: If True, atom_types are atomic numbers

    Returns:
        dict with validity metrics and list of valid SMILES
    """
    if not RDKIT_AVAILABLE or not OPENBABEL_AVAILABLE:
        return {
            "validity": 0.0,
            "num_valid": 0,
            "num_total": len(generated_molecules),
            "valid_smiles": [],
            "error": "RDKit or OpenBabel not available"
        }

    valid_smiles = []
    for mol_data in generated_molecules:
        smiles = mol_to_smiles(mol_data, use_atomic_numbers)
        if smiles is not None:
            valid_smiles.append(smiles)

    validity = len(valid_smiles) / len(generated_molecules) if len(generated_molecules) > 0 else 0.0

    return {
        "validity": validity,
        "num_valid": len(valid_smiles),
        "num_total": len(generated_molecules),
        "valid_smiles": valid_smiles
    }


def compute_uniqueness_smiles(valid_smiles: List[str]) -> Dict[str, float]:
    """Compute uniqueness from list of valid SMILES.

    Args:
        valid_smiles: List of valid SMILES strings

    Returns:
        dict with uniqueness metrics
    """
    if len(valid_smiles) == 0:
        return {"uniqueness": 0.0, "num_unique": 0, "num_total": 0}

    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles)

    return {
        "uniqueness": uniqueness,
        "num_unique": len(unique_smiles),
        "num_total": len(valid_smiles),
        "unique_smiles": list(unique_smiles)
    }


def compute_novelty_smiles(
    valid_smiles: List[str],
    training_smiles: Set[str]
) -> Dict[str, float]:
    """Compute novelty from list of valid SMILES.

    Args:
        valid_smiles: List of valid SMILES strings
        training_smiles: Set of training SMILES strings

    Returns:
        dict with novelty metrics
    """
    if len(valid_smiles) == 0:
        return {"novelty": 0.0, "num_novel": 0, "num_total": 0}

    novel_count = sum(1 for s in valid_smiles if s not in training_smiles)
    novelty = novel_count / len(valid_smiles)

    return {
        "novelty": novelty,
        "num_novel": novel_count,
        "num_total": len(valid_smiles)
    }


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
