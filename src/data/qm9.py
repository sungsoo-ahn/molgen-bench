"""QM9 dataset loader."""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple
from tqdm import tqdm


class QM9Dataset(Dataset):
    """QM9 dataset of small organic molecules.

    Contains ~134k small organic molecules with up to 9 heavy atoms (C, N, O, F)
    with 3D coordinates and various quantum chemical properties.

    Properties include:
        - mu: Dipole moment
        - alpha: Isotropic polarizability
        - homo: HOMO energy
        - lumo: LUMO energy
        - gap: HOMO-LUMO gap
        - r2: Electronic spatial extent
        - zpve: Zero point vibrational energy
        - U0, U, H, G: Internal energies and enthalpies
        - Cv: Heat capacity
    """

    # Atomic numbers for QM9 atoms (H, C, N, O, F)
    ATOM_TYPES = [1, 6, 7, 8, 9]

    # Property names and their indices in the QM9 target tensor
    PROPERTY_NAMES = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
        'U0', 'U', 'H', 'G', 'Cv'
    ]

    # Default train/val/test split sizes (following standard QM9 splits)
    SPLIT_INDICES = {
        'train': (0, 100000),
        'val': (100000, 110000),
        'test': (110000, 130831)
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        download: bool = True,
        transform: Optional[callable] = None
    ):
        """Initialize QM9 dataset.

        Args:
            data_dir: Directory to store/load data
            split: One of 'train', 'val', 'test'
            download: Whether to download if not present
            transform: Optional transform to apply
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        os.makedirs(data_dir, exist_ok=True)

        # Check if processed data exists
        self.processed_file = os.path.join(data_dir, f"qm9_{split}.pkl")

        if not os.path.exists(self.processed_file):
            if download:
                print(f"Downloading and processing QM9 {split} dataset...")
                self._download_and_process()
            else:
                raise FileNotFoundError(f"QM9 {split} data not found at {self.processed_file}")

        # Load data
        with open(self.processed_file, "rb") as f:
            self.data = pickle.load(f)

        print(f"Loaded {len(self.data)} molecules from QM9 {split} split")

    def _download_and_process(self):
        """Download and process QM9 dataset from PyTorch Geometric.

        Downloads the full QM9 dataset and splits it into train/val/test.
        """
        try:
            from torch_geometric.datasets import QM9 as PyGQM9
        except ImportError:
            raise ImportError(
                "PyTorch Geometric is required to download QM9. "
                "Install it with: uv add torch-geometric"
            )

        print(f"Downloading QM9 dataset using PyTorch Geometric...")
        print("This may take a while on first run (~30 minutes)...")

        # Download full QM9 dataset to a temp directory
        raw_dir = os.path.join(self.data_dir, "raw")
        pyg_dataset = PyGQM9(root=raw_dir)

        print(f"Processing {len(pyg_dataset)} molecules for {self.split} split...")

        # Get split indices
        start_idx, end_idx = self.SPLIT_INDICES[self.split]

        # Process molecules in this split
        data = []
        for idx in tqdm(range(start_idx, end_idx), desc=f"Processing {self.split}"):
            if idx >= len(pyg_dataset):
                break

            mol = pyg_dataset[idx]

            # Extract data from PyG format
            positions = mol.pos.numpy()  # (N, 3) coordinates
            atom_types = mol.z.numpy()   # (N,) atomic numbers

            # Extract properties (y contains 19 properties, we use first 12)
            # Index mapping: mu(0), alpha(1), homo(2), lumo(3), gap(4), r2(5), zpve(6),
            #                U0(7), U(8), H(9), G(10), Cv(11)
            y = mol.y.squeeze().numpy()
            properties = {
                "mu": float(y[0]),
                "alpha": float(y[1]),
                "homo": float(y[2]),
                "lumo": float(y[3]),
                "gap": float(y[4]),
                "r2": float(y[5]),
                "zpve": float(y[6]),
                "U0": float(y[7]),
                "U": float(y[8]),
                "H": float(y[9]),
                "G": float(y[10]),
                "Cv": float(y[11]),
            }

            # Get formal charges if available
            if hasattr(mol, 'charges'):
                charges = mol.charges.numpy()
            else:
                charges = np.zeros(len(atom_types))

            data.append({
                "positions": positions,
                "atom_types": atom_types,
                "charges": charges,
                "properties": properties,
                "id": idx
            })

        # Save processed data
        with open(self.processed_file, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved {len(data)} molecules to {self.processed_file}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a single molecule.

        Returns:
            dict with keys:
                - positions: (N, 3) array
                - atom_types: (N,) array
                - charges: (N,) array
                - properties: dict of scalar properties
                - id: molecule identifier
        """
        mol_data = self.data[idx]

        # Convert to tensors
        sample = {
            "positions": torch.tensor(mol_data["positions"], dtype=torch.float32),
            "atom_types": torch.tensor(mol_data["atom_types"], dtype=torch.long),
            "charges": torch.tensor(mol_data["charges"], dtype=torch.float32),
            "properties": {k: torch.tensor(v, dtype=torch.float32)
                          for k, v in mol_data["properties"].items()},
            "id": mol_data["id"]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_statistics(self) -> dict:
        """Compute dataset statistics for normalization.

        Returns:
            dict with mean and std for properties
        """
        all_properties = {name: [] for name in self.PROPERTY_NAMES}

        for mol_data in self.data:
            for name in self.PROPERTY_NAMES:
                all_properties[name].append(mol_data["properties"][name])

        stats = {}
        for name in self.PROPERTY_NAMES:
            values = np.array(all_properties[name])
            stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }

        return stats
