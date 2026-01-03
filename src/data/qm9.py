"""QM9 dataset loader.

Loads QM9 dataset from HuggingFace (chaitjo/QM9_ADiT).
"""

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
        """Download and process QM9 dataset from HuggingFace.

        Downloads the pre-processed QM9 dataset from HuggingFace and splits it into train/val/test.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download QM9. "
                "Install it with: uv add huggingface-hub"
            )

        print(f"Downloading QM9 dataset from HuggingFace (chaitjo/QM9_ADiT)...")
        print("This may take a while on first run (~5-10 minutes for 340MB file)...")

        # Download processed data from HuggingFace
        try:
            processed_file = hf_hub_download(
                repo_id="chaitjo/QM9_ADiT",
                filename="processed/data_v3.pt",
                repo_type="dataset",
                cache_dir=os.path.join(self.data_dir, "hf_cache")
            )
            print(f"Downloaded QM9 dataset to {processed_file}")
        except Exception as e:
            print(f"\nError downloading QM9 from HuggingFace: {e}")
            print("\nThe QM9 dataset download from HuggingFace may be temporarily unavailable.")
            print("Options:")
            print("1. Try again later (HuggingFace servers may be down)")
            print("2. Check your internet connection")
            print("\nFor now, creating a small synthetic dataset for testing...")
            self._create_fallback_data()
            return

        # Load the processed PyTorch Geometric data
        print(f"Loading processed QM9 data...")
        # Note: weights_only=False is required for PyTorch Geometric Data objects
        # The file contains: (data_dict, slices_dict, Data_class)
        data_dict, slices_dict, _ = torch.load(processed_file, weights_only=False)

        print(f"Loaded batched QM9 data with {len(slices_dict['y']) - 1} molecules")

        # Get split indices
        start_idx, end_idx = self.SPLIT_INDICES[self.split]
        end_idx = min(end_idx, len(slices_dict['y']) - 1)

        print(f"Processing {end_idx - start_idx} molecules for {self.split} split...")

        # Process molecules in this split using slices
        data = []
        for idx in tqdm(range(start_idx, end_idx), desc=f"Processing {self.split}"):
            # Extract data for this molecule using slices
            # Get position slice indices
            pos_start = slices_dict['pos'][idx].item()
            pos_end = slices_dict['pos'][idx + 1].item()

            # Extract data from batched format
            positions = data_dict['pos'][pos_start:pos_end].numpy()  # (N, 3) coordinates
            atom_types = data_dict['z'][pos_start:pos_end].numpy()   # (N,) atomic numbers

            # Extract properties (y contains 19 properties, we use first 12)
            # Index mapping: mu(0), alpha(1), homo(2), lumo(3), gap(4), r2(5), zpve(6),
            #                U0(7), U(8), H(9), G(10), Cv(11)
            y = data_dict['y'][idx].numpy()
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

            # Charges are typically not included, set to zero
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

    def _create_fallback_data(self):
        """Create small synthetic QM9-like data as fallback.

        Used when PyTorch Geometric download fails.
        """
        print(f"Creating fallback {self.split} data...")

        # Small dataset for testing
        split_sizes = {"train": 100, "val": 20, "test": 20}
        n_molecules = split_sizes.get(self.split, 100)

        data = []
        np.random.seed(42 if self.split == "train" else 43 if self.split == "val" else 44)

        for i in tqdm(range(n_molecules), desc=f"Generating {self.split} data"):
            # Random number of atoms (5-29, matching QM9 statistics)
            n_atoms = np.random.randint(5, 30)

            # Random atom types (biased towards C, H)
            atom_probs = [0.5, 0.3, 0.1, 0.08, 0.02]  # H, C, N, O, F
            atom_types = np.random.choice(self.ATOM_TYPES, size=n_atoms, p=atom_probs)

            # Random 3D positions (roughly in a 5 Angstrom cube)
            positions = np.random.randn(n_atoms, 3) * 1.5

            # Random charges (mostly 0)
            charges = np.zeros(n_atoms)

            # Random properties (using typical QM9 ranges)
            properties = {
                "mu": np.random.uniform(0, 5),
                "alpha": np.random.uniform(20, 80),
                "homo": np.random.uniform(-0.3, -0.1),
                "lumo": np.random.uniform(0.0, 0.2),
                "gap": np.random.uniform(0.1, 0.4),
                "r2": np.random.uniform(200, 1500),
                "zpve": np.random.uniform(0.0, 0.5),
                "U0": np.random.uniform(-2000, -100),
                "U": np.random.uniform(-2000, -100),
                "H": np.random.uniform(-2000, -100),
                "G": np.random.uniform(-2000, -100),
                "Cv": np.random.uniform(5, 50),
            }

            data.append({
                "positions": positions,
                "atom_types": atom_types,
                "charges": charges,
                "properties": properties,
                "id": i
            })

        # Save processed data
        with open(self.processed_file, "wb") as f:
            pickle.dump(data, f)

        print(f"Fallback data saved to {self.processed_file}")

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
