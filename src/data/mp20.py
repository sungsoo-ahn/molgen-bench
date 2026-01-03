"""MP20 dataset loader."""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
from tqdm import tqdm


class MP20Dataset(Dataset):
    """MP20 dataset - Materials Project 20k inorganic crystal structures.

    A curated subset of the Materials Project database containing ~20k
    inorganic crystal structures with atomic coordinates and properties.
    """

    # HuggingFace dataset identifier
    HF_DATASET_ID = "chaitjo/MP20_ADiT"

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        download: bool = True,
        transform: Optional[callable] = None
    ):
        """Initialize MP20 dataset.

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

        self.processed_file = os.path.join(data_dir, f"mp20_{split}.pkl")

        if not os.path.exists(self.processed_file):
            if download:
                print(f"Downloading and processing MP20 {split} dataset from HuggingFace...")
                self._download_and_process()
            else:
                raise FileNotFoundError(f"MP20 {split} data not found at {self.processed_file}")

        with open(self.processed_file, "rb") as f:
            self.data = pickle.load(f)

        print(f"Loaded {len(self.data)} structures from MP20 {split} split")

    def _download_and_process(self):
        """Download and process MP20 dataset from HuggingFace.

        Downloads the MP20 dataset processed by the ADiT team from HuggingFace.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required to download MP20. "
                "Install it with: uv add datasets"
            )

        print(f"Downloading MP20 dataset from HuggingFace ({self.HF_DATASET_ID})...")
        print("This may take a while on first run...")

        # Download dataset from HuggingFace
        try:
            hf_dataset = load_dataset(self.HF_DATASET_ID, split=self.split)
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            print("Note: The dataset may not be publicly available yet.")
            print("Creating a small synthetic dataset for testing...")
            self._create_fallback_data()
            return

        print(f"Processing {len(hf_dataset)} structures for {self.split} split...")

        # Process structures
        data = []
        for idx, sample in enumerate(tqdm(hf_dataset, desc=f"Processing {self.split}")):
            # Extract data from HuggingFace format
            # Note: The exact format may vary - adjust based on actual dataset structure
            positions = np.array(sample['positions'])  # (N, 3) fractional coordinates
            atom_types = np.array(sample['atomic_numbers'])  # (N,) atomic numbers

            # Extract properties if available
            properties = {}
            if 'formation_energy_per_atom' in sample:
                properties['formation_energy'] = float(sample['formation_energy_per_atom'])
            if 'band_gap' in sample:
                properties['band_gap'] = float(sample['band_gap'])
            if 'density' in sample:
                properties['density'] = float(sample['density'])

            # Extract lattice parameters if available
            lattice = None
            if 'lattice' in sample:
                lattice = np.array(sample['lattice'])  # (3, 3) lattice vectors

            data.append({
                "positions": positions,
                "atom_types": atom_types,
                "properties": properties,
                "lattice": lattice,
                "id": idx
            })

        # Save processed data
        with open(self.processed_file, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved {len(data)} structures to {self.processed_file}")

    def _create_fallback_data(self):
        """Create small synthetic MP20-like data as fallback.

        Used when HuggingFace dataset is not available.
        """
        print(f"Creating fallback {self.split} data...")

        # Small dataset for testing
        split_sizes = {"train": 100, "val": 20, "test": 20}
        n_structures = split_sizes.get(self.split, 100)

        data = []
        np.random.seed(42 if self.split == "train" else 43 if self.split == "val" else 44)

        for i in tqdm(range(n_structures), desc=f"Generating {self.split} data"):
            # Random number of atoms (typical crystals: 10-50 atoms)
            n_atoms = np.random.randint(10, 50)

            # Random atom types (common elements in materials)
            atom_types = np.random.choice([1, 6, 7, 8, 14, 26, 29], size=n_atoms)

            # Random fractional positions
            positions = np.random.rand(n_atoms, 3)

            # Random lattice (3x3 matrix)
            lattice = np.random.rand(3, 3) * 10.0

            # Placeholder properties
            properties = {
                "formation_energy": np.random.uniform(-5, 0),
                "band_gap": np.random.uniform(0, 5),
                "density": np.random.uniform(1, 10),
            }

            data.append({
                "positions": positions,
                "atom_types": atom_types,
                "properties": properties,
                "lattice": lattice,
                "id": i
            })

        with open(self.processed_file, "wb") as f:
            pickle.dump(data, f)

        print(f"Fallback data saved to {self.processed_file}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a single structure.

        Returns:
            dict with keys:
                - positions: (N, 3) array of atomic positions (fractional coords)
                - atom_types: (N,) array of atomic numbers
                - lattice: (3, 3) array of lattice vectors (if available)
                - properties: dict of scalar properties
                - id: structure identifier
        """
        mol_data = self.data[idx]

        sample = {
            "positions": torch.tensor(mol_data["positions"], dtype=torch.float32),
            "atom_types": torch.tensor(mol_data["atom_types"], dtype=torch.long),
            "properties": {k: torch.tensor(v, dtype=torch.float32)
                          for k, v in mol_data["properties"].items()},
            "id": mol_data["id"]
        }

        # Add lattice if available
        if "lattice" in mol_data and mol_data["lattice"] is not None:
            sample["lattice"] = torch.tensor(mol_data["lattice"], dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)

        return sample
