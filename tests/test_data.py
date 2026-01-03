"""Tests for data loading."""

import pytest
import torch
from src.data import QM9Dataset, collate_molecular_data


def test_qm9_dataset():
    """Test QM9 dataset loading."""
    dataset = QM9Dataset(data_dir="./data/test", split="train", download=True)

    assert len(dataset) > 0

    sample = dataset[0]
    assert "positions" in sample
    assert "atom_types" in sample
    assert "properties" in sample

    assert sample["positions"].shape[1] == 3
    assert len(sample["atom_types"]) == len(sample["positions"])


def test_collate():
    """Test batching."""
    dataset = QM9Dataset(data_dir="./data/test", split="train", download=True)

    batch_list = [dataset[i] for i in range(4)]
    batch = collate_molecular_data(batch_list)

    assert batch.num_molecules == 4
    assert batch.positions.shape[1] == 3
    assert batch.edge_index.shape[0] == 2


if __name__ == "__main__":
    test_qm9_dataset()
    test_collate()
    print("All data tests passed!")
