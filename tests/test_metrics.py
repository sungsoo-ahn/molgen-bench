"""Tests for evaluation metrics."""

import torch
import numpy as np
from src.evaluation import compute_validity, compute_uniqueness, compute_novelty


def test_validity():
    """Test validity metric."""
    molecules = [
        {
            "positions": torch.randn(5, 3),
            "atom_types": torch.tensor([1, 6, 6, 7, 8])
        }
        for _ in range(10)
    ]

    result = compute_validity(molecules)

    assert "validity" in result
    assert 0 <= result["validity"] <= 1
    assert result["num_total"] == 10


def test_uniqueness():
    """Test uniqueness metric."""
    # Create some duplicate molecules
    molecules = []

    # Add 5 unique molecules
    for i in range(5):
        molecules.append({
            "positions": torch.randn(5, 3) + i,
            "atom_types": torch.tensor([1, 6, 6, 7, 8])
        })

    # Add 5 duplicates
    for i in range(5):
        molecules.append(molecules[i])

    result = compute_uniqueness(molecules, use_positions=False)

    assert "uniqueness" in result
    # Should have ~50% uniqueness (5 unique out of 10 total)
    assert 0.4 <= result["uniqueness"] <= 0.6


def test_novelty():
    """Test novelty metric."""
    training = [
        {
            "positions": torch.randn(5, 3),
            "atom_types": torch.tensor([1, 6, 6, 7, 8])
        }
        for _ in range(5)
    ]

    # Generated molecules are different from training
    generated = [
        {
            "positions": torch.randn(5, 3) + 10,
            "atom_types": torch.tensor([1, 6, 6, 7, 8])
        }
        for _ in range(5)
    ]

    result = compute_novelty(generated, training)

    assert "novelty" in result
    assert result["novelty"] > 0.9  # Should be mostly novel


if __name__ == "__main__":
    test_validity()
    test_uniqueness()
    test_novelty()
    print("All metric tests passed!")
