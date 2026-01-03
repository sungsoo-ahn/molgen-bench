"""Tests for models."""

import pytest
import torch
from src.models.architectures import GNN, Transformer, Pairformer
from src.models.generative import FlowMatching


def test_gnn():
    """Test GNN architecture."""
    model = GNN(hidden_dim=64, num_layers=3)

    # Test forward pass
    N = 10
    atom_types = torch.randint(0, 5, (N,))
    positions = torch.randn(N, 3)
    edge_index = torch.combinations(torch.arange(N), r=2).T

    output = model(atom_types, positions, edge_index, return_features=True)

    assert output.shape == (N, 64)


def test_transformer():
    """Test Transformer architecture."""
    model = Transformer(hidden_dim=128, num_layers=2, num_heads=4)

    N = 10
    atom_types = torch.randint(0, 5, (N,))
    positions = torch.randn(N, 3)

    output = model(atom_types, positions, return_features=True)

    assert output.shape == (N, 128)


def test_flow_matching():
    """Test Flow Matching model."""
    backbone = GNN(hidden_dim=64, num_layers=2)
    model = FlowMatching(backbone=backbone, time_steps=100)

    N = 10
    atom_types = torch.randint(0, 5, (N,))
    positions = torch.randn(N, 3)
    edge_index = torch.combinations(torch.arange(N), r=2).T

    loss, info = model(positions, atom_types, edge_index)

    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


if __name__ == "__main__":
    test_gnn()
    test_transformer()
    test_flow_matching()
    print("All model tests passed!")
