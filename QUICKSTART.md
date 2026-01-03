# Quick Start Guide

This guide will help you get started with MolGen-Bench.

## Installation

```bash
# Clone or navigate to the repository
cd molgen-bench

# Install dependencies
pip install -e .

# Or install specific requirements
pip install -r requirements.txt
```

## Running Your First Experiment

### 1. Basic Training

Train a flow matching model with GNN architecture on QM9:

```bash
python experiments/benchmarks.py --config configs/default.yaml
```

### 2. Scaling Law Experiments

Run experiments to analyze scaling behavior:

```bash
python experiments/scaling_experiments.py --config configs/default.yaml
```

This will:
- Train models of different sizes
- Train on different dataset fractions
- Track loss vs model size and dataset size
- Generate scaling law plots

### 3. Custom Architectures

Try different architectures by specifying different configs:

```bash
# GNN (default)
python experiments/benchmarks.py --config configs/architectures/gnn.yaml

# Transformer
python experiments/benchmarks.py --config configs/architectures/transformer.yaml

# Pairformer
python experiments/benchmarks.py --config configs/architectures/pairformer.yaml
```

### 4. Different Generative Models

```bash
# Flow Matching (default)
python experiments/benchmarks.py --config configs/generative/flow_matching.yaml

# Diffusion (skeleton - needs implementation)
# python experiments/benchmarks.py --config configs/generative/diffusion.yaml

# Stochastic Interpolant (skeleton - needs implementation)
# python experiments/benchmarks.py --config configs/generative/stochastic_interpolant.yaml
```

## Understanding the Output

After training, you'll find:

```
outputs/
└── your_experiment_name/
    ├── checkpoints/
    │   ├── best_model.pt
    │   ├── final_model.pt
    │   └── checkpoint_epoch_*.pt
    └── scaling_logs/
        └── scaling_data.json
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_data.py
pytest tests/test_models.py
pytest tests/test_metrics.py

# Or run tests directly
python tests/test_data.py
python tests/test_models.py
python tests/test_metrics.py
```

## Project Structure

```
molgen-bench/
├── configs/              # Configuration files
│   ├── default.yaml      # Default config
│   ├── architectures/    # Architecture-specific configs
│   └── generative/       # Generative model configs
├── data/                 # Dataset loaders
│   ├── qm9.py           # QM9 dataset
│   ├── mp20.py          # MP20 dataset (template)
│   └── utils.py         # Data utilities
├── models/              # Model implementations
│   ├── architectures/   # GNN, Transformer, Pairformer
│   └── generative/      # Flow matching, diffusion, etc.
├── training/            # Training infrastructure
│   ├── trainer.py       # Main trainer
│   └── scaling_laws.py  # Scaling law tracking
├── evaluation/          # Evaluation metrics
│   ├── metrics.py       # Main interface
│   ├── sample_quality.py      # Validity, uniqueness, novelty
│   ├── distribution_matching.py  # Wasserstein, coverage
│   └── visualization.py       # Plotting utilities
├── experiments/         # Experiment scripts
│   ├── benchmarks.py           # Main training script
│   └── scaling_experiments.py  # Scaling law experiments
└── tests/              # Unit tests
```

## Next Steps

### 1. Implement Real Data Loading

The current QM9 and MP20 loaders create synthetic data. To use real data:

1. Download QM9 from: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
2. Implement proper data parsing in `data/qm9.py`
3. Update `_create_synthetic_data()` method

### 2. Complete Generative Model Implementations

- `models/generative/diffusion.py` - Implement DDPM/DDIM sampling
- `models/generative/stochastic_interpolant.py` - Implement interpolant training and sampling

### 3. Add RDKit Integration

For proper molecular validity checking:

```python
# Install RDKit
conda install -c conda-forge rdkit

# Update evaluation/sample_quality.py to use RDKit
```

### 4. Implement Full Transformer

Current transformer is a skeleton. Add:
- 3D distance bias in attention
- Proper positional encoding for molecules
- Efficient attention mechanisms

### 5. Run Large-Scale Experiments

Once everything is working:
- Train on full QM9 dataset
- Sweep over architectures and hyperparameters
- Analyze scaling laws
- Generate benchmark results

## Configuration Guide

Key configuration parameters in `configs/default.yaml`:

```yaml
# Data
data:
  dataset: qm9  # or mp20
  batch_size: 128

# Architecture
architecture:
  type: gnn  # gnn, transformer, pairformer
  hidden_dim: 256
  num_layers: 6

# Generative Model
generative:
  type: flow_matching  # flow_matching, diffusion, stochastic_interpolant
  time_steps: 1000

# Training
training:
  epochs: 500
  learning_rate: 1e-4
  track_scaling: true  # Enable scaling law tracking
```

## Tips

1. **Start small**: Test with small models and datasets first
2. **Monitor GPU memory**: Use mixed precision if running out of memory
3. **Track experiments**: Enable W&B logging in config for better tracking
4. **Scaling laws**: Run scaling experiments early to understand optimal model/data size
5. **Validation**: Always check samples are chemically valid

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` in config
- Enable `mixed_precision: true`
- Use smaller model (reduce `hidden_dim` or `num_layers`)

**Training slow?**
- Enable mixed precision
- Increase batch size if memory allows
- Use smaller evaluation intervals
- Enable `torch.compile` (PyTorch 2.0+)

**Poor sample quality?**
- Train longer
- Increase model size
- Check if loss is decreasing
- Verify data loading is correct

## Getting Help

- Check the README for project overview
- Read the code - it's well-commented!
- Run tests to verify setup
- Open an issue if you find bugs
