# Research Context: MolGen-Bench

## Overview
Benchmark suite for evaluating 3D molecular generative models across different neural architectures and generative modeling approaches.

## Current Status
- Repository restructured to follow research-template-main conventions
- Core implementations complete for all major components
- Ready for systematic benchmarking and scaling law experiments

## Key Components

### Datasets
- **QM9Dataset** (`src/data/qm9.py`): Small organic molecules dataset (~130k molecules)
  - 3D coordinates, atom types, molecular properties
  - Standard train/val/test splits
  - Automatic download and preprocessing

- **MP20Dataset** (`src/data/mp20.py`): Materials Project structures (~20k)
  - Inorganic crystal structures
  - Materials properties and metadata

- **Data utilities** (`src/data/utils.py`): Custom collation for batching variable-size molecules

### Architectures

Located in `src/models/architectures/`:

- **GNN** (`gnn.py`): Graph Neural Network backbone
  - Message passing on molecular graphs
  - Edge features and distance-based interactions
  - Configurable depth and hidden dimensions

- **Transformer** (`transformer.py`): Attention-based architecture
  - Self-attention over atoms
  - Positional encodings for 3D coordinates
  - Multi-head attention mechanism

- **Pairformer** (`pairformer.py`): Pair representation model
  - Atom and pair representations
  - Triangle updates and attention
  - Inspired by AlphaFold architecture

### Generative Models

Located in `src/models/generative/`:

- **FlowMatching** (`flow_matching.py`): Continuous normalizing flows
  - Optimal transport paths
  - Conditional flow matching loss
  - Efficient sampling via ODE solver

- **Diffusion** (`diffusion.py`): Denoising diffusion probabilistic models
  - Forward and reverse diffusion processes
  - Noise schedules (linear, cosine)
  - DDPM/DDIM sampling

- **StochasticInterpolant** (`stochastic_interpolant.py`): Generalized interpolant framework
  - Flexible interpolation between distributions
  - Unified framework for flows and diffusion
  - Configurable coupling and scheduling

### Training Infrastructure

Located in `src/training/`:

- **Trainer** (`trainer.py`): Main training loop
  - Supports all architecture × generative model combinations
  - EMA, gradient clipping, warmup scheduling
  - Logging to wandb and local files
  - Checkpointing and resuming

- **ScalingLawTracker** (`scaling_laws.py`): Scaling law analysis
  - Track loss vs compute, model size, dataset size
  - Power law fitting
  - Visualization and analysis tools

### Evaluation Metrics

Located in `src/evaluation/`:

- **Sample Quality** (`sample_quality.py`):
  - Validity: Chemical validity of generated molecules
  - Uniqueness: Diversity of generated samples
  - Novelty: Comparison to training set
  - Molecular properties (QED, SA, logP)

- **Distribution Matching** (`distribution_matching.py`):
  - Wasserstein distance between distributions
  - Fréchet ChemNet Distance (FCD)
  - Property distribution comparisons
  - Coverage metrics

- **Visualization** (`visualization.py`):
  - Molecular structure plots
  - Property distribution comparisons
  - Training curves and metrics

## Experiments

### Benchmarks (`src/scripts/benchmarks.py`)
Main training and evaluation pipeline supporting all architecture × generative model combinations.

**Key Features:**
- Systematic comparison of 9 combinations (3 architectures × 3 generative models)
- Full training with evaluation on val/test sets
- Comprehensive metrics logging
- Configurable via YAML files

**Usage:**
```bash
bash scripts/run_benchmarks.sh configs/default.yaml
```

### Scaling Experiments (`src/scripts/scaling_experiments.py`)
Studies scaling laws across multiple dimensions:

**Experiments:**
- Model size scaling (vary number of layers, hidden dimensions)
- Dataset size scaling (subsample training data)
- Training compute scaling (vary number of epochs)

**Usage:**
```bash
bash scripts/run_scaling.sh configs/default.yaml
```

## Configuration

Configs are organized in `configs/`:

- **`configs/default.yaml`**: Default training configuration
  - Dataset selection (qm9/mp20)
  - Architecture and generative model types
  - Training hyperparameters
  - Evaluation metrics
  - Output directories

- **`configs/architectures/`**: Architecture-specific settings
  - `gnn.yaml`: GNN-specific parameters
  - `transformer.yaml`: Transformer configuration
  - `pairformer.yaml`: Pairformer settings

- **`configs/generative/`**: Generative model settings
  - `flow_matching.yaml`: Flow matching parameters
  - `diffusion.yaml`: Diffusion model configuration
  - `stochastic_interpolant.yaml`: Interpolant settings

## Research Questions

1. **Architecture Comparison**: Which neural architecture (GNN, Transformer, Pairformer) performs best for 3D molecular generation?

2. **Generative Model Comparison**: How do Flow Matching, Diffusion, and Stochastic Interpolants compare in terms of sample quality and training efficiency?

3. **Scaling Laws**: What are the compute-optimal scaling coefficients for molecular generative models?

4. **Transfer Learning**: Do models trained on QM9 transfer well to MP20 and vice versa?

## Next Steps

- [ ] Run comprehensive benchmarks across all architecture × generative model combinations
- [ ] Analyze scaling laws to identify compute-optimal configurations
- [ ] Investigate transfer learning between QM9 and MP20
- [ ] Optimize sampling procedures for better quality-speed tradeoffs
- [ ] Add support for conditional generation (property-guided generation)
- [ ] Implement multi-GPU training for larger models

## Development Workflow

See `CLAUDE.md` for development guidelines and `docs/repo_usage.md` for detailed practices.

**Key Conventions:**
- All code in `src/`
- Experiments run via bash scripts in `scripts/`
- Configs in `configs/` with `output_dir` specified
- Results saved to `data/` (gitignored)
- Use `uv` for package management
