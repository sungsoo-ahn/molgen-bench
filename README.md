# MolGen-Bench: 3D Molecular Generative Model Benchmark

A comprehensive benchmark for evaluating and comparing different neural network architectures and generative modeling approaches for 3D molecular generation.

## Overview

This benchmark systematically compares:
- **Architectures**: GNNs, Transformers, Pairformers
- **Generative Models**: Flow Matching, Diffusion Models, Stochastic Interpolants
- **Datasets**: QM9, MP20
- **Focus**: Scaling laws, sample quality, distribution matching

## Workflow

This repository is designed for AI-assisted development workflows:

1. **Start a session**: Read `docs/start.txt` and referenced documentation
2. **Do research**: Run experiments, write code, analyze results
3. **End a session**: Complete tasks in `docs/closing_tasks.md`

## Project Structure

| Folder | Purpose |
|--------|---------|
| `src/` | All Python source code |
| `src/scripts/` | Entry point scripts (benchmarks.py, scaling_experiments.py) |
| `src/data/` | Dataset loading and preprocessing |
| `src/models/` | Neural architectures and generative models |
| `src/models/architectures/` | GNN, Transformer, Pairformer |
| `src/models/generative/` | Flow matching, diffusion, stochastic interpolants |
| `src/training/` | Training loops and scaling law tracking |
| `src/evaluation/` | Metrics and evaluation tools |
| `configs/` | YAML configuration files |
| `configs/architectures/` | Architecture-specific configs |
| `configs/generative/` | Generative model configs |
| `scripts/` | Bash wrappers for running experiments |
| `data/` | Experiment outputs (gitignored) |
| `scratch/` | Temporary work directory (gitignored) |
| `resources/` | Papers, references (gitignored) |
| `docs/` | Documentation and logs |
| `tests/` | Unit tests |

## Setup

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

## Running Experiments

### Benchmarks
```bash
bash scripts/run_benchmarks.sh
# or directly:
uv run python src/scripts/benchmarks.py configs/default.yaml
```

### Scaling Experiments
```bash
bash scripts/run_scaling.sh
# or directly:
uv run python src/scripts/scaling_experiments.py configs/default.yaml
```

### Tests
```bash
bash scripts/run_tests.sh
# or directly:
uv run pytest tests/ -v
```

## Evaluation Metrics

### Sample Quality
- Validity (chemical validity of generated molecules)
- Uniqueness (diversity of samples)
- Novelty (compared to training set)
- Molecular properties (QED, SA, logP)

### Distribution Matching
- Wasserstein distance
- Fréchet ChemNet Distance (FCD)
- Coverage metrics
- Property distribution comparisons

### Scaling Behavior
- Loss vs model size
- Loss vs dataset size
- Compute-optimal scaling coefficients
- Inference time vs model size

## Key Files

- `CLAUDE.md` - Development guidelines for AI agents
- `docs/repo_usage.md` - Detailed development practices
- `docs/research_context.md` - Current research state and context
- `pyproject.toml` - Project dependencies and configuration

## Development

See `CLAUDE.md` for development conventions and `docs/repo_usage.md` for detailed guidelines.

## Citation

```bibtex
@software{molgen_bench,
  title={MolGen-Bench: A Benchmark for 3D Molecular Generative Models},
  author={Your Name},
  year={2026}
}
```
