# Dataset Download Scripts

This directory contains scripts to download and process the QM9 and MP20 datasets based on the [all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer) repository.

## Available Datasets

### QM9
- **Source**: PyTorch Geometric (`torch_geometric.datasets.QM9`)
- **Size**: ~134k small organic molecules
- **Properties**: Quantum chemical properties (dipole moment, HOMO-LUMO gap, etc.)
- **Splits**: train (100k), val (10k), test (~31k)

### MP20
- **Source**: HuggingFace (`chaitjo/MP20_ADiT`)
- **Size**: ~20k inorganic crystal structures
- **Properties**: Formation energy, band gap, density
- **Splits**: train (80%), val (10%), test (10%)

## Usage

### Download All Datasets

```bash
bash scripts/data_generation/download_all.sh [DATA_DIR]
```

Default data directory is `./data/downloaded`.

### Download Individual Datasets

**QM9 only:**
```bash
bash scripts/data_generation/download_qm9.sh [DATA_DIR]
```

**MP20 only:**
```bash
bash scripts/data_generation/download_mp20.sh [DATA_DIR]
```

## Testing

Test dataset loading:

```bash
# Test QM9
uv run python src/scripts/test_datasets.py --dataset qm9

# Test MP20
uv run python src/scripts/test_datasets.py --dataset mp20

# Test both
uv run python src/scripts/test_datasets.py --dataset all
```

## Download Time

- **First download**: ~30-40 minutes (downloads and processes data)
- **Subsequent runs**: Instant (uses cached processed files)

## Data Structure

After downloading, data is organized as:

```
data/downloaded/
├── qm9/
│   ├── raw/              # PyTorch Geometric raw data
│   ├── qm9_train.pkl     # Processed train split
│   ├── qm9_val.pkl       # Processed val split
│   └── qm9_test.pkl      # Processed test split
└── mp20/
    ├── mp20_train.pkl    # Processed train split
    ├── mp20_val.pkl      # Processed val split
    └── mp20_test.pkl     # Processed test split
```

## Dependencies

Required packages (automatically installed):
- `torch-geometric` - For QM9 dataset
- `datasets` - For HuggingFace MP20 dataset
- `huggingface-hub` - For HuggingFace integration

## Notes

- QM9 download uses PyTorch Geometric and may take ~30 minutes on first run
- MP20 downloads from HuggingFace's `chaitjo/MP20_ADiT` repository
- Processed data is cached to avoid re-downloading
- If MP20 is not accessible from HuggingFace, a small fallback dataset is created for testing

## Reference

Dataset implementations based on:
- [all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer) by Meta AI Research
