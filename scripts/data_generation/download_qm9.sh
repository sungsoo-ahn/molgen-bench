#!/bin/bash
# Download and process QM9 dataset
# This will download the QM9 dataset using PyTorch Geometric
# and process it into train/val/test splits.

set -e

# Default data directory
DATA_DIR="${1:-./data/downloaded}"

echo "========================================="
echo "Downloading QM9 Dataset"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo ""
echo "This will download ~134k molecules from HuggingFace (chaitjo/QM9_ADiT)."
echo "First download may take ~5-10 minutes (340MB file)."
echo ""

# Run the download script
uv run python -c "
import sys
sys.path.insert(0, '.')
from src.data.qm9 import QM9Dataset

data_dir = '$DATA_DIR/qm9'
print(f'Creating QM9 dataset in {data_dir}...')

# Download and process all splits
for split in ['train', 'val', 'test']:
    print(f'\n--- Processing {split} split ---')
    dataset = QM9Dataset(data_dir=data_dir, split=split, download=True)
    print(f'✓ {split}: {len(dataset)} molecules')

print('\n========================================')
print('QM9 dataset download complete!')
print('========================================')
"

echo ""
echo "Done! QM9 dataset saved to: $DATA_DIR/qm9"
