#!/bin/bash
# Download and process MP20 dataset
# This will download the MP20 dataset from HuggingFace
# and process it into train/val/test splits.

set -e

# Default data directory
DATA_DIR="${1:-./data/downloaded}"

echo "========================================="
echo "Downloading MP20 Dataset"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo ""
echo "This will download ~20k crystal structures from HuggingFace."
echo "First download may take some time."
echo ""

# Run the download script
uv run python -c "
import sys
sys.path.insert(0, '.')
from src.data.mp20 import MP20Dataset

data_dir = '$DATA_DIR/mp20'
print(f'Creating MP20 dataset in {data_dir}...')

# Download and process all splits
for split in ['train', 'val', 'test']:
    print(f'\n--- Processing {split} split ---')
    dataset = MP20Dataset(data_dir=data_dir, split=split, download=True)
    print(f'✓ {split}: {len(dataset)} structures')

print('\n========================================')
print('MP20 dataset download complete!')
print('========================================')
"

echo ""
echo "Done! MP20 dataset saved to: $DATA_DIR/mp20"
