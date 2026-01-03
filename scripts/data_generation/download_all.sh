#!/bin/bash
# Download all datasets (QM9 and MP20)

set -e

# Default data directory
DATA_DIR="${1:-./data/downloaded}"

echo "========================================="
echo "Downloading All Datasets"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo ""
echo "This will download both QM9 and MP20 datasets from HuggingFace."
echo "Total download time may take ~10-20 minutes on first run."
echo ""

# Download QM9
echo "Step 1/2: Downloading QM9..."
bash scripts/data_generation/download_qm9.sh "$DATA_DIR"

echo ""
echo ""

# Download MP20
echo "Step 2/2: Downloading MP20..."
bash scripts/data_generation/download_mp20.sh "$DATA_DIR"

echo ""
echo "========================================="
echo "All datasets downloaded successfully!"
echo "========================================="
echo "QM9:  $DATA_DIR/qm9"
echo "MP20: $DATA_DIR/mp20"
echo ""
