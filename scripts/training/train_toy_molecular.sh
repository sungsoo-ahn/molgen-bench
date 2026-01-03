#!/bin/bash
# Train 3D toy molecular generative model
#
# Usage:
#   bash scripts/training/train_toy_molecular.sh                    # Full training
#   bash scripts/training/train_toy_molecular.sh --quick            # Quick test

set -e

# Default config
CONFIG="configs/toy_molecular.yaml"

# Check for quick flag
if [ "$1" = "--quick" ]; then
    CONFIG="configs/toy_molecular_quick.yaml"
    echo "Using quick config for fast testing..."
fi

echo "Training 3D toy molecular model..."
echo "Config: $CONFIG"
echo ""

uv run python src/scripts/train_toy_molecular.py --config "$CONFIG"
