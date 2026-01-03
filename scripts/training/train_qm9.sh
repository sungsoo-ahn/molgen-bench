#!/bin/bash
# Train QM9 generative model

set -e

# Default config
CONFIG="${1:-configs/qm9_quick.yaml}"

echo "========================================="
echo "Training QM9 Generative Model"
echo "========================================="
echo "Config: $CONFIG"
echo ""

# Run training
uv run python src/scripts/train_qm9.py --config "$CONFIG"

echo ""
echo "Training complete!"
