#!/bin/bash
# Run overnight experiments across 4 GPUs
#
# Usage:
#   ./scripts/experiments/run_overnight.sh          # Run all experiments
#   ./scripts/experiments/run_overnight.sh --dry-run  # Dry run
#   ./scripts/experiments/run_overnight.sh --filter muon  # Only muon experiments
#   ./scripts/experiments/run_overnight.sh --gpus 0,1  # Use specific GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

# Regenerate configs (in case of changes)
echo "Generating experiment configs..."
uv run python scripts/experiments/generate_configs.py

# Run experiments
echo ""
echo "Starting parallel experiment runner..."
echo "Logs will be saved to: data/experiments/logs/"
echo ""

uv run python scripts/experiments/run_parallel.py \
    --gpus 0,1,2,3 \
    --log-dir data/experiments/logs \
    "$@"

echo ""
echo "Experiments completed!"
