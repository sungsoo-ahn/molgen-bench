#!/bin/bash
# Run scaling law experiments

CONFIG=${1:-configs/default.yaml}

uv run python src/scripts/scaling_experiments.py "$CONFIG"
