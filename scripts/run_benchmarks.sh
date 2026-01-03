#!/bin/bash
# Run main benchmarks

CONFIG=${1:-configs/default.yaml}

uv run python src/scripts/benchmarks.py "$CONFIG"
