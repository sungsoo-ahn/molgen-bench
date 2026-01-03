#!/usr/bin/env python3
"""Run multiple experiments in parallel across GPUs.

This script manages parallel execution of experiments across multiple GPUs,
handling job queuing, GPU allocation, and failure recovery.

Usage:
    # Run all experiments from the list
    python scripts/experiments/run_parallel.py

    # Run specific experiments
    python scripts/experiments/run_parallel.py --configs config1.yaml config2.yaml

    # Limit to specific GPUs
    python scripts/experiments/run_parallel.py --gpus 0,1,2,3

    # Dry run (print commands without executing)
    python scripts/experiments/run_parallel.py --dry-run
"""

import argparse
import subprocess
import yaml
import os
import time
import signal
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
from threading import Lock
import multiprocessing


@dataclass
class Experiment:
    """Represents a single experiment."""
    name: str
    config_path: str
    gpu_id: int = -1
    status: str = "pending"  # pending, running, completed, failed
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None


def load_experiment_list(list_path: str) -> List[str]:
    """Load experiment configs from a YAML list file."""
    with open(list_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("experiments", [])


def run_experiment(config_path: str, gpu_id: int, log_dir: Path) -> dict:
    """Run a single experiment on a specific GPU.

    Args:
        config_path: Path to experiment config
        gpu_id: GPU ID to use
        log_dir: Directory for log files

    Returns:
        dict with status and timing info
    """
    config_name = Path(config_path).stem
    log_file = log_dir / f"{config_name}.log"
    error_file = log_dir / f"{config_name}.error.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = ["uv", "run", "python", "src/scripts/train_qm9.py", "--config", config_path]

    start_time = time.time()

    try:
        with open(log_file, "w") as log_f, open(error_file, "w") as err_f:
            log_f.write(f"Running: {' '.join(cmd)}\n")
            log_f.write(f"GPU: {gpu_id}\n")
            log_f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write("=" * 60 + "\n\n")
            log_f.flush()

            process = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=err_f,
                env=env,
                cwd=str(Path(__file__).parent.parent.parent),
            )

            end_time = time.time()
            duration = end_time - start_time

            log_f.write(f"\n" + "=" * 60 + "\n")
            log_f.write(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Duration: {duration:.1f}s ({duration/60:.1f}min)\n")
            log_f.write(f"Return code: {process.returncode}\n")

        if process.returncode == 0:
            return {
                "status": "completed",
                "duration": duration,
                "config": config_path,
                "gpu": gpu_id,
            }
        else:
            with open(error_file, "r") as f:
                error_content = f.read()
            return {
                "status": "failed",
                "duration": duration,
                "config": config_path,
                "gpu": gpu_id,
                "error": error_content[-1000:] if len(error_content) > 1000 else error_content,
            }

    except Exception as e:
        end_time = time.time()
        return {
            "status": "failed",
            "duration": end_time - start_time,
            "config": config_path,
            "gpu": gpu_id,
            "error": str(e),
        }


def run_experiment_wrapper(args):
    """Wrapper for multiprocessing."""
    config_path, gpu_id, log_dir = args
    return run_experiment(config_path, gpu_id, log_dir)


class ExperimentRunner:
    """Manages parallel execution of experiments across GPUs."""

    def __init__(
        self,
        config_paths: List[str],
        gpu_ids: List[int],
        log_dir: Path,
        dry_run: bool = False,
    ):
        self.config_paths = config_paths
        self.gpu_ids = gpu_ids
        self.log_dir = log_dir
        self.dry_run = dry_run

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = [
            Experiment(name=Path(p).stem, config_path=p)
            for p in config_paths
        ]

        self.completed = []
        self.failed = []

    def run_all(self):
        """Run all experiments in parallel across GPUs."""
        print(f"\n{'='*60}")
        print(f"Experiment Runner")
        print(f"{'='*60}")
        print(f"Total experiments: {len(self.experiments)}")
        print(f"Available GPUs: {self.gpu_ids}")
        print(f"Log directory: {self.log_dir}")
        print(f"{'='*60}\n")

        if self.dry_run:
            print("DRY RUN - Commands that would be executed:\n")
            for exp in self.experiments:
                print(f"  CUDA_VISIBLE_DEVICES=<GPU> uv run python src/scripts/train_qm9.py --config {exp.config_path}")
            return

        start_time = time.time()

        # Create work items: (config_path, gpu_id, log_dir)
        # We'll cycle through GPUs for each experiment
        work_items = []
        for i, exp in enumerate(self.experiments):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            work_items.append((exp.config_path, gpu_id, self.log_dir))

        # Run experiments in parallel (max workers = number of GPUs)
        with ProcessPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            future_to_exp = {
                executor.submit(run_experiment_wrapper, item): item[0]
                for item in work_items
            }

            for future in as_completed(future_to_exp):
                config_path = future_to_exp[future]
                config_name = Path(config_path).stem

                try:
                    result = future.result()

                    if result["status"] == "completed":
                        self.completed.append(result)
                        print(f"✓ {config_name} completed in {result['duration']/60:.1f}min (GPU {result['gpu']})")
                    else:
                        self.failed.append(result)
                        print(f"✗ {config_name} FAILED on GPU {result['gpu']}")
                        if result.get("error"):
                            print(f"  Error: {result['error'][:200]}...")

                except Exception as e:
                    self.failed.append({
                        "status": "failed",
                        "config": config_path,
                        "error": str(e),
                    })
                    print(f"✗ {config_name} FAILED with exception: {e}")

        end_time = time.time()
        total_duration = end_time - start_time

        # Print summary
        self._print_summary(total_duration)

    def _print_summary(self, total_duration: float):
        """Print final summary of all experiments."""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
        print(f"Completed: {len(self.completed)}/{len(self.experiments)}")
        print(f"Failed: {len(self.failed)}/{len(self.experiments)}")

        if self.completed:
            print(f"\n{'='*40}")
            print("Completed experiments:")
            for exp in self.completed:
                name = Path(exp["config"]).stem
                print(f"  ✓ {name}: {exp['duration']/60:.1f}min")

        if self.failed:
            print(f"\n{'='*40}")
            print("Failed experiments:")
            for exp in self.failed:
                name = Path(exp["config"]).stem
                print(f"  ✗ {name}")
                if exp.get("error"):
                    # Print first line of error
                    error_line = exp["error"].split("\n")[-1][:100]
                    print(f"    {error_line}")

        # Save summary to file
        summary_path = self.log_dir / "summary.yaml"
        summary = {
            "total_duration_seconds": total_duration,
            "total_experiments": len(self.experiments),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "completed_experiments": [Path(e["config"]).stem for e in self.completed],
            "failed_experiments": [Path(e["config"]).stem for e in self.failed],
        }
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)
        print(f"\nSummary saved to: {summary_path}")


def get_available_gpus() -> List[int]:
    """Get list of available CUDA GPUs."""
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except ImportError:
        # Fallback: try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True
            )
            return list(range(len(result.stdout.strip().split("\n"))))
        except Exception:
            return [0]  # Default to single GPU


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments in parallel across GPUs"
    )
    parser.add_argument(
        "--list",
        type=str,
        default="configs/experiments/experiment_list.yaml",
        help="Path to experiment list YAML"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=str,
        help="Specific config files to run (overrides --list)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/experiments/logs",
        help="Directory for experiment logs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter experiments by name pattern"
    )

    args = parser.parse_args()

    # Get experiment configs
    if args.configs:
        config_paths = args.configs
    else:
        config_paths = load_experiment_list(args.list)

    # Apply filter if specified
    if args.filter:
        config_paths = [p for p in config_paths if args.filter in p]

    if not config_paths:
        print("No experiments to run!")
        return

    # Get GPU IDs
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = get_available_gpus()

    if not gpu_ids:
        print("No GPUs available!")
        return

    # Create runner and execute
    log_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S")

    runner = ExperimentRunner(
        config_paths=config_paths,
        gpu_ids=gpu_ids,
        log_dir=log_dir,
        dry_run=args.dry_run,
    )

    runner.run_all()


if __name__ == "__main__":
    main()
