"""Test script to verify QM9 and MP20 dataset loading."""

import argparse
import sys
from pathlib import Path

from src.data import QM9Dataset, MP20Dataset


def test_qm9(data_dir: str, split: str = "train"):
    """Test QM9 dataset loading."""
    print(f"\n{'='*60}")
    print(f"Testing QM9 Dataset ({split} split)")
    print(f"{'='*60}")

    try:
        dataset = QM9Dataset(data_dir=data_dir, split=split, download=True)
        print(f"✓ Dataset loaded: {len(dataset)} molecules")

        # Test getting a single sample
        sample = dataset[0]
        print(f"\nSample 0 structure:")
        print(f"  - Positions shape: {sample['positions'].shape}")
        print(f"  - Atom types shape: {sample['atom_types'].shape}")
        print(f"  - Atom types: {sample['atom_types'].numpy()}")
        print(f"  - Properties: {list(sample['properties'].keys())}")
        print(f"  - mu (dipole moment): {sample['properties']['mu']:.3f}")

        # Test dataset statistics
        if split == "train":
            stats = dataset.get_statistics()
            print(f"\nDataset statistics (sample properties):")
            for prop in ['mu', 'gap', 'homo', 'lumo']:
                print(f"  - {prop}: mean={stats[prop]['mean']:.3f}, "
                      f"std={stats[prop]['std']:.3f}")

        print(f"\n✓ QM9 {split} dataset test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ QM9 {split} dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mp20(data_dir: str, split: str = "train"):
    """Test MP20 dataset loading."""
    print(f"\n{'='*60}")
    print(f"Testing MP20 Dataset ({split} split)")
    print(f"{'='*60}")

    try:
        dataset = MP20Dataset(data_dir=data_dir, split=split, download=True)
        print(f"✓ Dataset loaded: {len(dataset)} structures")

        # Test getting a single sample
        sample = dataset[0]
        print(f"\nSample 0 structure:")
        print(f"  - Positions shape: {sample['positions'].shape}")
        print(f"  - Atom types shape: {sample['atom_types'].shape}")
        print(f"  - Atom types: {sample['atom_types'].numpy()[:10]}...")  # First 10
        if 'lattice' in sample:
            print(f"  - Lattice shape: {sample['lattice'].shape}")
        print(f"  - Properties: {list(sample['properties'].keys())}")

        print(f"\n✓ MP20 {split} dataset test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ MP20 {split} dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["qm9", "mp20", "all"],
        default="all",
        help="Which dataset to test"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/downloaded",
        help="Data directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to test"
    )
    args = parser.parse_args()

    results = []

    if args.dataset in ["qm9", "all"]:
        qm9_dir = Path(args.data_dir) / "qm9"
        result = test_qm9(str(qm9_dir), args.split)
        results.append(("QM9", result))

    if args.dataset in ["mp20", "all"]:
        mp20_dir = Path(args.data_dir) / "mp20"
        result = test_mp20(str(mp20_dir), args.split)
        results.append(("MP20", result))

    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")

    # Exit with appropriate code
    all_passed = all(result for _, result in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
