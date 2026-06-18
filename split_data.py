"""
split_data.py – Physically separate CSV files into train/ and test/ directories.

Usage
-----
    python split_data.py --data_root ./data --test_ratio 0.15 --seed 42

Handles mixed resolutions (e.g. HDPE_c4 + HDPE_c8).  Each folder is split
independently; the training pipeline merges them by class at load time.

Produces:
    data/
    ├── train/
    │   ├── HDPE_c4/
    │   ├── HDPE_c8/
    │   ├── PET_c4/
    │   └── ...
    └── test/
        ├── HDPE_c4/
        ├── HDPE_c8/
        ├── PET_c4/
        └── ...

Files are COPIED (originals remain untouched).  Use --move to move instead.
"""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Split FTIR data into train/test on disk")
    p.add_argument("--data_root", type=str, required=True,
                    help="Root dir containing the *_c4 class folders")
    p.add_argument("--test_ratio", type=float, default=0.15,
                    help="Fraction of files to hold out for testing (default 0.15)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    p.add_argument("--move", action="store_true",
                    help="Move files instead of copying (saves disk space)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)

    # Discover class folders (anything ending in _c4, or all subdirs)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name not in ("train", "test")])
    if not class_dirs:
        print(f"No class folders found in {root}")
        return

    train_root = root / "train"
    test_root = root / "test"
    train_root.mkdir(exist_ok=True)
    test_root.mkdir(exist_ok=True)

    transfer = shutil.move if args.move else shutil.copy2
    action = "Moving" if args.move else "Copying"

    total_train, total_test = 0, 0

    print(f"Splitting with test_ratio={args.test_ratio}, seed={args.seed}")
    print(f"{'Class':<25} {'Total':>6} {'Train':>6} {'Test':>6}")
    print("-" * 47)

    for class_dir in class_dirs:
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            print(f"  {class_dir.name}: no CSV files, skipping")
            continue

        # Stratified isn't needed within a single class — just random split
        train_files, test_files = train_test_split(
            csv_files,
            test_size=args.test_ratio,
            random_state=args.seed,
        )

        # Create class subdirectories
        (train_root / class_dir.name).mkdir(exist_ok=True)
        (test_root / class_dir.name).mkdir(exist_ok=True)

        for f in train_files:
            transfer(str(f), str(train_root / class_dir.name / f.name))
        for f in test_files:
            transfer(str(f), str(test_root / class_dir.name / f.name))

        total_train += len(train_files)
        total_test += len(test_files)
        print(f"{class_dir.name:<25} {len(csv_files):>6} {len(train_files):>6} {len(test_files):>6}")

    print("-" * 47)
    print(f"{'TOTAL':<25} {total_train + total_test:>6} {total_train:>6} {total_test:>6}")
    print(f"\n{action} complete.")
    print(f"  Train → {train_root}")
    print(f"  Test  → {test_root}")


if __name__ == "__main__":
    main()
