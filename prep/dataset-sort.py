"""
Sort a flat Kaggle ImageNet validation dump into torchvision layout:

    <imagenet_root>/val/<wnid>/*.JPEG

Requires LOC_val_solution.csv from the original Kaggle ImageNet download.

Example:
  python dataset-sort.py --imagenet_root D:\\Datasets\\imagenet_torchvision \\
                         --kaggle_root D:\\Datasets\\imagenet_torchvision
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _already_sorted(val_dir: Path) -> bool:
    """True if val/ already looks like class folders with JPEGs inside."""
    class_dirs = [p for p in val_dir.iterdir() if p.is_dir()]
    if len(class_dirs) < 100:
        return False
    sample = class_dirs[:20]
    return all(any(p.suffix.upper() == ".JPEG" for p in d.glob("*")) for d in sample)


def _flat_jpegs(val_dir: Path) -> list[Path]:
    return [p for p in val_dir.iterdir() if p.is_file() and p.suffix.upper() == ".JPEG"]


def preprocess_validation_set(imagenet_root: str, original_kaggle_root: str) -> None:
    val_dir = Path(imagenet_root) / "val"
    if not val_dir.is_dir():
        raise SystemExit(f"Error: validation directory not found at {val_dir}")

    solution_file = Path(original_kaggle_root) / "LOC_val_solution.csv"
    if not solution_file.is_file():
        raise SystemExit(f"Error: validation solution file not found at {solution_file}")

    flat = _flat_jpegs(val_dir)
    if not flat and _already_sorted(val_dir):
        n_classes = sum(1 for p in val_dir.iterdir() if p.is_dir())
        print(f"Validation set already sorted ({n_classes} class folders). Nothing to do.")
        return

    print("Reading validation solutions...")
    df = pd.read_csv(solution_file)
    if "PredictionString" not in df.columns or "ImageId" not in df.columns:
        raise SystemExit("LOC_val_solution.csv must contain ImageId and PredictionString columns.")

    df["class_id"] = df["PredictionString"].astype(str).str.split().str[0]
    print(f"Found {len(df)} validation labels to apply.")

    moved = 0
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sorting validation images"):
        image_name = f"{row['ImageId']}.JPEG"
        class_id = row["class_id"]
        class_dir = val_dir / class_id
        class_dir.mkdir(parents=True, exist_ok=True)

        src = val_dir / image_name
        dest = class_dir / image_name
        if dest.exists():
            continue
        if src.exists():
            shutil.move(str(src), str(dest))
            moved += 1
        else:
            missing += 1

    remaining_flat = len(_flat_jpegs(val_dir))
    print(
        f"Validation set preprocessing complete. "
        f"moved={moved}, already_placed_or_skipped={len(df) - moved - missing}, "
        f"missing_sources={missing}, remaining_flat_jpegs={remaining_flat}"
    )
    if remaining_flat:
        print(
            "Warning: flat JPEGs remain under val/. Check ImageId spelling "
            "or that --imagenet_root points at the torchvision-style tree."
        )


def main():
    parser = argparse.ArgumentParser(description="Preprocess ImageNet validation set into class folders.")
    parser.add_argument(
        "--imagenet_root",
        type=str,
        required=True,
        help="Path to the imagenet_torchvision root (contains val/).",
    )
    parser.add_argument(
        "--kaggle_root",
        type=str,
        required=True,
        help="Path containing LOC_val_solution.csv (often the same as imagenet_root).",
    )
    args = parser.parse_args()
    preprocess_validation_set(args.imagenet_root, args.kaggle_root)


if __name__ == "__main__":
    main()
