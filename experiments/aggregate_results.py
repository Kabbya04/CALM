"""Mean ± std over JSON records in CALM/results."""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.results_io import DEFAULT_RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate CALM result JSON files.")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["avg_acc", "bwt", "accuracy", "mean_accuracy"],
        help="Numeric fields to summarize when present.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"No results directory at {args.results_dir}")

    groups = defaultdict(list)
    for name in sorted(os.listdir(args.results_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(args.results_dir, name)
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        key = (
            record.get("variant", name),
            record.get("dataset", "?"),
            record.get("protocol", "?"),
        )
        groups[key].append(record)

    print(f"{'variant':<24} {'dataset':<12} {'protocol':<16} {'n':>3}  metrics")
    print("-" * 90)
    for key, rows in sorted(groups.items()):
        variant, dataset, protocol = key
        parts = []
        for field in args.keys:
            values = [float(row[field]) for row in rows if field in row]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
            parts.append(f"{field}={mean:.2f}±{var ** 0.5:.2f}")
        print(f"{variant:<24} {dataset:<12} {protocol:<16} {len(rows):>3}  {', '.join(parts)}")


if __name__ == "__main__":
    main()
