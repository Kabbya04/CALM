import json
import os
from datetime import datetime, timezone


DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)


def write_result(payload: dict, output_json: str = None, results_dir: str = None) -> str:
    """Write one run record. Returns the path that was written."""
    record = dict(payload)
    record.setdefault("written_at", datetime.now(timezone.utc).isoformat())
    if output_json:
        path = output_json
    else:
        directory = results_dir or DEFAULT_RESULTS_DIR
        os.makedirs(directory, exist_ok=True)
        variant = record.get("variant", "run")
        dataset = record.get("dataset", "unknown")
        seed = record.get("seed", "na")
        path = os.path.join(directory, f"{variant}_{dataset}_seed{seed}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"Wrote results to {path}")
    return path


def add_output_flag(parser):
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path for a single-run JSON record.",
    )
    parser.add_argument(
        "--results_dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory used when --output_json is omitted.",
    )
