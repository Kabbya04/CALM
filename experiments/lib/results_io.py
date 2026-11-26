import json
import os
import sys
from datetime import datetime, timezone

_CALM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CALM_ROOT not in sys.path:
    sys.path.insert(0, _CALM_ROOT)

from atomic_json import atomic_write_json  # noqa: E402


DEFAULT_RESULTS_DIR = os.path.join(_CALM_ROOT, "results")


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
    atomic_write_json(path, record)
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
