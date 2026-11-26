"""
Run the post-update CALM experiment campaign on a GPU machine.

Prerequisites (on the target device):
  1. conda env with CUDA PyTorch + prep/requirements.txt
  2. python prep/cuda-check.py  (must PASS)
  3. ImageNet val sorted via prep/dataset-sort.py

Each job is a separate process so GPU memory is released between runs.
Progress is shown with a terminal progress bar; per-job logs and JSON
results land under CALM/results/.

Example:
  python run_all_experiments.py --imagenet_root D:\\Datasets\\imagenet_torchvision
  python run_all_experiments.py --suite correction --seeds 42
  python run_all_experiments.py --suite all --force

Resume / fail-safe (default):
  Re-run the same command after a crash or load-shedding outage.
  Completed jobs in results/ are skipped; incomplete or interrupted jobs retry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

PREP_DIR = Path(__file__).resolve().parent
CALM_ROOT = PREP_DIR.parent
if str(CALM_ROOT) not in sys.path:
    sys.path.insert(0, str(CALM_ROOT))
from atomic_json import atomic_write_json  # noqa: E402

RESULTS = CALM_ROOT / "results"
EXP = CALM_ROOT / "experiments"
V1 = CALM_ROOT / "CALM-v1" / "CALM-Continual_Adaptive_Learning_Model" / "final-evaluation"
V1O = V1 / "final-eval-other"
V2I = CALM_ROOT / "CALM-v2" / "imagenet-eval"
V2O = CALM_ROOT / "CALM-v2" / "final-eval-other"

DEFAULT_SEEDS = (42, 43, 44)
PAPER_DATASETS = ("cifar10", "cifar100", "stl10", "svhn", "fashionmnist")
SUITES = (
    "correction",
    "imagenet_baselines",
    "imagenet_calm",
    "others_v2pe",
    "ablations",
    "readiness",
    "all",
)


def parse_args():
    parser = argparse.ArgumentParser(description="CALM post-update experiment runner.")
    parser.add_argument(
        "--imagenet_root",
        default=os.environ.get("CALM_IMAGENET_ROOT", r"D:\Datasets\imagenet_torchvision"),
        help="ImageNet torchvision root (contains val/<wnid>/).",
    )
    parser.add_argument(
        "--data_root",
        default=os.environ.get("CALM_DATA_ROOT", str(CALM_ROOT / "data")),
        help="Download/cache root for CIFAR/STL/SVHN/FashionMNIST.",
    )
    parser.add_argument(
        "--results_dir",
        default=str(RESULTS),
        help="Directory for JSON results and per-job logs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable of the CUDA conda env.",
    )
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default="all",
        help="Which experiment block to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip jobs that already have a valid results JSON (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not skip completed jobs (same effect as --force for scheduling).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every job even when a valid results JSON exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the job list without executing.",
    )
    parser.add_argument(
        "--stop_on_fail",
        action="store_true",
        help="Abort the suite after the first failing job.",
    )
    return parser.parse_args()


def job(name: str, cwd: Path, args: list[str], output_json: Path) -> dict:
    return {
        "name": name,
        "cwd": str(cwd),
        "args": args,
        "output_json": str(output_json),
    }


def build_jobs(args) -> list[dict]:
    py = args.python
    device = args.device
    seeds = tuple(args.seeds)
    res = Path(args.results_dir)
    imagenet = args.imagenet_root
    data_root = args.data_root
    suite = args.suite
    jobs: list[dict] = []

    want = {
        "correction": suite in ("correction", "all"),
        "imagenet_baselines": suite in ("imagenet_baselines", "all"),
        "imagenet_calm": suite in ("imagenet_calm", "all"),
        "others_v2pe": suite in ("others_v2pe", "all"),
        "ablations": suite in ("ablations", "all"),
        "readiness": suite in ("readiness", "all"),
    }

    # B1 / B3: S0–S4 correction on CIFAR-10 and CIFAR-100 (one JSON per system)
    if want["correction"]:
        for dataset in ("cifar10", "cifar100"):
            for seed in seeds:
                for system in ("S0", "S1", "S2", "S3", "S4"):
                    out = res / f"corr_{system}_{dataset}_seed{seed}.json"
                    jobs.append(
                        job(
                            f"corr_{system}_{dataset}_seed{seed}",
                            EXP,
                            [
                                py,
                                "run_correction_cost.py",
                                "--dataset",
                                dataset,
                                "--data_root",
                                data_root,
                                "--system",
                                system,
                                "--seed",
                                str(seed),
                                "--device",
                                device,
                                "--output_json",
                                str(out),
                                "--results_dir",
                                str(res),
                            ],
                            out,
                        )
                    )

    # A3 / B3: ImageNet continual last-task mean for matched baselines
    if want["imagenet_baselines"]:
        for seed in seeds:
            rows = [
                (
                    f"clip_zeroshot_imagenet_continual_seed{seed}.json",
                    [
                        py,
                        "run_clip_zeroshot.py",
                        "--imagenet_root",
                        imagenet,
                        "--num_tasks",
                        "10",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                    ],
                ),
                (
                    f"tip_adapter_imagenet_continual_seed{seed}.json",
                    [
                        py,
                        "run_tip_adapter.py",
                        "--imagenet_root",
                        imagenet,
                        "--num_tasks",
                        "10",
                        "--k_shot",
                        "5",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                    ],
                ),
                (
                    f"icarl_clip_imagenet_continual_seed{seed}.json",
                    [
                        py,
                        "run_icarl.py",
                        "--imagenet_root",
                        imagenet,
                        "--num_tasks",
                        "10",
                        "--k_shot",
                        "5",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--backbone",
                        "clip",
                    ],
                ),
                (
                    f"dualprompt_imagenet_continual_seed{seed}.json",
                    [
                        py,
                        "run_dualprompt.py",
                        "--imagenet_root",
                        imagenet,
                        "--num_tasks",
                        "10",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                    ],
                ),
            ]
            for fname, cmd in rows:
                out = res / fname
                jobs.append(
                    job(
                        fname[:-5],
                        EXP,
                        cmd + ["--output_json", str(out), "--results_dir", str(res)],
                        out,
                    )
                )

    # B2: fixed per-encoder v2 + v1 ImageNet few-shot / continual
    if want["imagenet_calm"]:
        for seed in seeds:
            out = res / f"calm_v1_imagenet_fewshot_seed{seed}.json"
            jobs.append(
                job(
                    f"calm_v1_imagenet_fewshot_seed{seed}",
                    V1,
                    [
                        py,
                        "run_imagenet_evaluation.py",
                        "--imagenet_root",
                        imagenet,
                        "--n_way",
                        "5",
                        "--k_shot",
                        "5",
                        "--num_query",
                        "15",
                        "--num_episodes",
                        "600",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--output_json",
                        str(out),
                    ],
                    out,
                )
            )
            out = res / f"calm_v2pe_imagenet_fewshot_seed{seed}.json"
            jobs.append(
                job(
                    f"calm_v2pe_imagenet_fewshot_seed{seed}",
                    V2I,
                    [
                        py,
                        "run_imagenet_evaluation.py",
                        "--imagenet_root",
                        imagenet,
                        "--n_way",
                        "5",
                        "--k_shot",
                        "5",
                        "--num_query",
                        "15",
                        "--num_episodes",
                        "600",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--temperature",
                        "1.0",
                        "--output_json",
                        str(out),
                    ],
                    out,
                )
            )
            out = res / f"calm_v1_imagenet_continual_seed{seed}.json"
            jobs.append(
                job(
                    f"calm_v1_imagenet_continual_seed{seed}",
                    V1,
                    [
                        py,
                        "run_imagenet_continual.py",
                        "--imagenet_root",
                        imagenet,
                        "--num_tasks",
                        "10",
                        "--k_shot",
                        "5",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--output_json",
                        str(out),
                    ],
                    out,
                )
            )
            out = res / f"calm_v2pe_imagenet_continual_seed{seed}.json"
            jobs.append(
                job(
                    f"calm_v2pe_imagenet_continual_seed{seed}",
                    V2I,
                    [
                        py,
                        "run_imagenet_continual.py",
                        "--imagenet_root",
                        imagenet,
                        "--num_tasks",
                        "10",
                        "--k_shot",
                        "5",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--temperature",
                        "1.0",
                        "--output_json",
                        str(out),
                    ],
                    out,
                )
            )

    # B2: fixed-v2 paper configs on all six datasets
    if want["others_v2pe"]:
        for dataset in PAPER_DATASETS:
            for seed in seeds:
                out = res / f"calm_v2pe_{dataset}_seed{seed}.json"
                cmd = [
                    py,
                    "final-others.py",
                    "--datasetname",
                    dataset,
                    "--device",
                    device,
                    "--seed",
                    str(seed),
                    "--temperature",
                    "1.0",
                    "--output_json",
                    str(out),
                ]
                if data_root:
                    cmd.extend(["--data_root", data_root])
                jobs.append(job(f"calm_v2pe_{dataset}_seed{seed}", V2O, cmd, out))

    # B2 / B3 ablations that remain valid after the fix (no byte-budget sweeps)
    if want["ablations"]:
        for seed in seeds:
            rows = [
                (
                    f"ablate_B3_ensemble_flat_cifar100_seed{seed}.json",
                    ["--flat_memory"],
                ),
                (
                    f"ablate_B6_single_backbone_cifar100_seed{seed}.json",
                    ["--single_backbone"],
                ),
            ]
            for fname, extra in rows:
                out = res / fname
                cmd = [
                    py,
                    "final-others.py",
                    "--datasetname",
                    "cifar100",
                    "--device",
                    device,
                    "--seed",
                    str(seed),
                    *extra,
                    "--output_json",
                    str(out),
                ]
                if data_root:
                    cmd.extend(["--data_root", data_root])
                jobs.append(job(fname[:-5], V2O, cmd, out))

            out = res / f"ablate_B6_single_backbone_imagenet_seed{seed}.json"
            jobs.append(
                job(
                    f"ablate_B6_single_backbone_imagenet_seed{seed}",
                    V2I,
                    [
                        py,
                        "run_imagenet_continual.py",
                        "--imagenet_root",
                        imagenet,
                        "--num_tasks",
                        "10",
                        "--k_shot",
                        "5",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--single_backbone",
                        "--output_json",
                        str(out),
                    ],
                    out,
                )
            )

    # B2 readiness (v1 kept; v2 uses per-encoder banks → readiness_v2pe)
    if want["readiness"]:
        for dataset in ("cifar10", "cifar100"):
            for seed in seeds:
                out = res / f"ready_v1_{dataset}_seed{seed}.json"
                jobs.append(
                    job(
                        f"ready_v1_{dataset}_seed{seed}",
                        EXP,
                        [
                            py,
                            "run_readiness.py",
                            "--variant",
                            "v1",
                            "--dataset",
                            dataset,
                            "--data_root",
                            data_root,
                            "--k_shot",
                            "5",
                            "--seed",
                            str(seed),
                            "--device",
                            device,
                            "--output_json",
                            str(out),
                            "--results_dir",
                            str(res),
                        ],
                        out,
                    )
                )
                out = res / f"ready_v2pe_{dataset}_seed{seed}.json"
                jobs.append(
                    job(
                        f"ready_v2pe_{dataset}_seed{seed}",
                        EXP,
                        [
                            py,
                            "run_readiness.py",
                            "--variant",
                            "v2",
                            "--dataset",
                            dataset,
                            "--data_root",
                            data_root,
                            "--k_shot",
                            "5",
                            "--seed",
                            str(seed),
                            "--device",
                            device,
                            "--output_json",
                            str(out),
                            "--results_dir",
                            str(res),
                        ],
                        out,
                    )
                )

    return jobs


def running_marker_path(output_json: str) -> str:
    return f"{output_json}.running"


def mark_running(item: dict, results_dir: Path, index: int, total: int) -> None:
    atomic_write_json(
        running_marker_path(item["output_json"]),
        {
            "name": item["name"],
            "output_json": item["output_json"],
            "index": index,
            "total": total,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(results_dir),
        },
    )


def clear_running(item: dict) -> None:
    path = running_marker_path(item["output_json"])
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def required_fields(record: dict) -> list[str]:
    protocol = str(record.get("protocol", "")).lower()
    if protocol == "correction_loop":
        return ["autonomous_accuracy", "heldout_accuracy_after_stream", "labels_used"]
    if protocol == "readiness":
        return ["curve"]
    if protocol == "few_shot":
        return ["mean_accuracy"]
    if protocol == "few_shot+continual":
        return ["mean_accuracy", "avg_acc"]
    if protocol == "continual":
        return ["avg_acc", "last_task_mean"]
    if "continual" in protocol:
        return ["avg_acc", "last_task_mean"]
    # Unknown protocol: accept any common metric key so we do not false-retry.
    for key in ("avg_acc", "mean_accuracy", "accuracy", "autonomous_accuracy", "curve"):
        if key in record:
            return [key]
    return ["variant", "dataset"]


def done(path: str) -> bool:
    """True when a usable, schema-complete result JSON exists."""
    if not (os.path.isfile(path) and os.path.getsize(path) > 20):
        return False
    # Never treat an in-progress write as complete.
    if os.path.isfile(f"{path}.tmp"):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(record, dict):
        return False
    for field in required_fields(record):
        if field not in record:
            return False
    return True


def append_log(results_dir: Path, message: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with open(results_dir / "run_all.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_status(results_dir: Path, payload: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(results_dir / "run_all_status.json"), payload)


def reconcile_interrupted(jobs: list[dict], results_dir: Path) -> list[str]:
    """Clear stale .running markers; return names that were interrupted mid-job."""
    interrupted = []
    for item in jobs:
        marker = running_marker_path(item["output_json"])
        if not os.path.isfile(marker):
            continue
        if done(item["output_json"]):
            clear_running(item)
            continue
        interrupted.append(item["name"])
        # Drop partial/corrupt JSON so the job retries cleanly.
        out = item["output_json"]
        for path in (out, f"{out}.tmp"):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        clear_running(item)
        append_log(
            results_dir,
            f"INTERRUPT {item['name']} (.running found; will retry)",
        )
    return interrupted


def summarize_queue(jobs: list[dict], skip_completed: bool) -> dict:
    if skip_completed:
        completed = [item for item in jobs if done(item["output_json"])]
        pending = [item for item in jobs if not done(item["output_json"])]
    else:
        completed = []
        pending = list(jobs)
    return {
        "total": len(jobs),
        "already_done": len(completed),
        "pending": len(pending),
        "pending_names": [item["name"] for item in pending],
        "done_names": [item["name"] for item in completed],
    }


def run_one(item: dict, results_dir: Path, index: int, total: int) -> tuple[bool, int, float]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    # Limit allocator fragmentation; each subprocess still exits fully after the job.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_path = results_dir / f"{item['name']}.log"
    append_log(results_dir, f"START {index}/{total} {item['name']}")
    mark_running(item, results_dir, index, total)
    write_status(
        results_dir,
        {
            "current": item["name"],
            "index": index,
            "total": total,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "output_json": item["output_json"],
            "running_marker": running_marker_path(item["output_json"]),
        },
    )

    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as handle:
            proc = subprocess.run(
                item["args"],
                cwd=item["cwd"],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - t0
        ok = proc.returncode == 0 and done(item["output_json"])
        append_log(
            results_dir,
            f"{'OK' if ok else 'FAIL'} {index}/{total} {item['name']} "
            f"code={proc.returncode} {elapsed:.1f}s",
        )
        return ok, proc.returncode, elapsed
    finally:
        # Always drop the marker when the subprocess returns (success or fail).
        # A hard kill leaves the marker for the next resume pass.
        clear_running(item)


def aggregate(results_dir: Path, python_exe: str) -> None:
    append_log(results_dir, "Running aggregator...")
    agg = subprocess.run(
        [
            python_exe,
            "aggregate_results.py",
            "--results_dir",
            str(results_dir),
        ],
        cwd=str(EXP),
        capture_output=True,
        text=True,
    )
    aggregate_path = results_dir / "aggregate.txt"
    with open(aggregate_path, "w", encoding="utf-8") as handle:
        handle.write(agg.stdout or "")
        handle.write(agg.stderr or "")


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    skip_completed = bool(args.resume) and not bool(args.force)

    if args.dry_run:
        summary = summarize_queue(jobs, skip_completed=skip_completed)
        print(
            f"{summary['total']} jobs "
            f"(already_done={summary['already_done']}, pending={summary['pending']}, "
            f"resume={'on' if skip_completed else 'off'}):"
        )
        for i, item in enumerate(jobs, start=1):
            flag = "DONE" if skip_completed and done(item["output_json"]) else "PEND"
            print(f"  {i:3d}. [{flag}] {item['name']}")
        return 0

    interrupted = reconcile_interrupted(jobs, results_dir)
    summary = summarize_queue(jobs, skip_completed=skip_completed)
    append_log(
        results_dir,
        f"Queue has {summary['total']} jobs. suite={args.suite} device={args.device} "
        f"seeds={args.seeds} resume={'on' if skip_completed else 'off'} "
        f"already_done={summary['already_done']} pending={summary['pending']} "
        f"interrupted_retry={len(interrupted)} sequential=1",
    )
    write_status(
        results_dir,
        {
            "phase": "starting",
            "resume": skip_completed,
            "suite": args.suite,
            "summary": {
                "total": summary["total"],
                "already_done": summary["already_done"],
                "pending": summary["pending"],
                "interrupted_retry": len(interrupted),
            },
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    failed: list[dict] = []
    skipped = 0
    completed = 0

    with tqdm(total=len(jobs), desc="CALM experiments", unit="job", dynamic_ncols=True) as bar:
        for i, item in enumerate(jobs, start=1):
            bar.set_postfix_str(item["name"][:48], refresh=True)
            if skip_completed and done(item["output_json"]):
                clear_running(item)
                append_log(results_dir, f"SKIP {i}/{len(jobs)} {item['name']} (json exists)")
                skipped += 1
                bar.update(1)
                continue

            ok, code, _elapsed = run_one(item, results_dir, i, len(jobs))
            if ok:
                completed += 1
            else:
                failed.append({"name": item["name"], "code": code})
                write_status(
                    results_dir,
                    {
                        "failed": failed,
                        "stopped_at": item["name"],
                        "index": i,
                        "total": len(jobs),
                        "resume": skip_completed,
                    },
                )
                if args.stop_on_fail:
                    bar.update(1)
                    break
            bar.update(1)

    aggregate(results_dir, args.python)
    write_status(
        results_dir,
        {
            "finished": True,
            "failed": failed,
            "completed": completed,
            "skipped": skipped,
            "total": len(jobs),
            "resume": skip_completed,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    append_log(
        results_dir,
        f"Done. completed={completed} skipped={skipped} failed={len(failed)}",
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
