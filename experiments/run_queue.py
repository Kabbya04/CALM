"""Run the test-order.md jobs one at a time on the CALM CUDA env."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

PY = r"C:\Users\aiden\miniconda3\envs\CALM\python.exe"
IN = r"D:\Datasets\imagenet_torchvision"
RES = r"E:\Academics\CALM\CALM\results"
V1 = r"E:\Academics\CALM\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation"
V1O = os.path.join(V1, "final-eval-other")
V2I = r"E:\Academics\CALM\CALM\CALM-v2\imagenet-eval"
V2O = r"E:\Academics\CALM\CALM\CALM-v2\final-eval-other"
EXP = r"E:\Academics\CALM\CALM\experiments"
LOG = os.path.join(RES, "queue.log")
STATUS = os.path.join(RES, "queue_status.json")
SEEDS = (42, 43, 44)
DATASETS = ("cifar10", "cifar100", "stl10", "svhn", "fashionmnist")
BYTES = (("512k", 524288), ("1mb", 1048576), ("2mb", 2097152), ("4mb", 4194304))


def job(name, cwd, args, output_json):
    return {"name": name, "cwd": cwd, "args": args, "output_json": output_json}


def build_jobs():
    jobs = []

    for s in SEEDS:
        out = os.path.join(RES, f"calm_v1_imagenet_fewshot_seed{s}.json")
        jobs.append(job(
            f"1_v1_imagenet_fewshot_seed{s}",
            V1,
            [PY, "run_imagenet_evaluation.py", "--imagenet_root", IN, "--n_way", "5",
             "--k_shot", "5", "--num_query", "15", "--num_episodes", "600",
             "--seed", str(s), "--device", "cuda", "--output_json", out],
            out,
        ))
    for s in SEEDS:
        out = os.path.join(RES, f"calm_v2_imagenet_fewshot_seed{s}.json")
        jobs.append(job(
            f"1_v2_imagenet_fewshot_seed{s}",
            V2I,
            [PY, "run_imagenet_evaluation.py", "--imagenet_root", IN, "--n_way", "5",
             "--k_shot", "5", "--num_query", "15", "--num_episodes", "600",
             "--seed", str(s), "--device", "cuda", "--temperature", "1.0",
             "--output_json", out],
            out,
        ))

    for s in SEEDS:
        out = os.path.join(RES, f"clip_zeroshot_imagenet_continual_seed{s}.json")
        jobs.append(job(
            f"2_clip_zeroshot_seed{s}",
            EXP,
            [PY, "run_clip_zeroshot.py", "--imagenet_root", IN, "--num_tasks", "10",
             "--seed", str(s), "--device", "cuda", "--output_json", out],
            out,
        ))
        out = os.path.join(RES, f"tip_adapter_imagenet_continual_seed{s}.json")
        jobs.append(job(
            f"2_tip_adapter_seed{s}",
            EXP,
            [PY, "run_tip_adapter.py", "--imagenet_root", IN, "--num_tasks", "10",
             "--k_shot", "5", "--seed", str(s), "--device", "cuda", "--output_json", out],
            out,
        ))
        out = os.path.join(RES, f"icarl_clip_imagenet_continual_seed{s}.json")
        jobs.append(job(
            f"2_icarl_seed{s}",
            EXP,
            [PY, "run_icarl.py", "--imagenet_root", IN, "--num_tasks", "10",
             "--k_shot", "5", "--seed", str(s), "--device", "cuda", "--backbone", "clip",
             "--output_json", out],
            out,
        ))
        out = os.path.join(RES, f"dualprompt_imagenet_continual_seed{s}.json")
        jobs.append(job(
            f"2_dualprompt_seed{s}",
            EXP,
            [PY, "run_dualprompt.py", "--imagenet_root", IN, "--num_tasks", "10",
             "--seed", str(s), "--device", "cuda", "--output_json", out],
            out,
        ))
    for s in SEEDS:
        out = os.path.join(RES, f"calm_v1_imagenet_continual_seed{s}.json")
        jobs.append(job(
            f"2_v1_imagenet_continual_seed{s}",
            V1,
            [PY, "run_imagenet_continual.py", "--imagenet_root", IN, "--num_tasks", "10",
             "--k_shot", "5", "--seed", str(s), "--device", "cuda", "--output_json", out],
            out,
        ))
    for s in SEEDS:
        out = os.path.join(RES, f"calm_v2_imagenet_continual_seed{s}.json")
        jobs.append(job(
            f"2_v2_imagenet_continual_seed{s}",
            V2I,
            [PY, "run_imagenet_continual.py", "--imagenet_root", IN, "--num_tasks", "10",
             "--k_shot", "5", "--seed", str(s), "--device", "cuda", "--temperature", "1.0",
             "--output_json", out],
            out,
        ))

    for d in DATASETS:
        for s in SEEDS:
            out = os.path.join(RES, f"calm_v1_{d}_seed{s}.json")
            jobs.append(job(
                f"3_v1_{d}_seed{s}",
                V1O,
                [PY, "final-others.py", "--datasetname", d, "--device", "cuda",
                 "--seed", str(s), "--output_json", out],
                out,
            ))
    for d in DATASETS:
        for s in SEEDS:
            out = os.path.join(RES, f"calm_v2_{d}_seed{s}.json")
            jobs.append(job(
                f"3_v2_{d}_seed{s}",
                V2O,
                [PY, "final-others.py", "--datasetname", d, "--device", "cuda",
                 "--seed", str(s), "--temperature", "1.0", "--output_json", out],
                out,
            ))

    for s in SEEDS:
        rows = [
            (f"ablate_B1_hier_nomigrate_cifar100_seed{s}.json",
             ["--single_backbone", "--tau_recent", "1000000000"]),
            (f"ablate_B2_compress_cifar100_seed{s}.json",
             ["--single_backbone", "--train_compression", "--age_mode", "per_example", "--log_memory"]),
            (f"ablate_B3_ensemble_flat_cifar100_seed{s}.json",
             ["--flat_memory"]),
            (f"ablate_B6_single_backbone_cifar100_seed{s}.json",
             ["--single_backbone"]),
            (f"ablate_B7_temp07_cifar100_seed{s}.json",
             ["--temperature", "0.7"]),
        ]
        for fname, extra in rows:
            out = os.path.join(RES, fname)
            jobs.append(job(
                f"4_{fname[:-5]}",
                V2O,
                [PY, "final-others.py", "--datasetname", "cifar100", "--device", "cuda",
                 "--seed", str(s), *extra, "--output_json", out],
                out,
            ))

    for s in SEEDS:
        out = os.path.join(RES, f"ablate_B6_single_backbone_imagenet_seed{s}.json")
        jobs.append(job(
            f"4b_B6_imagenet_seed{s}",
            V2I,
            [PY, "run_imagenet_continual.py", "--imagenet_root", IN, "--num_tasks", "10",
             "--k_shot", "5", "--seed", str(s), "--device", "cuda", "--single_backbone",
             "--output_json", out],
            out,
        ))
        out = os.path.join(RES, f"ablate_B2_compress_imagenet_seed{s}.json")
        jobs.append(job(
            f"4b_B2_imagenet_seed{s}",
            V2I,
            [PY, "run_imagenet_continual.py", "--imagenet_root", IN, "--num_tasks", "10",
             "--k_shot", "5", "--seed", str(s), "--device", "cuda", "--single_backbone",
             "--train_compression", "--age_mode", "per_example", "--log_memory",
             "--output_json", out],
            out,
        ))

    for s in SEEDS:
        rows = [
            (f"compress_C1_untrained_cifar100_seed{s}.json",
             ["--age_mode", "per_example", "--log_memory"]),
            (f"compress_C2_trained_cifar100_seed{s}.json",
             ["--age_mode", "per_example", "--train_compression", "--log_memory"]),
            (f"compress_C3_longstream_cifar100_seed{s}.json",
             ["--num_tasks", "20", "--age_mode", "per_example", "--train_compression", "--log_memory"]),
        ]
        for fname, extra in rows:
            out = os.path.join(RES, fname)
            jobs.append(job(
                f"5_{fname[:-5]}",
                V2O,
                [PY, "final-others.py", "--datasetname", "cifar100", "--device", "cuda",
                 "--seed", str(s), *extra, "--output_json", out],
                out,
            ))

    for name, nbytes in BYTES:
        for s in SEEDS:
            out = os.path.join(RES, f"bytes_Dflat_cifar100_{name}_seed{s}.json")
            jobs.append(job(
                f"6_Dflat_{name}_seed{s}",
                V1O,
                [PY, "final-others.py", "--datasetname", "cifar100", "--device", "cuda",
                 "--seed", str(s), "--max_bytes", str(nbytes), "--output_json", out],
                out,
            ))
    for name, nbytes in BYTES:
        for s in SEEDS:
            out = os.path.join(RES, f"bytes_Dhier_cifar100_{name}_seed{s}.json")
            jobs.append(job(
                f"6_Dhier_{name}_seed{s}",
                V2O,
                [PY, "final-others.py", "--datasetname", "cifar100", "--device", "cuda",
                 "--seed", str(s), "--single_backbone", "--train_compression",
                 "--age_mode", "per_example", "--max_bytes", str(nbytes), "--log_memory",
                 "--output_json", out],
                out,
            ))

    for d in ("cifar10", "cifar100"):
        for s in SEEDS:
            out = os.path.join(RES, f"ready_v1_{d}_seed{s}.json")
            jobs.append(job(
                f"7_ready_v1_{d}_seed{s}",
                EXP,
                [PY, "run_readiness.py", "--variant", "v1", "--dataset", d, "--k_shot", "5",
                 "--seed", str(s), "--device", "cuda", "--output_json", out],
                out,
            ))
            out = os.path.join(RES, f"ready_v2_{d}_seed{s}.json")
            jobs.append(job(
                f"7_ready_v2_{d}_seed{s}",
                EXP,
                [PY, "run_readiness.py", "--variant", "v2", "--dataset", d, "--k_shot", "5",
                 "--seed", str(s), "--device", "cuda", "--output_json", out],
                out,
            ))
            out = os.path.join(RES, f"ready_v2dis_{d}_seed{s}.json")
            jobs.append(job(
                f"7_ready_v2dis_{d}_seed{s}",
                EXP,
                [PY, "run_readiness.py", "--variant", "v2", "--dataset", d, "--k_shot", "5",
                 "--seed", str(s), "--device", "cuda", "--use_disagreement",
                 "--output_json", out],
                out,
            ))

    for s in SEEDS:
        corr = [
            (f"corr_F2_none_cifar10_seed{s}.json", ["--mode", "none"]),
            (f"corr_F1_threshold_cifar10_seed{s}.json", ["--mode", "threshold", "--rho", "0.7"]),
            (f"corr_F3_oracle_cifar10_seed{s}.json", ["--mode", "oracle"]),
            (f"corr_F4_e05_cifar10_seed{s}.json", ["--mode", "inject", "--error_rate", "0.05"]),
            (f"corr_F4_e10_cifar10_seed{s}.json", ["--mode", "inject", "--error_rate", "0.10"]),
            (f"corr_F4_e20_cifar10_seed{s}.json", ["--mode", "inject", "--error_rate", "0.20"]),
        ]
        for fname, extra in corr:
            out = os.path.join(RES, fname)
            jobs.append(job(
                f"8_{fname[:-5]}",
                EXP,
                [PY, "run_correction_cost.py", "--dataset", "cifar10", "--seed", str(s),
                 "--device", "cuda", *extra, "--output_json", out],
                out,
            ))
    return jobs


def log(msg):
    os.makedirs(RES, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_status(payload):
    with open(STATUS, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def done(path):
    return os.path.isfile(path) and os.path.getsize(path) > 20


def run_one(item, index, total):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    log_path = os.path.join(RES, f"{item['name']}.log")
    log(f"START {index}/{total} {item['name']}")
    write_status({
        "current": item["name"],
        "index": index,
        "total": total,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "output_json": item["output_json"],
    })
    t0 = time.time()
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
    log(f"{'OK' if ok else 'FAIL'} {index}/{total} {item['name']} code={proc.returncode} {elapsed:.1f}s")
    return ok, proc.returncode


def main():
    os.makedirs(RES, exist_ok=True)
    jobs = build_jobs()
    log(f"Queue has {len(jobs)} jobs. CUDA device 0. Sequential only.")
    failed = []
    for i, item in enumerate(jobs, start=1):
        if done(item["output_json"]):
            log(f"SKIP {i}/{len(jobs)} {item['name']} (json exists)")
            continue
        ok, code = run_one(item, i, len(jobs))
        if not ok:
            failed.append({"name": item["name"], "code": code})
            write_status({"failed": failed, "stopped_at": item["name"], "index": i, "total": len(jobs)})
            # Keep going only after a failure log; still sequential, do not abort
            # the whole campaign unless CUDA is missing.
            if code == 0:
                continue

    log("GPU queue finished. Running aggregator.")
    agg = subprocess.run(
        [PY, "aggregate_results.py", "--results_dir", RES],
        cwd=EXP,
        capture_output=True,
        text=True,
    )
    with open(os.path.join(RES, "aggregate.txt"), "w", encoding="utf-8") as handle:
        handle.write(agg.stdout or "")
        handle.write(agg.stderr or "")
    write_status({
        "finished": True,
        "failed": failed,
        "total": len(jobs),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    })
    log(f"Done. failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
