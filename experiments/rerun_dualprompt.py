"""Rerun only the three DualPrompt ImageNet jobs, one seed at a time."""

import os
import subprocess
import sys

PY = r"C:\Users\aiden\miniconda3\envs\CALM\python.exe"
IN = r"D:\Datasets\imagenet_torchvision"
RES = r"E:\Academics\CALM\CALM\results"
EXP = r"E:\Academics\CALM\CALM\experiments"

def main():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    failed = 0
    for seed in (42, 43, 44):
        out = os.path.join(RES, f"dualprompt_imagenet_continual_seed{seed}.json")
        log_path = os.path.join(RES, f"2_dualprompt_seed{seed}.log")
        print(f"START dualprompt seed {seed}", flush=True)
        with open(log_path, "w", encoding="utf-8") as handle:
            proc = subprocess.run(
                [PY, "run_dualprompt.py", "--imagenet_root", IN, "--num_tasks", "10",
                 "--seed", str(seed), "--device", "cuda", "--output_json", out],
                cwd=EXP,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        ok = proc.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 20
        print(f"{'OK' if ok else 'FAIL'} dualprompt seed {seed} code={proc.returncode}", flush=True)
        if not ok:
            failed += 1
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
