# CALM remote-device run sequence

Use this checklist on the GPU machine (conda env + ImageNet + experiment campaign).
Commands assume the repo root contains the `CALM/` folder.

Replace `D:\Datasets\imagenet_torchvision` with the local ImageNet path if different.

---

## 0. One-time environment setup

```bash
# Create the CALM env from the YAML (Python 3.10 + CUDA 12.4 torch wheels + deps)
conda env create -f CALM/prep/environment.yml

# Activate before every session
conda activate CALM
```

If the env already exists and you only need to refresh packages:

```bash
conda activate CALM
conda env update -f CALM/prep/environment.yml --prune
```

---

## 1. Verify CUDA before any experiment

```bash
conda activate CALM
cd CALM/prep
python cuda-check.py
# Must print PASS. If FAIL, fix the CUDA torch wheel / driver before continuing.
```

---

## 2. Prepare ImageNet validation layout

Expected tree:

```text
<imagenet_root>/val/<wnid>/*.JPEG
```

If `val/` still has flat `*.JPEG` files:

```bash
conda activate CALM
cd CALM/prep

# --kaggle_root must contain LOC_val_solution.csv (often the same folder)
python dataset-sort.py ^
  --imagenet_root D:\Datasets\imagenet_torchvision ^
  --kaggle_root D:\Datasets\imagenet_torchvision
# Idempotent: safe to re-run if class folders already exist.
```

On Linux/macOS use `\` line breaks instead of `^`.

---

## 3. (Optional) Smoke the correction fork first

Decides method vs measurement paper before the long ImageNet queue:

```bash
conda activate CALM
cd CALM/prep

python run_all_experiments.py ^
  --imagenet_root D:\Datasets\imagenet_torchvision ^
  --suite correction ^
  --seeds 42
# Writes corr_S0..S4_* JSON under CALM/results/
```

---

## 4. Full experiment campaign

```bash
conda activate CALM
cd CALM/prep

# Preview what will run / skip (no GPU work)
python run_all_experiments.py ^
  --imagenet_root D:\Datasets\imagenet_torchvision ^
  --suite all ^
  --dry_run

# Run everything (resume is ON by default)
python run_all_experiments.py ^
  --imagenet_root D:\Datasets\imagenet_torchvision ^
  --suite all
```

Useful suite values: `correction`, `imagenet_baselines`, `imagenet_calm`,
`others_v2pe`, `ablations`, `readiness`, `all`.

Outputs:

| Path | Meaning |
|---|---|
| `CALM/results/*.json` | Per-job metrics (atomic write) |
| `CALM/results/*.json.running` | Present only while a job is in flight |
| `CALM/results/<job>.log` | Captured stdout/stderr for that job |
| `CALM/results/run_all.log` | Campaign timeline |
| `CALM/results/run_all_status.json` | Current / last status snapshot |
| `CALM/results/aggregate.txt` | Mean±std summary after the queue finishes |

---

## 5. After a crash / load-shedding outage (fail-safe resume)

**Do nothing special.** Re-run the same command:

```bash
conda activate CALM
cd CALM/prep

python run_all_experiments.py ^
  --imagenet_root D:\Datasets\imagenet_torchvision ^
  --suite all
```

What happens automatically:

1. Jobs with a **valid** result JSON are **skipped** (`--resume` is the default).
2. Jobs interrupted mid-run (leftover `*.json.running`) are **retried**.
3. Truncated / schema-incomplete JSON is treated as incomplete and **retried**.
4. Progress bar + `run_all.log` show skipped vs pending vs failed.

Only use `--force` if you intentionally want to recompute finished jobs:

```bash
python run_all_experiments.py --imagenet_root D:\Datasets\imagenet_torchvision --suite all --force
```

`--no-resume` also disables skipping (same scheduling effect as `--force`).

---

## 6. Aggregate later (optional)

If you only need to refresh the summary table:

```bash
conda activate CALM
cd CALM/experiments
python aggregate_results.py --results_dir ..\results
```

---

## Notes for the device owner

- Keep the `CALM/results/` directory on a persistent disk (not a volatile temp folder).
- One job runs at a time so GPU memory is released between subprocesses.
- A kill **during** a long ImageNet/DualPrompt job still restarts that **one** job from the beginning; completed siblings stay skipped.
- Prefer starting with `--suite correction --seeds 42` if power is unreliable, then continue with `--suite all`.
