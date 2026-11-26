# CALM prep (new machine)

Full command checklist with comments: see [`RUN_SEQUENCE.md`](RUN_SEQUENCE.md).

## 1. Conda + CUDA packages

```bash
conda env create -f environment.yml
conda activate CALM
python cuda-check.py
```

Or stepwise:

```bash
conda create -n CALM python=3.10 -y
conda activate CALM
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python cuda-check.py
```

Use `cu121` / `cu118` wheels if that matches your driver. `cuda-check.py` must print **PASS** before experiments.

## 2. ImageNet validation layout

Expect:

```text
<imagenet_root>/val/<wnid>/*.JPEG
```

If validation images are still flat under `val/`:

```bash
python dataset-sort.py --imagenet_root D:\Datasets\imagenet_torchvision --kaggle_root D:\Datasets\imagenet_torchvision
```

`--kaggle_root` must contain `LOC_val_solution.csv`. The script is idempotent if class folders already exist.

## 3. Run the experiment campaign

```bash
python run_all_experiments.py --imagenet_root D:\Datasets\imagenet_torchvision
```

**Resume is on by default.** After a crash or load-shedding outage, re-run the same command; finished JSONs in `CALM/results/` are skipped and interrupted jobs retry.

Useful flags:

| Flag | Meaning |
|---|---|
| `--suite correction` | CIFAR-10/100 S0–S4 only (first fork) |
| `--suite all` | Full post-fix campaign (default) |
| `--seeds 42` | Smoke one seed |
| `--resume` | Skip completed jobs (default) |
| `--force` / `--no-resume` | Re-run even if JSON exists |
| `--dry_run` | Print DONE/PEND job list |
| `--stop_on_fail` | Abort after first failure |

JSON + per-job `.log` files go to `CALM/results/`. Fixed ensemble-v2 runs use the `calm_v2pe_*` / `ready_v2pe_*` names so older broken-v2 files stay archived.
