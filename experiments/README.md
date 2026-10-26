# Experiments

Protocol-matched scripts for the ISCMI review gaps. They use the same
ImageNet root as CALM-v1/v2 (`--imagenet_root`, val split, 10 tasks × 100
classes, 5 supports per class). They do not rewrite that directory.

JSON records go to `CALM/results/` unless `--output_json` is set.

```powershell
conda activate CALM
cd E:\Academics\CALM\CALM\experiments
```

Install DualPrompt extras once:

```powershell
python -m pip install timm
```

## ImageNet comparisons (keep DualPrompt and iCaRL)

```powershell
$IN = "D:\Datasets\imagenet_torchvision"

python run_clip_zeroshot.py --imagenet_root $IN --seed 42 --device cuda
python run_tip_adapter.py --imagenet_root $IN --seed 42 --device cuda
python run_icarl.py --imagenet_root $IN --seed 42 --device cuda --backbone clip
python run_dualprompt.py --imagenet_root $IN --seed 42 --device cuda
```

Repeat with `--seed 43` and `--seed 44`.

iCaRL default is frozen CLIP features, herding of `k` exemplars per class,
and a distilled linear head. `--backbone resnet18` trains a ResNet-18
encoder on the same val images.

DualPrompt freezes a timm ViT-B/16, trains G/E prompts and a classifier
head, and selects the E-prompt with query-key matching.

CALM-v1 / v2 numbers still come from the original runners:

```powershell
cd ..\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation
python run_imagenet_continual.py --imagenet_root $IN --seed 42 --device cuda

cd ..\..\..\CALM-v2\imagenet-eval
python run_imagenet_continual.py --imagenet_root $IN --seed 42 --device cuda
```

## Readiness (coverage, deferral, deployed error)

One run writes the curve for `--rho 0.5 0.7 0.85 1.0`.

```powershell
python run_readiness.py --variant v1 --dataset cifar10 --k_shot 5 --seed 42 --device cuda
python run_readiness.py --variant v2 --dataset cifar10 --k_shot 5 --seed 42 --device cuda
python run_readiness.py --variant v2 --dataset cifar10 --k_shot 5 --seed 42 --device cuda --use_disagreement
```

## Correction cost

```powershell
python run_correction_cost.py --dataset cifar10 --mode none --seed 42 --device cuda
python run_correction_cost.py --dataset cifar10 --mode threshold --rho 0.7 --seed 42 --device cuda
python run_correction_cost.py --dataset cifar10 --mode oracle --seed 42 --device cuda
python run_correction_cost.py --dataset cifar10 --mode inject --error_rate 0.1 --seed 42 --device cuda
```

## Aggregate seeds

```powershell
python aggregate_results.py --results_dir E:\Academics\CALM\CALM\results
```
