# CALM-v2

Ensemble CLIP (ViT-B/32 + ViT-B/16) and hierarchical episodic memory
(recent 512-d / medium 256-d / old 128-d).

Paper runners: `imagenet-eval/` and `final-eval-other/`. Shared memory and
ensemble code: `episodic-memory/`. ImageNet is still the existing
`--imagenet_root` tree. Do not rearrange it.

## Layout

```
episodic-memory/
  memory.py                    # HierarchicalEpisodicMemory
  ensemble_calibration.py
imagenet-eval/
  run_imagenet_evaluation.py
  run_imagenet_continual.py
final-eval-other/
  final-others.py
readiness-agent/
  evaluate_agent.py
```

`episodic-memory/run_continual_learning.py` is the older CIFAR trainer.
Use the ImageNet and `final-others` runners for paper numbers.

## Defaults that hid compression

`increment_episode()` used to run once per task, with `tau_recent=100`.
A 10-task stream never left the recent tier. For migration, pass
`--age_mode per_example` (or lower `--tau_recent` / `--tau_medium`).
`--train_compression` fits the autoencoders before the stream.
`--log_memory` prints tier counts and stored bytes.

## ImageNet

```powershell
conda activate CALM
cd imagenet-eval
python run_imagenet_evaluation.py --imagenet_root D:\Datasets\imagenet_torchvision --seed 42 --device cuda
python run_imagenet_continual.py --imagenet_root D:\Datasets\imagenet_torchvision --num_tasks 10 --k_shot 5 --seed 42 --device cuda --age_mode per_example --train_compression --log_memory
```

## Other datasets

```powershell
cd final-eval-other
python final-others.py --datasetname cifar100 --device cuda --seed 42
python final-others.py --datasetname cifar100 --device cuda --seed 42 --single_backbone
python final-others.py --datasetname cifar100 --device cuda --seed 42 --flat_memory --age_mode per_example --train_compression
```

Comparison methods, matched readiness, and correction cost:
`CALM/experiments/`.
