# CALM-v1

Frozen CLIP ViT-B/32, flat episodic memory, and prototypical classification.
Paper runners live in `final-evaluation/`. Pass the existing ImageNet tree
with `--imagenet_root` and leave that directory as it is.

## Layout

```
final-evaluation/
  run_imagenet_evaluation.py   # 5-way 5-shot ImageNet
  run_imagenet_continual.py    # 10-task ImageNet AvgAcc / BWT
  imagenet_sampler.py
  preprocess_val.py            # one-time val folder layout helper
  memory.py
  final-eval-other/            # CIFAR-10/100, STL-10, SVHN, FashionMNIST
readiness-agent/
  evaluate_agent.py
```

Older exploratory folders (dynamic-eval, RLHF, readiness-agent-v2) were removed.

## Environment

```powershell
conda activate CALM
```

PyTorch 2.3.1+cu121, Python 3.11. Interpreter:

`C:\Users\aiden\miniconda3\envs\CALM\python.exe`

## ImageNet

Use the existing preprocessed root, for example
`D:\Datasets\imagenet_torchvision`, with `val/<wnid>/*.JPEG`.
`--imagenet_root` must point at that root.

```powershell
cd final-evaluation
python run_imagenet_evaluation.py --imagenet_root D:\Datasets\imagenet_torchvision --seed 42 --device cuda --output_json ..\..\..\results\v1_imagenet_fs_seed42.json
python run_imagenet_continual.py --imagenet_root D:\Datasets\imagenet_torchvision --num_tasks 10 --k_shot 5 --seed 42 --device cuda --output_json ..\..\..\results\v1_imagenet_cl_seed42.json
```

## Other datasets

```powershell
cd final-evaluation\final-eval-other
python final-others.py --datasetname cifar100 --device cuda --seed 42 --output_json ..\..\..\..\results\v1_cifar100_seed42.json
```

`--datasetname`: `cifar10`, `cifar100`, `stl10`, `svhn`, `fashionmnist`.

New comparison, readiness, and correction scripts are in `CALM/experiments/`.
