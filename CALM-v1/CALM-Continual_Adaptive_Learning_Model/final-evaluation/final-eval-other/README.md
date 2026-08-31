# CALM-v1 final evaluation on other datasets

`final-others.py` runs both final evaluation protocols used for the ImageNet
experiment:

1. **Few-shot:** 5-way, 5-shot, 15 query images per class, 600 episodes.
2. **Continual:** sequential class tasks, 5-shot memory per class, average
   accuracy, and backward transfer (BWT).

CALM-v1 uses a frozen CLIP ViT-B/32 backbone and flat episodic memory.
The official test split is used for these datasets, analogous to the ImageNet
validation split used by the ImageNet runner.

## Supported datasets

| Dataset | Continual tasks |
|---|---:|
| `cifar10` | 5 |
| `cifar100` | 10 |
| `stl10` | 5 |
| `svhn` | 5 |
| `fashionmnist` | 5 |

## GPU setup

The tested project-compatible environment is Python 3.11 with CUDA-enabled
PyTorch 2.3.1:

```powershell
conda activate CALM
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers Pillow numpy tqdm scikit-learn
```

Verify CUDA before starting an evaluation:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output includes:

```text
2.3.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Ti
```

In Cursor, select this interpreter:

```text
C:\Users\aiden\miniconda3\envs\CALM\python.exe
```

## Run commands

Run from this directory:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation\final-eval-other
```

Run each dataset with CUDA:

```powershell
python final-others.py --datasetname cifar10 --device cuda
python final-others.py --datasetname cifar100 --device cuda
python final-others.py --datasetname stl10 --device cuda
python final-others.py --datasetname svhn --device cuda
python final-others.py --datasetname fashionmnist --device cuda
```

Each command downloads its dataset into the local `data/` directory if
necessary, then runs both few-shot and continual evaluation.

Useful options:

```powershell
python final-others.py --datasetname cifar10 --device cuda --batch_size 32 --seed 42
python final-others.py --datasetname cifar100 --device cuda --num_tasks 10
python final-others.py --datasetname cifar10 --device cuda --num_episodes 600
```

Use `--data_root <path>` to choose a different dataset cache directory.
The default `--device auto` may fall back to CPU; use `--device cuda` for the
final GPU evaluation.
