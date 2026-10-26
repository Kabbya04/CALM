# CALM-v2, non-ImageNet datasets

`final-others.py` runs 5-way 5-shot few-shot (600 episodes) and sequential
continual tasks on the official test split.

| `--datasetname` | tasks |
|---|---:|
| cifar10 | 5 |
| cifar100 | 10 |
| stl10 | 5 |
| svhn | 5 |
| fashionmnist | 5 |

```powershell
conda activate CALM
cd E:\Academics\CALM\CALM\CALM-v2\final-eval-other
python final-others.py --datasetname cifar100 --device cuda --seed 42
python final-others.py --datasetname cifar100 --device cuda --seed 42 --single_backbone
python final-others.py --datasetname cifar100 --device cuda --seed 42 --age_mode per_example --train_compression --log_memory
```

Datasets download into `./data` unless `--data_root` is set. ImageNet is
not stored here; use `../imagenet-eval/` with `--imagenet_root`.
