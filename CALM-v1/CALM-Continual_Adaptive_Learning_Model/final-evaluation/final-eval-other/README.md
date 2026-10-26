# CALM-v1, non-ImageNet datasets

`final-others.py` runs 5-way 5-shot few-shot (600 episodes) and sequential
continual tasks on the official test split. Encoder: frozen CLIP ViT-B/32.
Memory: flat embeddings.

| `--datasetname` | tasks |
|---|---:|
| cifar10 | 5 |
| cifar100 | 10 |
| stl10 | 5 |
| svhn | 5 |
| fashionmnist | 5 |

```powershell
conda activate CALM
cd E:\Academics\CALM\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation\final-eval-other
python final-others.py --datasetname cifar10 --device cuda --seed 42
python final-others.py --datasetname cifar100 --device cuda --seed 42 --output_json ..\..\..\..\results\v1_cifar100_seed42.json
```

Datasets download into `./data` unless `--data_root` is set.
