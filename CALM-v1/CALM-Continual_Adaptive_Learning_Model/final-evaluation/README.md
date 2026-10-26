# CALM-v1 ImageNet runners

`--imagenet_root` is the existing preprocessed ImageNet directory
(for example `D:\Datasets\imagenet_torchvision`). Do not rearrange it.

```powershell
conda activate CALM
cd E:\Academics\CALM\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation

python run_imagenet_evaluation.py --imagenet_root D:\Datasets\imagenet_torchvision --n_way 5 --k_shot 5 --num_query 15 --num_episodes 600 --seed 42 --device cuda

python run_imagenet_continual.py --imagenet_root D:\Datasets\imagenet_torchvision --num_tasks 10 --k_shot 5 --seed 42 --device cuda
```

`preprocess_val.py` only if the val folder is still in the original
Kaggle layout. Skip it if `val/<wnid>/*.JPEG` is already in place.
