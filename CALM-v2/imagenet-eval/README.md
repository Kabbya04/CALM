# CALM-v2 ImageNet runners

Same preprocessed ImageNet tree as CALM-v1. Pass it with `--imagenet_root`.
Do not change the folder layout.

```powershell
conda activate CALM
cd E:\Academics\CALM\CALM\CALM-v2\imagenet-eval

python run_imagenet_evaluation.py --imagenet_root D:\Datasets\imagenet_torchvision --n_way 5 --k_shot 5 --num_query 15 --num_episodes 600 --seed 42 --device cuda

python run_imagenet_continual.py --imagenet_root D:\Datasets\imagenet_torchvision --num_tasks 10 --k_shot 5 --seed 42 --device cuda --temperature 1.0
```

Useful continual flags:

- `--age_mode per_example` — advance memory age after each stored support so medium/old can fill
- `--train_compression` — fit the 512→256 and 512→128 codecs first
- `--flat_memory` — store 512-d only
- `--single_backbone` — score with ViT-B/32 only
- `--log_memory` — print tier counts and bytes
- `--max_bytes N` — evict oldest codes past a byte budget
- `--output_json path` — write one JSON record

Memory and ensemble code stay in `../episodic-memory/`.
