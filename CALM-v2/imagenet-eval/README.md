# CALM-v2 ImageNet Evaluation

This folder contains the **ImageNet-aligned few-shot evaluation** for CALM-v2,
mirroring the protocol used in `CALM-v1/CALM-Continual_Adaptive_Learning_Model/final-evaluation`.

It reuses the **same preprocessed ImageNet directory** you used for CALM-v1, so no
dataset changes are required.

## Files

- `run_imagenet_evaluation.py`  
  Runs N-way K-shot ImageNet episodes using:
  - `HierarchicalEpisodicMemory` from the CALM-v2 episodic-memory module
  - `EnsembleCLIPModels` (ViT-B/32, ViT-B/16) for ensemble embeddings

- `imagenet_sampler.py`  
  A minimal copy of the CALM-v1 ImageNet sampler to index the existing
  ImageNet tree and provide N-way, K-shot episodes.

## Usage

From the `CALM-v2/imagenet-eval` directory (or with this directory on your
`PYTHONPATH`), run:

```bash
python run_imagenet_evaluation.py \
    --imagenet_root D:\Datasets\imagenet_torchvision \
    --n_way 5 \
    --k_shot 5 \
    --num_query 15 \
    --num_episodes 600
```

Adjust `--imagenet_root` to the same path you used for the CALM-v1 evaluation.

The script will print:

- Mean accuracy across episodes
- 95% confidence interval (same style as the v1 ImageNet evaluation)

