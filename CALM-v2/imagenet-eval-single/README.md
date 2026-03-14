# CALM-v2 ImageNet Evaluation (Single-Backbone Variant)

This folder contains an **alternate ImageNet evaluation setup** for CALM-v2 that
uses **only a single CLIP backbone (the first variant, typically ViT-B/32)** for
both building episodic memory and classifying queries.

The goal is to:

- Remove the embedding-space mismatch between ViT-B/32 and ViT-B/16.
- Make the results **directly comparable to CALM-v1**, which also used a single
  CLIP ViT-B/32 backbone for episodic memory.

## Files

- `run_imagenet_evaluation.py`  
  Few-shot 5-way 5-shot ImageNet evaluation (600 episodes by default) using:
  - `HierarchicalEpisodicMemory` (shared with CALM-v2 core)
  - Only the **first CLIP variant** from `EnsembleCLIPModels` for both support
    and query embeddings.

- `run_imagenet_continual.py`  
  10-task continual ImageNet evaluation using:
  - `HierarchicalEpisodicMemory`
  - Only the **first CLIP variant** for both memory construction and testing.

- `imagenet_sampler.py`  
  Episodic sampler for N-way K-shot episodes (copied from the main ImageNet eval).

- `memory.py`  
  Thin wrapper that re-exports `HierarchicalEpisodicMemory` from
  `CALM-v2/episodic-memory/memory.py`.

## Few-Shot Usage (5-way 5-shot)

From `CALM-v2/imagenet-eval-single`:

```bash
python run_imagenet_evaluation.py \
  --imagenet_root D:\Datasets\imagenet_torchvision \
  --n_way 5 \
  --k_shot 5 \
  --num_query 15 \
  --num_episodes 600 \
  --temperature 1.0
```

## Continual Usage (10 tasks, 5-shot)

```bash
python run_imagenet_continual.py \
  --imagenet_root D:\Datasets\imagenet_torchvision \
  --num_tasks 10 \
  --k_shot 5 \
  --batch_size 128 \
  --temperature 1.0
```

You can adjust `--temperature` (e.g., `0.8` or `0.7`) if you want to explore the
effect of softmax sharpening on accuracy and BWT.

