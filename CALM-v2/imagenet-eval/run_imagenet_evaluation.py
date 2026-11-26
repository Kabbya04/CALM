"""
ImageNet-aligned few-shot evaluation for CALM-v2.

This mirrors CALM-v1's `run_imagenet_evaluation.py` protocol
but uses:
  - HierarchicalEpisodicMemory (episodic-memory/memory.py)
  - EnsembleCLIPModels (episodic-memory/ensemble_calibration.py)

Dataset assumption:
  - Reuses the same prepared ImageNet directory tree as CALM-v1
    (e.g., `imagenet_root/val/<class_name>/*.JPEG`).

Example usage (same as CALM-v1, but from CALM-v2 folder):

  python run_imagenet_evaluation.py \
      --imagenet_root D:\\Datasets\\imagenet_torchvision \
      --k_shot 5 \
      --n_way 5 \
      --num_query 15 \
      --num_episodes 600
"""

import argparse
import json
import os
import ssl
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Make sure we can import the episodic-memory modules when running this script directly.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALM_V2_ROOT = os.path.dirname(THIS_DIR)
EPISODIC_DIR = os.path.join(CALM_V2_ROOT, "episodic-memory")

if EPISODIC_DIR not in sys.path:
    sys.path.insert(0, EPISODIC_DIR)

from memory import HierarchicalEpisodicMemory  # type: ignore
from ensemble_calibration import EnsembleCLIPModels  # type: ignore
from imagenet_sampler import ImageNetSampler


# --- SSL fix copied from CALM-v1 script to avoid cert issues when transformers download weights ---
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CALM-v2 on ImageNet (few-shot episodic).")
    parser.add_argument(
        "--imagenet_root",
        type=str,
        required=True,
        help="Root directory of the preprocessed ImageNet dataset (same as used for CALM-v1).",
    )
    parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode.")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of support examples per class.")
    parser.add_argument("--num_query", type=int, default=15, help="Number of query examples per class.")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=600,
        help="Number of episodes to average over for a stable result.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing embeddings.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for ensemble softmax (used when combining models).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--single_backbone", action="store_true")
    return parser.parse_args()


def get_best_device(requested: str = "auto") -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_memory_from_support(
    support_set: List[Tuple[object, int]],
    ensemble_models: EnsembleCLIPModels,
    memory: HierarchicalEpisodicMemory,
    device: str,
    batch_size: int,
    single_backbone: bool = False,
):
    """Store support embeddings: one tower if single_backbone, else per-encoder dicts."""
    support_images = [item[0] for item in support_set]
    support_labels = [item[1] for item in support_set]
    first_variant = list(ensemble_models.models.keys())[0]

    with torch.inference_mode():
        for i in range(0, len(support_images), batch_size):
            batch = support_images[i : i + batch_size]
            labels = support_labels[i : i + len(batch)]
            if single_backbone:
                features = ensemble_models.get_image_features(batch, variant=first_variant)
                for embedding, label in zip(features, labels):
                    memory.add_example(embedding.cpu(), label, importance=1.0, task_id=None)
            else:
                all_features = ensemble_models.get_image_features(batch)
                for j, label in enumerate(labels):
                    payload = {
                        name: tensor[j].cpu() for name, tensor in all_features.items()
                    }
                    memory.add_example(payload, label, importance=1.0, task_id=None)
                del all_features


def evaluate_episode(
    support_set: List[Tuple[object, int]],
    query_set: List[Tuple[object, int]],
    ensemble_models: EnsembleCLIPModels,
    device: str,
    batch_size: int,
    temperature: float,
    single_backbone: bool = False,
) -> float:
    """
    One N-way K-shot ImageNet episode.

    Ensemble mode stores and scores per-encoder prototype banks, then fuses scores.
    """
    memory = HierarchicalEpisodicMemory(device=device)
    build_memory_from_support(
        support_set, ensemble_models, memory, device, batch_size, single_backbone
    )

    first_variant = list(ensemble_models.models.keys())[0]
    if single_backbone:
        prototypes, class_ids = memory.get_prototypes(variant=first_variant)
        if prototypes.nelement() == 0:
            return 0.0
        prototypes = F.normalize(prototypes.to(device), p=2, dim=1)
        banks = None
    else:
        banks, class_ids = memory.get_prototypes_by_variant()
        if not class_ids:
            return 0.0
        prototypes = None

    query_images = [item[0] for item in query_set]
    query_labels = torch.tensor([item[1] for item in query_set], device=device)
    correct_predictions = 0

    with torch.inference_mode():
        for i in range(0, len(query_images), batch_size):
            batch = query_images[i : i + batch_size]
            if single_backbone:
                features = F.normalize(
                    ensemble_models.get_image_features(batch, variant=first_variant),
                    p=2,
                    dim=1,
                )
                ensemble_probs = F.softmax((features @ prototypes.T) / temperature, dim=1)
            else:
                all_features = ensemble_models.get_image_features(batch)
                similarities = ensemble_models.compute_prototype_similarities(
                    all_features, banks, class_ids
                )
                ensemble_probs, _ = ensemble_models.ensemble_confidence(
                    similarities, temperature=temperature
                )
                del all_features

            pred_indices = torch.argmax(ensemble_probs, dim=1)
            predicted_class_ids = torch.tensor([class_ids[p] for p in pred_indices], device=device)
            batch_labels = query_labels[i : i + len(batch)]
            correct_predictions += (predicted_class_ids == batch_labels).sum().item()

    return (correct_predictions / len(query_labels)) * 100.0


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_best_device(args.device)
    print(f"Using device: {device}")

    # 1) Initialize ensemble CLIP models (same models used elsewhere in CALM-v2)
    print("Initializing CLIP ensemble (CALM-v2)...")
    ensemble_models = EnsembleCLIPModels(device=device)

    # 2) Load ImageNet sampler using the same transform as CALM-v1
    #    CLIPProcessor will handle tensor conversion and normalization;
    #    here we only resize and center-crop, leaving images as PIL.
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    sampler = ImageNetSampler(root_path=args.imagenet_root, split="val", transform=transform)

    # 3) Run episodic evaluation
    print(f"\nStarting CALM-v2 ImageNet evaluation for {args.num_episodes} episodes...")
    all_accuracies: List[float] = []

    for episode_idx in range(args.num_episodes):
        support_set, query_set = None, None
        while support_set is None:
            support_set, query_set = sampler.sample_episode(args.n_way, args.k_shot, args.num_query)

        acc = evaluate_episode(
            support_set,
            query_set,
            ensemble_models,
            device,
            batch_size=args.batch_size,
            temperature=args.temperature,
            single_backbone=args.single_backbone,
        )
        all_accuracies.append(acc)

        if (episode_idx + 1) % 50 == 0 or episode_idx == 0:
            print(f"Episode {episode_idx + 1}/{args.num_episodes}: accuracy = {acc:.2f}%")

    # 4) Report final results with 95% confidence interval (same style as v1)
    mean_accuracy = float(np.mean(all_accuracies))
    std_dev = float(np.std(all_accuracies))
    confidence_interval = 1.96 * std_dev / np.sqrt(args.num_episodes)

    print("\n--- Final ImageNet Few-Shot Evaluation Results (CALM-v2) ---")
    print(f"Configuration: {args.n_way}-way, {args.k_shot}-shot with {args.num_query} query images per class.")
    print(f"Evaluated on {args.num_episodes} episodes.")
    print("-" * 60)
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"95% Confidence Interval: +/- {confidence_interval:.2f}%")
    print("-" * 60)
    if args.output_json:
        calm_root = os.path.dirname(CALM_V2_ROOT)
        if calm_root not in sys.path:
            sys.path.insert(0, calm_root)
        from atomic_json import atomic_write_json

        atomic_write_json(
            args.output_json,
            {
                "variant": (
                    "calm_v2_single" if args.single_backbone else "calm_v2pe"
                ),
                "dataset": "imagenet",
                "protocol": "few_shot",
                "seed": args.seed,
                "mean_accuracy": mean_accuracy,
                "ci95": float(confidence_interval),
            },
        )


if __name__ == "__main__":
    main()

