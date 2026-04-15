"""
ImageNet-aligned few-shot evaluation for CALM-v2 (single-backbone variant).

This script mirrors CALM-v1's ImageNet 5-way 5-shot protocol, but uses:
  - HierarchicalEpisodicMemory (CALM-v2 episodic memory)
  - ONLY the first CLIP variant from EnsembleCLIPModels (typically ViT-B/32)
    for both memory and query features when computing accuracy.

The ensemble can still be used elsewhere for confidence calibration, but
classification decisions here are based on a single, consistent embedding
space to make results directly comparable to CALM-v1.
"""

import argparse
import os
import ssl
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from memory import HierarchicalEpisodicMemory  # type: ignore
from imagenet_sampler import ImageNetSampler

# Ensure we can import modules from CALM-v2/episodic-memory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALM_V2_ROOT = os.path.dirname(THIS_DIR)
EPISODIC_DIR = os.path.join(CALM_V2_ROOT, "episodic-memory")
if EPISODIC_DIR not in sys.path:
    sys.path.insert(0, EPISODIC_DIR)

from ensemble_calibration import EnsembleCLIPModels  # type: ignore


# --- SSL fix copied from CALM-v1 script to avoid cert issues when transformers download weights ---
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CALM-v2 (single-backbone) on ImageNet (few-shot episodic)."
    )
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
        help="Temperature for softmax on cosine similarities.",
    )
    return parser.parse_args()


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_memory_from_support(
    support_set: List[Tuple[object, int]],
    first_variant: str,
    ensemble_models: EnsembleCLIPModels,
    memory: HierarchicalEpisodicMemory,
    device: str,
    batch_size: int,
):
    """
    Populate hierarchical episodic memory from the support set using a single backbone.
    """
    support_images = [item[0] for item in support_set]
    support_labels = [item[1] for item in support_set]

    with torch.no_grad():
        for i in range(0, len(support_images), batch_size):
            batch = support_images[i : i + batch_size]
            features = ensemble_models.get_image_features(batch, variant=first_variant)
            for j, embedding in enumerate(features):
                memory.add_example(embedding, support_labels[i + j], importance=1.0, task_id=None)


def evaluate_episode(
    support_set: List[Tuple[object, int]],
    query_set: List[Tuple[object, int]],
    ensemble_models: EnsembleCLIPModels,
    device: str,
    batch_size: int,
    temperature: float,
) -> float:
    """
    Run a single N-way K-shot ImageNet episode with a single-backbone CALM-v2 core.

    - Builds hierarchical episodic memory from support examples (first_variant only).
    - Uses the same backbone to embed query images and perform cosine-prototype classification.
    - Returns classification accuracy (%) on the query set.
    """
    memory = HierarchicalEpisodicMemory(device=device)

    # Choose the first available variant as the backbone (typically ViT-B/32)
    first_variant = list(ensemble_models.models.keys())[0]

    # 1) Build memory from support
    build_memory_from_support(support_set, first_variant, ensemble_models, memory, device, batch_size)

    # 2) Prepare prototypes
    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0:
        return 0.0

    prototypes = prototypes.to(device)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    # 3) Evaluate on query set using ONLY the same backbone
    query_images = [item[0] for item in query_set]
    query_labels = torch.tensor([item[1] for item in query_set], device=device)

    correct_predictions = 0

    with torch.no_grad():
        for i in range(0, len(query_images), batch_size):
            batch = query_images[i : i + batch_size]

            # Single-backbone features
            features = ensemble_models.get_image_features(batch, variant=first_variant)
            features = F.normalize(features.to(device), p=2, dim=1)

            # Cosine similarities and softmax with optional temperature
            logits = features @ prototypes.T
            logits = logits / temperature
            probs = F.softmax(logits, dim=1)

            pred_indices = torch.argmax(probs, dim=1)
            predicted_class_ids = torch.tensor([class_ids[p] for p in pred_indices], device=device)

            batch_labels = query_labels[i : i + len(batch)]
            correct_predictions += (predicted_class_ids == batch_labels).sum().item()

    return (correct_predictions / len(query_labels)) * 100.0


def main():
    args = parse_args()
    device = get_best_device()
    print(f"Using device: {device}")

    # 1) Initialize ensemble CLIP models (we will use only the first variant for accuracy)
    print("Initializing CLIP ensemble (single-backbone accuracy path)...")
    ensemble_models = EnsembleCLIPModels(device=device)

    # 2) Load ImageNet sampler using the same transform as CALM-v1
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    sampler = ImageNetSampler(root_path=args.imagenet_root, split="val", transform=transform)

    # 3) Run episodic evaluation
    print(f"\nStarting CALM-v2 (single-backbone) ImageNet evaluation for {args.num_episodes} episodes...")
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
        )
        all_accuracies.append(acc)

        if (episode_idx + 1) % 50 == 0 or episode_idx == 0:
            print(f"Episode {episode_idx + 1}/{args.num_episodes}: accuracy = {acc:.2f}%")

    # 4) Report final results with 95% confidence interval
    mean_accuracy = float(np.mean(all_accuracies))
    std_dev = float(np.std(all_accuracies))
    confidence_interval = 1.96 * std_dev / np.sqrt(args.num_episodes)

    print("\n--- Final ImageNet Few-Shot Evaluation Results (CALM-v2, single-backbone) ---")
    print(f"Configuration: {args.n_way}-way, {args.k_shot}-shot with {args.num_query} query images per class.")
    print(f"Evaluated on {args.num_episodes} episodes.")
    print("-" * 60)
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"95% Confidence Interval: +/- {confidence_interval:.2f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()

