"""
ImageNet continual learning evaluation for CALM-v2.

This mirrors CALM-v1's `run_imagenet_continual.py` protocol but uses:
  - HierarchicalEpisodicMemory (CALM-v2 episodic-memory)
  - EnsembleCLIPModels (CALM-v2 ensemble CLIP wrapper)

It reports:
  - Average accuracy across tasks (diag of results matrix)
  - Backward transfer (BWT) measuring forgetting

The ImageNet dataset/root should be the same preprocessed tree used for CALM-v1.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm

# Ensure we can import episodic-memory components when running this script directly.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALM_V2_ROOT = os.path.dirname(THIS_DIR)
EPISODIC_DIR = os.path.join(CALM_V2_ROOT, "episodic-memory")

if EPISODIC_DIR not in sys.path:
    sys.path.insert(0, EPISODIC_DIR)

from memory import HierarchicalEpisodicMemory  # type: ignore
from ensemble_calibration import EnsembleCLIPModels  # type: ignore


class ImageNetContinualManager:
    """
    Handles the ImageNet dataset for continual learning by splitting
    the 1000 classes into a sequence of tasks, exactly like CALM-v1.
    """

    def __init__(self, root: str, split: str, transform, num_tasks: int):
        print(f"Loading ImageNet '{split}' split for continual learning...")
        self.dataset = ImageNet(root=root, split=split, transform=transform)
        self.num_tasks = num_tasks

        if 1000 % num_tasks != 0:
            raise ValueError(f"1000 classes must be divisible by num_tasks ({num_tasks}).")

        self.classes_per_task = 1000 // num_tasks

        print("Creating robust class index by verifying files...")
        self.class_index = defaultdict(list)
        for master_idx, (path, class_idx) in enumerate(tqdm(self.dataset.samples, desc="Verifying samples")):
            if os.path.exists(path):
                self.class_index[class_idx].append(master_idx)

        total_samples = sum(len(v) for v in self.class_index.values())
        print(f"Index complete. Found {len(self.class_index)} classes and {total_samples} actual samples.")

    def get_class_range_for_task(self, task_id: int):
        start_class = task_id * self.classes_per_task
        end_class = (task_id + 1) * self.classes_per_task
        return range(start_class, end_class)

    def get_task_datasets(self, task_id: int) -> Subset:
        """
        Returns a Subset of the data corresponding to a specific task.
        """
        task_indices = []
        class_range = self.get_class_range_for_task(task_id)
        for class_id in class_range:
            task_indices.extend(self.class_index[class_id])

        return Subset(self.dataset, task_indices)


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


def evaluate_on_memory_ensemble(
    memory: HierarchicalEpisodicMemory,
    ensemble_models: EnsembleCLIPModels,
    data_manager: ImageNetContinualManager,
    task_id: int,
    device: str,
    batch_size: int,
    temperature: float,
    single_backbone: bool = False,
) -> float:
    """
    Evaluate accuracy on a specific task using:
      - prototypes from hierarchical memory
      - ensemble CLIP similarity for classification
    """
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

    test_loader = DataLoader(data_manager.get_task_datasets(task_id), batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            labels = labels.to(device)
            pil_images = [transforms.ToPILImage()(img) for img in images]

            if single_backbone:
                features = F.normalize(
                    ensemble_models.get_image_features(pil_images, variant=first_variant),
                    p=2,
                    dim=1,
                )
                ensemble_probs = F.softmax((features @ prototypes.T) / temperature, dim=1)
            else:
                all_features = ensemble_models.get_image_features(pil_images)
                similarities = ensemble_models.compute_prototype_similarities(
                    all_features, banks, class_ids
                )
                ensemble_probs, _ = ensemble_models.ensemble_confidence(
                    similarities, temperature=temperature
                )
                del all_features

            pred_indices = torch.argmax(ensemble_probs, dim=1)
            predicted_class_ids = torch.tensor([class_ids[i] for i in pred_indices], device=device)

            correct += (predicted_class_ids == labels).sum().item()
            total += len(labels)

    return (correct / total) * 100.0 if total > 0 else 0.0


def run_continual_task_sequence(
    ensemble_models: EnsembleCLIPModels,
    data_manager: ImageNetContinualManager,
    k_shot: int,
    device: str,
    batch_size: int,
    temperature: float,
    args=None,
):
    """
    Runs a full continual learning sequence over ImageNet and computes:
      - Average accuracy over tasks (diag of results matrix)
      - Backward transfer (BWT) as in CALM-v1
    """
    memory = HierarchicalEpisodicMemory(
        device=device,
        tau_recent=getattr(args, "tau_recent", 100),
        tau_medium=getattr(args, "tau_medium", 500),
        flat_memory=getattr(args, "flat_memory", False),
        max_bytes=getattr(args, "max_bytes", None),
    )
    if args is not None and getattr(args, "codec_path", None) and os.path.exists(args.codec_path):
        memory.load_codecs(args.codec_path)
    num_tasks = data_manager.num_tasks
    results_matrix = np.zeros((num_tasks, num_tasks))

    first_variant = list(ensemble_models.models.keys())[0]

    for task_id in range(num_tasks):
        class_range = data_manager.get_class_range_for_task(task_id)
        print(f"\n--- Learning Task {task_id + 1}/{num_tasks} (Classes {class_range.start}-{class_range.stop - 1}) ---")

        train_subset = data_manager.get_task_datasets(task_id)

        for class_id in tqdm(class_range, desc="Building memory for task"):
            # Collect indices in this subset that belong to this class
            class_subset_indices = [
                i for i in train_subset.indices if train_subset.dataset.samples[i][1] == class_id
            ]

            if not class_subset_indices:
                print(f"Warning: No samples found for class {class_id} in this split.")
                continue

            k_sample = min(k_shot, len(class_subset_indices))
            if k_sample < k_shot:
                print(f"Warning: Class {class_id} only has {k_sample} samples in this split.")

            support_indices = np.random.choice(class_subset_indices, k_sample, replace=False)

            use_single = bool(args.single_backbone) if args is not None else False
            for idx in support_indices:
                image_tensor, label = data_manager.dataset[idx]
                image_pil = transforms.ToPILImage()(image_tensor)

                with torch.inference_mode():
                    if use_single:
                        embedding = ensemble_models.get_image_features(
                            [image_pil], variant=first_variant
                        ).squeeze(0).cpu()
                        memory.add_example(embedding, int(label), importance=1.0, task_id=task_id)
                    else:
                        feats = ensemble_models.get_image_features([image_pil])
                        payload = {
                            name: tensor.squeeze(0).cpu() for name, tensor in feats.items()
                        }
                        memory.add_example(payload, int(label), importance=1.0, task_id=task_id)
                        del feats
                if args is not None and args.age_mode == "per_example":
                    memory.increment_episode()
            if args is not None and args.age_mode == "per_class":
                memory.increment_episode()

        if args is None or args.age_mode == "per_task":
            memory.increment_episode()
        if args is not None and args.log_memory:
            print(f"  memory stats: {memory.memory_stats()}")

        # Evaluate on all seen tasks so far
        for j in range(task_id + 1):
            acc = evaluate_on_memory_ensemble(
                memory,
                ensemble_models,
                data_manager,
                task_id=j,
                device=device,
                batch_size=batch_size,
                temperature=temperature,
                single_backbone=bool(args.single_backbone) if args is not None else False,
            )
            print(f"  Accuracy on Task {j + 1} after learning Task {task_id + 1}: {acc:.2f}%")
            results_matrix[task_id, j] = acc

    avg_accuracy = float(results_matrix.diagonal().mean())
    last_task_mean = float(results_matrix[-1].mean())
    bwt = 0.0
    if num_tasks > 1:
        for j in range(num_tasks - 1):
            bwt += results_matrix[num_tasks - 1, j] - results_matrix[j, j]
        bwt /= (num_tasks - 1)

    return {
        "avg_acc": avg_accuracy,
        "last_task_mean": last_task_mean,
        "bwt": float(bwt),
        "memory": memory.memory_stats(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="CALM-v2 Continual Learning on ImageNet")
    parser.add_argument(
        "--imagenet_root",
        type=str,
        required=True,
        help="Root directory of your preprocessed ImageNet (same as CALM-v1).",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10,
        help="Number of sequential tasks to create from 1000 classes (must divide 1000).",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=5,
        help="Number of support examples per class.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for ensemble softmax.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--single_backbone", action="store_true")
    parser.add_argument("--flat_memory", action="store_true")
    parser.add_argument("--train_compression", action="store_true")
    parser.add_argument("--compression_epochs", type=int, default=50)
    parser.add_argument("--codec_path", default=None)
    parser.add_argument("--tau_recent", type=int, default=100)
    parser.add_argument("--tau_medium", type=int, default=500)
    parser.add_argument("--age_mode", choices=["per_task", "per_example", "per_class"], default="per_task")
    parser.add_argument("--log_memory", action="store_true")
    parser.add_argument("--max_bytes", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_best_device(args.device)
    print(f"Using device: {device}")

    print(
        f"\n--- WARNING: This is a long-running experiment! "
        f"{args.num_tasks} tasks over 1000 ImageNet classes with {args.k_shot}-shot memory. ---\n"
    )

    # Initialize ensemble models
    print("\n--- Initializing Ensemble CLIP Models (CALM-v2) ---")
    ensemble_models = EnsembleCLIPModels(device=device)

    # Transform: resize + center crop + ToTensor (for DataLoader)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    data_manager = ImageNetContinualManager(
        root=args.imagenet_root,
        split="val",
        transform=transform,
        num_tasks=args.num_tasks,
    )

    if args.train_compression:
        print("\n--- Training compression networks on a random ImageNet val sample ---")
        first_variant = list(ensemble_models.models.keys())[0]
        sample_idx = np.random.choice(len(data_manager.dataset), size=min(2048, len(data_manager.dataset)), replace=False)
        sample_emb = []
        warmup = HierarchicalEpisodicMemory(device=device, tau_recent=args.tau_recent, tau_medium=args.tau_medium)
        for idx in sample_idx:
            image_tensor, _ = data_manager.dataset[int(idx)]
            image_pil = transforms.ToPILImage()(image_tensor)
            with torch.no_grad():
                sample_emb.append(
                    ensemble_models.get_image_features([image_pil], variant=first_variant).squeeze(0).cpu()
                )
        warmup.train_compression_networks(torch.stack(sample_emb), epochs=args.compression_epochs)
        if args.codec_path:
            warmup.save_codecs(args.codec_path)
            print(f"Saved codecs to {args.codec_path}")
        else:
            args.codec_path = os.path.join(THIS_DIR, "codecs_last.pt")
            warmup.save_codecs(args.codec_path)

    print(
        f"\n--- Running {args.k_shot}-shot continual prototypical sequence on ImageNet "
        f"with {args.num_tasks} tasks ---"
    )
    results = run_continual_task_sequence(
        ensemble_models=ensemble_models,
        data_manager=data_manager,
        k_shot=args.k_shot,
        device=device,
        batch_size=args.batch_size,
        temperature=args.temperature,
        args=args,
    )

    print(f"\n{'=' * 60}")
    print("    Final Continual Learning Results on ImageNet (CALM-v2)")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} | {'Value':<20}")
    print("-" * 55)
    print(f"{'K-Shot':<30} | {args.k_shot}")
    print(f"{'Number of Tasks':<30} | {args.num_tasks}")
    print("-" * 55)
    print(f"{'Average Accuracy (%)':<30} | {results['avg_acc']:.2f}")
    print(f"{'Last-task mean (%)':<30} | {results['last_task_mean']:.2f}")
    print(f"{'Backward Transfer (BWT)':<30} | {results['bwt']:.2f}")
    print("-" * 55)
    print(
        "\nBWT Definition: A value close to 0 indicates low forgetting. "
        "A large negative value indicates severe catastrophic forgetting."
    )
    if args.log_memory:
        print(f"Final memory: {results.get('memory')}")
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
                "protocol": "continual",
                "seed": args.seed,
                "age_mode": args.age_mode,
                "flat_memory": args.flat_memory,
                "avg_acc": results["avg_acc"],
                "last_task_mean": results["last_task_mean"],
                "bwt": results["bwt"],
                "memory": results.get("memory"),
            },
        )


if __name__ == "__main__":
    main()

