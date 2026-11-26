"""
CALM-v2 final evaluation on non-ImageNet datasets.

The runner reproduces the ImageNet experiments for one selected dataset:
  1. Few-shot: 5-way, 5-shot, 15 query images per class, 600 episodes.
  2. Continual: sequential class tasks, 5-shot memory per class, reporting
     average accuracy and BWT.

By default this uses the CALM-v2 architecture:
  - EnsembleCLIPModels (ViT-B/32 + ViT-B/16)
  - HierarchicalEpisodicMemory

For an apples-to-apples diagnostic against the v1 accuracy path, add
`--single_backbone`. That keeps hierarchical memory but uses the first
available CLIP variant for both support and query classification.

Supported values for --datasetname:
  cifar10, cifar100, stl10, svhn, fashionmnist

Example:
  python final-others.py --datasetname cifar10
"""

import argparse
import os
import ssl
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, SVHN, STL10
from tqdm import tqdm

from ensemble_calibration import EnsembleCLIPModels
from imagenet_sampler import build_class_index, get_targets
from memory import HierarchicalEpisodicMemory


ssl._create_default_https_context = ssl._create_unverified_context

DATASET_CONFIG = {
    "cifar10": {"classes": 10, "tasks": 5},
    "cifar100": {"classes": 100, "tasks": 10},
    "stl10": {"classes": 10, "tasks": 5},
    "svhn": {"classes": 10, "tasks": 5},
    "fashionmnist": {"classes": 10, "tasks": 5},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run final CALM-v2 evaluation on another dataset.")
    parser.add_argument(
        "--datasetname",
        required=True,
        type=str.lower,
        choices=sorted(DATASET_CONFIG),
        help="Dataset to evaluate: cifar10, cifar100, stl10, svhn, or fashionmnist.",
    )
    parser.add_argument("--data_root", default=None, help="Dataset download/cache directory.")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--num_query", type=int, default=15)
    parser.add_argument("--num_episodes", type=int, default=600)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device. Use cuda to require the NVIDIA GPU.",
    )
    parser.add_argument(
        "--single_backbone",
        action="store_true",
        help="Use only the first CLIP variant for query classification.",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--flat_memory", action="store_true")
    parser.add_argument("--train_compression", action="store_true")
    parser.add_argument("--compression_epochs", type=int, default=50)
    parser.add_argument("--tau_recent", type=int, default=100)
    parser.add_argument("--tau_medium", type=int, default=500)
    parser.add_argument("--age_mode", choices=["per_task", "per_example", "per_class"], default="per_task")
    parser.add_argument("--log_memory", action="store_true")
    parser.add_argument("--max_bytes", type=int, default=None)
    return parser.parse_args()


def get_best_device(requested: str = "auto") -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Install a CUDA-enabled "
                "PyTorch build in this environment and verify with "
                "python -c \"import torch; print(torch.cuda.is_available())\"."
            )
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if torch.cuda.is_available():
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dataset(dataset_name: str, root: str, transform):
    if dataset_name == "cifar10":
        return CIFAR10(root=root, train=False, download=True, transform=transform)
    if dataset_name == "cifar100":
        return CIFAR100(root=root, train=False, download=True, transform=transform)
    if dataset_name == "stl10":
        return STL10(root=root, split="test", download=True, transform=transform)
    if dataset_name == "svhn":
        return SVHN(root=root, split="test", download=True, transform=transform)
    if dataset_name == "fashionmnist":
        return FashionMNIST(root=root, train=False, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_image(dataset, index: int):
    image, label = dataset[index]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return image, int(label)


def sample_episode(dataset, class_to_indices, n_way: int, k_shot: int, num_query: int):
    available_classes = np.array(sorted(class_to_indices))
    if len(available_classes) < n_way:
        raise ValueError(f"Dataset has only {len(available_classes)} usable classes.")

    episode_classes = np.random.choice(available_classes, n_way, replace=False)
    support, query = [], []
    required = k_shot + num_query

    for class_id in episode_classes:
        indices = class_to_indices[int(class_id)]
        if len(indices) < required:
            return None, None
        selected = np.random.choice(indices, required, replace=False)
        support.extend(get_image(dataset, int(index)) for index in selected[:k_shot])
        query.extend(get_image(dataset, int(index)) for index in selected[k_shot:])

    return support, query


def embed_images(images, ensemble_models, device: str, batch_size: int, variant: str):
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            features.append(
                ensemble_models.get_image_features(batch, variant=variant)
            )
    return torch.cat(features, dim=0) if features else torch.empty((0, 512), device=device)


def embed_all_variants(images, ensemble_models, device: str, batch_size: int):
    """Return list of {variant: 1-d tensor} aligned with images (CPU)."""
    variants = list(ensemble_models.models.keys())
    banks = {name: [] for name in variants}
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            all_features = ensemble_models.get_image_features(batch)
            for name, feats in all_features.items():
                banks[name].append(feats.detach().cpu())
            del all_features
    stacked = {
        name: torch.cat(chunks, dim=0) if chunks else torch.empty((0, 512))
        for name, chunks in banks.items()
    }
    return [
        {name: stacked[name][i] for name in variants}
        for i in range(len(images))
    ]


def score_batch(memory, ensemble_models, images, device, temperature, single_backbone, first_variant):
    """Return predicted class-id tensor for a batch of PIL images."""
    if single_backbone:
        prototypes, class_ids = memory.get_prototypes(variant=first_variant)
        if prototypes.nelement() == 0:
            return torch.empty(0, dtype=torch.long, device=device), []
        prototypes = F.normalize(prototypes.to(device), p=2, dim=1)
        features = F.normalize(
            embed_images(images, ensemble_models, device, max(1, len(images)), first_variant),
            p=2,
            dim=1,
        )
        probs = F.softmax((features @ prototypes.T) / temperature, dim=1)
    else:
        banks, class_ids = memory.get_prototypes_by_variant()
        if not class_ids:
            return torch.empty(0, dtype=torch.long, device=device), []
        all_features = ensemble_models.get_image_features(images)
        similarities = ensemble_models.compute_prototype_similarities(
            all_features, banks, class_ids
        )
        probs, _ = ensemble_models.ensemble_confidence(
            similarities, temperature=temperature
        )
        del all_features
    pred_idx = torch.argmax(probs, dim=1)
    predicted = torch.tensor([class_ids[i] for i in pred_idx], device=device)
    return predicted, class_ids


def evaluate_few_shot_episode(
    support,
    query,
    ensemble_models,
    device: str,
    batch_size: int,
    temperature: float,
    single_backbone: bool,
):
    memory = HierarchicalEpisodicMemory(device=device)
    first_variant = list(ensemble_models.models.keys())[0]

    support_images = [item[0] for item in support]
    support_labels = [item[1] for item in support]
    if single_backbone:
        support_features = embed_images(
            support_images, ensemble_models, device, batch_size, first_variant
        )
        for embedding, label in zip(support_features, support_labels):
            memory.add_example(embedding.cpu(), label, importance=1.0, task_id=None)
    else:
        payloads = embed_all_variants(support_images, ensemble_models, device, batch_size)
        for payload, label in zip(payloads, support_labels):
            memory.add_example(payload, label, importance=1.0, task_id=None)

    query_images = [item[0] for item in query]
    query_labels = torch.tensor([item[1] for item in query], device=device)
    correct = 0

    with torch.inference_mode():
        for start in range(0, len(query_images), batch_size):
            batch = query_images[start : start + batch_size]
            predicted, class_ids = score_batch(
                memory,
                ensemble_models,
                batch,
                device,
                temperature,
                single_backbone,
                first_variant,
            )
            if not class_ids:
                continue
            correct += int(
                (predicted == query_labels[start : start + len(batch)]).sum().item()
            )

    return (correct / len(query_labels)) * 100.0 if len(query_labels) else 0.0


def evaluate_indices(
    indices: List[int],
    dataset,
    memory,
    ensemble_models,
    first_variant: str,
    device: str,
    batch_size: int,
    temperature: float,
    single_backbone: bool,
):
    if not indices:
        return 0.0

    correct, total = 0, 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            images, labels = zip(*(get_image(dataset, int(index)) for index in batch_indices))
            predicted, class_ids = score_batch(
                memory,
                ensemble_models,
                list(images),
                device,
                temperature,
                single_backbone,
                first_variant,
            )
            if not class_ids:
                continue
            label_tensor = torch.tensor(labels, device=device)
            correct += int((predicted == label_tensor).sum().item())
            total += len(labels)

    return (correct / total) * 100.0 if total else 0.0


def run_continual(
    dataset,
    ensemble_models,
    device: str,
    batch_size: int,
    k_shot: int,
    num_tasks: int,
    temperature: float,
    single_backbone: bool,
    args=None,
):
    targets = get_targets(dataset)
    class_ids = sorted(int(class_id) for class_id in np.unique(targets))
    class_count = len(class_ids)
    if class_count % num_tasks:
        raise ValueError(f"{class_count} classes cannot be evenly divided into {num_tasks} tasks.")

    classes_per_task = class_count // num_tasks
    class_to_indices = build_class_index(dataset)
    memory = HierarchicalEpisodicMemory(
        device=device,
        tau_recent=getattr(args, "tau_recent", 100),
        tau_medium=getattr(args, "tau_medium", 500),
        flat_memory=getattr(args, "flat_memory", False),
        max_bytes=getattr(args, "max_bytes", None),
    )
    codec_path = getattr(args, "_codec_path", None) if args is not None else None
    if codec_path and os.path.exists(codec_path):
        memory.load_codecs(codec_path)
    results = np.zeros((num_tasks, num_tasks), dtype=np.float64)
    first_variant = list(ensemble_models.models.keys())[0]

    for task_id in range(num_tasks):
        task_classes = class_ids[
            task_id * classes_per_task : (task_id + 1) * classes_per_task
        ]
        print(f"\n--- CALM-v2 continual task {task_id + 1}/{num_tasks} ---")

        for class_id in tqdm(task_classes, desc="Adding support examples"):
            indices = class_to_indices[class_id]
            selected = np.random.choice(indices, min(k_shot, len(indices)), replace=False)
            images = []
            labels = []
            for index in selected:
                image, label = get_image(dataset, int(index))
                images.append(image)
                labels.append(label)
            if single_backbone:
                feats = embed_images(images, ensemble_models, device, batch_size, first_variant)
                for emb, label in zip(feats, labels):
                    memory.add_example(emb.cpu(), label, importance=1.0, task_id=task_id)
                    if args is not None and args.age_mode == "per_example":
                        memory.increment_episode()
            else:
                payloads = embed_all_variants(images, ensemble_models, device, batch_size)
                for payload, label in zip(payloads, labels):
                    memory.add_example(payload, label, importance=1.0, task_id=task_id)
                    if args is not None and args.age_mode == "per_example":
                        memory.increment_episode()
            if args is not None and args.age_mode == "per_class":
                memory.increment_episode()

        if args is None or args.age_mode == "per_task":
            memory.increment_episode()
        if args is not None and args.log_memory:
            print(f"  memory stats: {memory.memory_stats()}")

        for seen_task in range(task_id + 1):
            seen_classes = class_ids[
                seen_task * classes_per_task : (seen_task + 1) * classes_per_task
            ]
            eval_indices = np.concatenate(
                [class_to_indices[class_id] for class_id in seen_classes]
            ).tolist()
            results[task_id, seen_task] = evaluate_indices(
                eval_indices,
                dataset,
                memory,
                ensemble_models,
                first_variant,
                device,
                batch_size,
                temperature,
                single_backbone,
            )
            print(
                f"Task {seen_task + 1} after task {task_id + 1}: "
                f"{results[task_id, seen_task]:.2f}%"
            )

    avg_accuracy = float(np.diag(results).mean())
    last_task_mean = float(results[-1].mean())
    bwt = float(
        np.mean(results[-1, :-1] - np.diag(results)[:-1]) if num_tasks > 1 else 0.0
    )
    return avg_accuracy, bwt, memory.memory_stats(), last_task_mean


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_best_device(args.device)
    data_root = args.data_root or os.path.join(os.path.dirname(__file__), "data")
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224)]
    )

    mode = "single-backbone" if args.single_backbone else "ensemble"
    print(
        f"CALM-v2 final evaluation | dataset={args.datasetname} | "
        f"mode={mode} | device={device}"
    )
    dataset = load_dataset(args.datasetname, data_root, transform)
    ensemble_models = EnsembleCLIPModels(device=device)
    targets = get_targets(dataset)
    class_to_indices = build_class_index(dataset)

    if args.n_way > len(class_to_indices):
        raise ValueError(
            f"--n_way={args.n_way} exceeds the {len(class_to_indices)} available classes."
        )

    print(
        f"\n--- Few-shot: {args.n_way}-way {args.k_shot}-shot, "
        f"{args.num_query} query/class, {args.num_episodes} episodes ---"
    )
    accuracies = []
    for _ in tqdm(range(args.num_episodes), desc="Few-shot episodes"):
        support, query = None, None
        while support is None:
            support, query = sample_episode(
                dataset,
                class_to_indices,
                args.n_way,
                args.k_shot,
                args.num_query,
            )
        accuracies.append(
            evaluate_few_shot_episode(
                support,
                query,
                ensemble_models,
                device,
                args.batch_size,
                args.temperature,
                args.single_backbone,
            )
        )

    mean_accuracy = float(np.mean(accuracies))
    confidence_interval = float(
        1.96 * np.std(accuracies) / np.sqrt(args.num_episodes)
    )
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"95% Confidence Interval: +/- {confidence_interval:.2f}%")

    num_tasks = args.num_tasks or DATASET_CONFIG[args.datasetname]["tasks"]
    print(
        f"\n--- Continual: {num_tasks} tasks, "
        f"{args.k_shot}-shot memory/class ---"
    )
    if args.train_compression:
        print("Training compression networks on a random embedding sample...")
        sample_idx = np.random.choice(len(dataset), size=min(1024, len(dataset)), replace=False)
        first_variant = list(ensemble_models.models.keys())[0]
        sample_images = [get_image(dataset, int(i))[0] for i in sample_idx]
        sample_emb = embed_images(sample_images, ensemble_models, device, args.batch_size, first_variant)
        warmup = HierarchicalEpisodicMemory(
            device=device, tau_recent=args.tau_recent, tau_medium=args.tau_medium
        )
        warmup.train_compression_networks(sample_emb, epochs=args.compression_epochs)
        codec_path = os.path.join(os.path.dirname(__file__), "codecs_last.pt")
        warmup.save_codecs(codec_path)
        args._codec_path = codec_path

    avg_accuracy, bwt, memory_stats, last_task_mean = run_continual(
        dataset,
        ensemble_models,
        device,
        args.batch_size,
        args.k_shot,
        num_tasks,
        args.temperature,
        args.single_backbone,
        args,
    )
    print(f"Average Accuracy (%): {avg_accuracy:.2f}")
    print(f"Last-task mean (%): {last_task_mean:.2f}")
    print(f"Backward Transfer (BWT): {bwt:.2f}")
    if args.output_json:
        calm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if calm_root not in sys.path:
            sys.path.insert(0, calm_root)
        from atomic_json import atomic_write_json

        atomic_write_json(
            args.output_json,
            {
                "variant": (
                    "calm_v2_single" if args.single_backbone else "calm_v2pe"
                ),
                "dataset": args.datasetname,
                "protocol": "few_shot+continual",
                "seed": args.seed,
                "mean_accuracy": mean_accuracy,
                "ci95": confidence_interval,
                "avg_acc": avg_accuracy,
                "last_task_mean": last_task_mean,
                "bwt": bwt,
                "memory": memory_stats,
            },
        )


if __name__ == "__main__":
    main()

