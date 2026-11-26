"""
CALM-v1 final evaluation on non-ImageNet datasets.

This reproduces the two ImageNet evaluation protocols:
  1. Few-shot: 5-way, 5-shot, 15 query images per class, 600 episodes.
  2. Continual: sequential class tasks, 5-shot memory per class, with
     average accuracy and BWT.

Supported values for --datasetname:
  cifar10, cifar100, stl10, svhn, fashionmnist

Example:
  python final-others.py --datasetname cifar10
"""

import argparse
import os
import ssl
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, SVHN, STL10
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from memory import EpisodicMemory


ssl._create_default_https_context = ssl._create_unverified_context

DATASET_CONFIG = {
    "cifar10": {"classes": 10, "tasks": 5},
    "cifar100": {"classes": 100, "tasks": 10},
    "stl10": {"classes": 10, "tasks": 5},
    "svhn": {"classes": 10, "tasks": 5},
    "fashionmnist": {"classes": 10, "tasks": 5},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run final CALM-v1 evaluation on another dataset.")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device. Use cuda to require the NVIDIA GPU.",
    )
    parser.add_argument("--output_json", default=None)
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


def get_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    raise AttributeError("Dataset does not expose targets or labels.")


def get_image(dataset, index: int):
    image, label = dataset[index]
    # FashionMNIST returns grayscale; CLIP expects RGB.
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return image, int(label)


def sample_episode(
    dataset,
    class_to_indices: dict[int, np.ndarray],
    n_way: int,
    k_shot: int,
    num_query: int,
):
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


def embed_images(images, model, processor, device: str, batch_size: int):
    features = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            features.append(model.get_image_features(**inputs))
    return torch.cat(features, dim=0) if features else torch.empty((0, 512), device=device)


def evaluate_few_shot_episode(support, query, model, processor, device, batch_size):
    memory = EpisodicMemory()
    support_images = [item[0] for item in support]
    support_labels = [item[1] for item in support]
    support_features = embed_images(support_images, model, processor, device, batch_size)

    for embedding, label in zip(support_features, support_labels):
        memory.add_example(embedding, label)

    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0:
        return 0.0
    prototypes = F.normalize(prototypes.to(device), p=2, dim=1)

    query_images = [item[0] for item in query]
    query_labels = torch.tensor([item[1] for item in query], device=device)
    query_features = F.normalize(
        embed_images(query_images, model, processor, device, batch_size), p=2, dim=1
    )
    predictions = torch.argmax(query_features @ prototypes.T, dim=1)
    predicted_labels = torch.tensor([class_ids[index] for index in predictions], device=device)
    return float((predicted_labels == query_labels).float().mean().item() * 100.0)


def evaluate_indices(indices, dataset, memory, model, processor, device, batch_size):
    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0 or not indices:
        return 0.0
    prototypes = F.normalize(prototypes.to(device), p=2, dim=1)

    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            images, labels = zip(*(get_image(dataset, int(index)) for index in batch_indices))
            features = F.normalize(
                embed_images(list(images), model, processor, device, batch_size), p=2, dim=1
            )
            predictions = torch.argmax(features @ prototypes.T, dim=1)
            predicted_labels = torch.tensor([class_ids[index] for index in predictions], device=device)
            label_tensor = torch.tensor(labels, device=device)
            correct += int((predicted_labels == label_tensor).sum().item())
            total += len(labels)
    return (correct / total) * 100.0 if total else 0.0


def run_continual(dataset, model, processor, device, batch_size, k_shot, num_tasks, max_bytes=None):
    targets = get_targets(dataset)
    class_count = int(targets.max()) + 1
    classes_per_task = class_count // num_tasks
    if class_count % num_tasks:
        raise ValueError(f"{class_count} classes cannot be evenly divided into {num_tasks} tasks.")

    class_to_indices = {
        class_id: np.flatnonzero(targets == class_id) for class_id in range(class_count)
    }
    results = np.zeros((num_tasks, num_tasks), dtype=np.float64)
    memory = EpisodicMemory(max_bytes=max_bytes)

    for task_id in range(num_tasks):
        task_classes = range(task_id * classes_per_task, (task_id + 1) * classes_per_task)
        print(f"\n--- CALM-v1 continual task {task_id + 1}/{num_tasks} ---")

        for class_id in tqdm(task_classes, desc="Adding support examples"):
            indices = class_to_indices[class_id]
            selected = np.random.choice(indices, min(k_shot, len(indices)), replace=False)
            for index in selected:
                image, label = get_image(dataset, int(index))
                feature = embed_images([image], model, processor, device, 1).squeeze(0)
                memory.add_example(feature, label)

        for seen_task in range(task_id + 1):
            start_class = seen_task * classes_per_task
            end_class = (seen_task + 1) * classes_per_task
            eval_indices = np.concatenate(
                [class_to_indices[class_id] for class_id in range(start_class, end_class)]
            ).tolist()
            results[task_id, seen_task] = evaluate_indices(
                eval_indices, dataset, memory, model, processor, device, batch_size
            )
            print(
                f"Task {seen_task + 1} after task {task_id + 1}: "
                f"{results[task_id, seen_task]:.2f}%"
            )

    avg_accuracy = float(np.diag(results).mean())
    bwt = float(
        np.mean(results[-1, :-1] - np.diag(results)[:-1]) if num_tasks > 1 else 0.0
    )
    return avg_accuracy, bwt


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_best_device(args.device)
    data_root = args.data_root or os.path.join(os.path.dirname(__file__), "data")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

    print(f"CALM-v1 final evaluation | dataset={args.datasetname} | device={device}")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    dataset = load_dataset(args.datasetname, data_root, transform)
    targets = get_targets(dataset)
    class_to_indices = {
        class_id: np.flatnonzero(targets == class_id) for class_id in np.unique(targets)
    }

    print(
        f"\n--- Few-shot: {args.n_way}-way {args.k_shot}-shot, "
        f"{args.num_query} query/class, {args.num_episodes} episodes ---"
    )
    accuracies = []
    for _ in tqdm(range(args.num_episodes), desc="Few-shot episodes"):
        support, query = None, None
        while support is None:
            support, query = sample_episode(
                dataset, class_to_indices, args.n_way, args.k_shot, args.num_query
            )
        accuracies.append(
            evaluate_few_shot_episode(
                support, query, model, processor, device, args.batch_size
            )
        )
    mean = float(np.mean(accuracies))
    ci = float(1.96 * np.std(accuracies) / np.sqrt(args.num_episodes))
    print(f"Mean Accuracy: {mean:.2f}%")
    print(f"95% Confidence Interval: +/- {ci:.2f}%")

    num_tasks = args.num_tasks or DATASET_CONFIG[args.datasetname]["tasks"]
    print(
        f"\n--- Continual: {num_tasks} tasks, {args.k_shot}-shot memory/class ---"
    )
    avg_accuracy, bwt = run_continual(
        dataset, model, processor, device, args.batch_size, args.k_shot, num_tasks, args.max_bytes
    )
    print(f"Average Accuracy (%): {avg_accuracy:.2f}")
    print(f"Backward Transfer (BWT): {bwt:.2f}")
    if args.output_json:
        _calm_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        if _calm_root not in sys.path:
            sys.path.insert(0, _calm_root)
        from atomic_json import atomic_write_json

        atomic_write_json(
            args.output_json,
            {
                "variant": "calm_v1",
                "dataset": args.datasetname,
                "protocol": "few_shot+continual",
                "seed": args.seed,
                "mean_accuracy": mean,
                "ci95": ci,
                "avg_acc": avg_accuracy,
                "bwt": bwt,
            },
        )


if __name__ == "__main__":
    main()

