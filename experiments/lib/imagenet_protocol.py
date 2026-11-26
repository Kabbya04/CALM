"""ImageNet split used by CALM-v1/v2. Do not rearrange the dataset tree."""

import os
from collections import defaultdict

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm


IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

# PIL-only transform for CLIP processors.
IMAGENET_PIL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
)


class ImageNetContinualManager:
    """1000 ImageNet val classes split into equal sequential tasks."""

    def __init__(self, root: str, split: str, transform, num_tasks: int):
        print(f"Loading ImageNet '{split}' split for continual learning...")
        self.dataset = ImageNet(root=root, split=split, transform=transform)
        self.num_tasks = num_tasks
        if 1000 % num_tasks != 0:
            raise ValueError(f"1000 classes must be divisible by num_tasks ({num_tasks}).")
        self.classes_per_task = 1000 // num_tasks

        print("Creating robust class index by verifying files...")
        self.class_index = defaultdict(list)
        for master_idx, (path, class_idx) in enumerate(
            tqdm(self.dataset.samples, desc="Verifying samples")
        ):
            if os.path.exists(path):
                self.class_index[class_idx].append(master_idx)

        total_samples = sum(len(items) for items in self.class_index.values())
        print(
            f"Index complete. Found {len(self.class_index)} classes "
            f"and {total_samples} actual samples."
        )

    def get_class_range_for_task(self, task_id: int):
        start = task_id * self.classes_per_task
        return range(start, start + self.classes_per_task)

    def get_task_datasets(self, task_id: int) -> Subset:
        indices = []
        for class_id in self.get_class_range_for_task(task_id):
            indices.extend(self.class_index[class_id])
        return Subset(self.dataset, indices)

    def class_wnid(self, class_id: int) -> str:
        return self.dataset.wnids[class_id]

    def class_name(self, class_id: int) -> str:
        names = self.dataset.classes[class_id]
        if isinstance(names, (list, tuple)):
            return names[0].replace("_", " ")
        return str(names).replace("_", " ")


def tensor_to_pil(image_tensor):
    return transforms.ToPILImage()(image_tensor)


def sample_supports(manager: ImageNetContinualManager, class_id: int, k_shot: int):
    indices = manager.class_index[class_id]
    if not indices:
        return []
    k_sample = min(k_shot, len(indices))
    chosen = np.random.choice(indices, k_sample, replace=False)
    return [int(idx) for idx in chosen]


def continual_metrics(results_matrix: np.ndarray) -> dict:
    num_tasks = results_matrix.shape[0]
    avg_acc = float(results_matrix.diagonal().mean())
    last_task_mean = float(results_matrix[-1].mean()) if num_tasks else 0.0
    bwt = 0.0
    if num_tasks > 1:
        bwt = float(np.mean(results_matrix[-1, :-1] - np.diag(results_matrix)[:-1]))
    return {
        "avg_acc": avg_acc,
        "last_task_mean": last_task_mean,
        "bwt": bwt,
    }
