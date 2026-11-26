"""Tip-Adapter (training-free CLIP cache) on the CALM ImageNet split."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib.cli import add_common_flags
from lib.clip_backbone import embed_pils, embed_text, load_clip
from lib.devices import get_device, set_seed
from lib.imagenet_protocol import (
    IMAGENET_TRANSFORM,
    ImageNetContinualManager,
    continual_metrics,
    sample_supports,
    tensor_to_pil,
)
from lib.results_io import add_output_flag, write_result


def parse_args():
    parser = argparse.ArgumentParser(description="Tip-Adapter on the CALM ImageNet split.")
    add_common_flags(parser)
    add_output_flag(parser)
    parser.add_argument("--beta", type=float, default=5.5, help="Affinity sharpness.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Cache residual weight.")
    parser.add_argument("--prompt", default="a photo of a {}")
    return parser.parse_args()


class TipCache:
    def __init__(self, text_features, alpha, beta):
        self.text_features = text_features
        self.alpha = alpha
        self.beta = beta
        self.keys = []
        self.onehot = []
        self.num_classes = text_features.shape[0]

    def add(self, embeddings: torch.Tensor, labels):
        labels = torch.as_tensor(labels, device=embeddings.device, dtype=torch.long)
        onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        self.keys.append(F.normalize(embeddings, p=2, dim=1))
        self.onehot.append(onehot)

    def predict(self, query: torch.Tensor) -> torch.Tensor:
        clip_logits = 100.0 * query @ self.text_features.T
        if not self.keys:
            return clip_logits
        keys = torch.cat(self.keys, dim=0)
        values = torch.cat(self.onehot, dim=0)
        affinity = torch.exp(-self.beta * (1.0 - query @ keys.T))
        cache_logits = affinity @ values
        return clip_logits + self.alpha * cache_logits


@torch.no_grad()
def evaluate_task(manager, task_id, cache, model, processor, device, batch_size):
    loader = DataLoader(manager.get_task_datasets(task_id), batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for images, labels in loader:
        labels = labels.to(device)
        pils = [tensor_to_pil(image) for image in images]
        query = embed_pils(pils, model, processor, device, batch_size)
        preds = torch.argmax(cache.predict(query), dim=1)
        correct += int((preds == labels).sum().item())
        total += len(labels)
    return (correct / total) * 100.0 if total else 0.0


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    model, processor = load_clip(device)
    manager = ImageNetContinualManager(
        root=args.imagenet_root,
        split=args.split,
        transform=IMAGENET_TRANSFORM,
        num_tasks=args.num_tasks,
    )
    prompts = [args.prompt.format(manager.class_name(class_id)) for class_id in range(1000)]
    text_features = embed_text(prompts, model, processor, device)
    cache = TipCache(text_features, alpha=args.alpha, beta=args.beta)

    results = np.zeros((args.num_tasks, args.num_tasks), dtype=np.float64)
    for task_id in range(args.num_tasks):
        class_range = manager.get_class_range_for_task(task_id)
        print(f"\n--- Tip-Adapter task {task_id + 1}/{args.num_tasks} ---")
        for class_id in tqdm(class_range, desc="Building cache"):
            support_idx = sample_supports(manager, class_id, args.k_shot)
            if not support_idx:
                continue
            pils, labels = [], []
            for idx in support_idx:
                image, label = manager.dataset[idx]
                pils.append(tensor_to_pil(image))
                labels.append(int(label))
            embeddings = embed_pils(pils, model, processor, device, args.batch_size)
            cache.add(embeddings, labels)

        for seen in range(task_id + 1):
            acc = evaluate_task(
                manager, seen, cache, model, processor, device, args.batch_size
            )
            results[task_id, seen] = acc
            print(f"  Task {seen + 1} after task {task_id + 1}: {acc:.2f}%")

    metrics = continual_metrics(results)
    print(f"Average Accuracy (%): {metrics['avg_acc']:.2f}")
    print(f"Last-task mean (%): {metrics['last_task_mean']:.2f}")
    print(f"Backward Transfer (BWT): {metrics['bwt']:.2f}")
    write_result(
        {
            "variant": "tip_adapter",
            "dataset": "imagenet",
            "protocol": "continual",
            "seed": args.seed,
            "k_shot": args.k_shot,
            "alpha": args.alpha,
            "beta": args.beta,
            **metrics,
        },
        args.output_json,
        args.results_dir,
    )


if __name__ == "__main__":
    main()
