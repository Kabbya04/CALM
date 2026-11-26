"""CLIP zero-shot on the same ImageNet 10-task val split used by CALM."""

import argparse
import os
import sys

import numpy as np
import torch
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
    tensor_to_pil,
)
from lib.results_io import add_output_flag, write_result


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP zero-shot on the CALM ImageNet split.")
    add_common_flags(parser)
    add_output_flag(parser)
    parser.add_argument("--prompt", default="a photo of a {}")
    return parser.parse_args()


@torch.no_grad()
def evaluate_task(manager, task_id, text_features, model, processor, device, batch_size):
    loader = DataLoader(manager.get_task_datasets(task_id), batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for images, labels in loader:
        labels = labels.to(device)
        pils = [tensor_to_pil(image) for image in images]
        image_features = embed_pils(pils, model, processor, device, batch_size)
        logits = image_features @ text_features.T
        preds = torch.argmax(logits, dim=1)
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
    print(f"Encoding {len(prompts)} class prompts...")
    text_features = embed_text(prompts, model, processor, device)

    results = np.zeros((args.num_tasks, args.num_tasks), dtype=np.float64)
    for task_id in range(args.num_tasks):
        print(f"\n--- Zero-shot task {task_id + 1}/{args.num_tasks} ---")
        for seen in range(task_id + 1):
            acc = evaluate_task(
                manager, seen, text_features, model, processor, device, args.batch_size
            )
            results[task_id, seen] = acc
            print(f"  Task {seen + 1} after task {task_id + 1}: {acc:.2f}%")

    metrics = continual_metrics(results)
    print(f"Average Accuracy (%): {metrics['avg_acc']:.2f}")
    print(f"Last-task mean (%): {metrics['last_task_mean']:.2f}")
    print(f"Backward Transfer (BWT): {metrics['bwt']:.2f}")
    write_result(
        {
            "variant": "clip_zeroshot",
            "dataset": "imagenet",
            "protocol": "continual",
            "seed": args.seed,
            "num_tasks": args.num_tasks,
            **metrics,
        },
        args.output_json,
        args.results_dir,
    )


if __name__ == "__main__":
    main()
