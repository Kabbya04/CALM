"""
iCaRL on the CALM ImageNet 10-task val split.

Default backbone is frozen CLIP ViT-B/32 with a trainable classifier, herding
exemplars (k per class), and distillation. That matches the CALM image pool
and shot budget. Use --backbone resnet18 to train a representation instead.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib.cli import add_common_flags
from lib.clip_backbone import embed_pils, load_clip
from lib.devices import get_device, set_seed
from lib.imagenet_protocol import (
    IMAGENET_TRANSFORM,
    ImageNetContinualManager,
    continual_metrics,
    tensor_to_pil,
)
from lib.results_io import add_output_flag, write_result


def parse_args():
    parser = argparse.ArgumentParser(description="iCaRL on the CALM ImageNet split.")
    add_common_flags(parser)
    add_output_flag(parser)
    parser.add_argument("--backbone", choices=["clip", "resnet18"], default="clip")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--distill", type=float, default=1.0)
    return parser.parse_args()


class LinearHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, features):
        return self.fc(features)


def herding_select(features: torch.Tensor, k: int) -> torch.Tensor:
    if features.shape[0] <= k:
        return torch.arange(features.shape[0])
    mean = features.mean(dim=0)
    selected = []
    running = torch.zeros_like(mean)
    remaining = list(range(features.shape[0]))
    for count in range(1, k + 1):
        best_idx, best_dist = None, None
        for idx in remaining:
            candidate = (running + features[idx]) / count
            dist = torch.norm(mean - candidate)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        running = running + features[best_idx]
    return torch.tensor(selected, dtype=torch.long)


@torch.no_grad()
def collect_task_features(manager, task_id, embed_fn, device, batch_size):
    loader = DataLoader(manager.get_task_datasets(task_id), batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    for images, y in loader:
        feats.append(embed_fn(images, device))
        labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def train_classifier(head, features, labels, old_logits, epochs, lr, device, distill):
    dataset = TensorDataset(features, labels, old_logits)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(epochs):
        for batch_x, batch_y, batch_old in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_old = batch_old.to(device)
            logits = head(batch_x)
            cls = F.cross_entropy(logits, batch_y)
            seen = batch_old.abs().sum(dim=1) > 0
            distill_loss = torch.tensor(0.0, device=device)
            if seen.any() and distill > 0:
                old_idx = batch_old[seen].abs().sum(dim=0) > 0
                if old_idx.any():
                    distill_loss = F.binary_cross_entropy_with_logits(
                        logits[seen][:, old_idx],
                        torch.sigmoid(batch_old[seen][:, old_idx]),
                    )
            loss = cls + distill * distill_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()


@torch.no_grad()
def ncm_predict(query, exemplars):
    class_ids = sorted(exemplars)
    proto = torch.stack([exemplars[cid].mean(dim=0) for cid in class_ids])
    proto = F.normalize(proto, p=2, dim=1)
    query = F.normalize(query, p=2, dim=1)
    sims = query @ proto.T
    pred_idx = torch.argmax(sims, dim=1)
    mapped = torch.tensor([class_ids[i] for i in pred_idx], device=query.device)
    return mapped


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    manager = ImageNetContinualManager(
        root=args.imagenet_root,
        split=args.split,
        transform=IMAGENET_TRANSFORM,
        num_tasks=args.num_tasks,
    )

    if args.backbone == "clip":
        clip_model, processor = load_clip(device)

        def embed_fn(images, dev):
            pils = [tensor_to_pil(image) for image in images]
            return embed_pils(pils, clip_model, processor, dev, args.batch_size)

        feat_dim = 512
    else:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        net = net.to(device)
        net.eval()

        def embed_fn(images, dev):
            images = images.to(dev)
            mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
            with torch.no_grad():
                return net((images - mean) / std)

    head = LinearHead(feat_dim, 1000).to(device)
    exemplars = {}
    results = np.zeros((args.num_tasks, args.num_tasks), dtype=np.float64)

    for task_id in range(args.num_tasks):
        print(f"\n--- iCaRL task {task_id + 1}/{args.num_tasks} ---")
        features, labels = collect_task_features(
            manager, task_id, embed_fn, device, args.batch_size
        )
        if exemplars:
            ex_feats = torch.cat(list(exemplars.values()), dim=0)
            ex_labels = torch.cat(
                [
                    torch.full((emb.shape[0],), cid, dtype=torch.long)
                    for cid, emb in exemplars.items()
                ]
            )
            features = torch.cat([features, ex_feats.to(features.device)], dim=0)
            labels = torch.cat([labels, ex_labels], dim=0)

        with torch.no_grad():
            old_logits = head(features.to(device)).cpu()
            if task_id == 0:
                old_logits = torch.zeros_like(old_logits)

        train_classifier(
            head, features, labels, old_logits, args.epochs, args.lr, device, args.distill
        )

        task_only_x, task_only_y = collect_task_features(
            manager, task_id, embed_fn, device, args.batch_size
        )
        for class_id in manager.get_class_range_for_task(task_id):
            mask = task_only_y == class_id
            if not mask.any():
                continue
            class_feats = task_only_x[mask]
            keep = herding_select(class_feats, args.k_shot)
            exemplars[int(class_id)] = class_feats[keep].cpu()

        for seen in range(task_id + 1):
            query, y = collect_task_features(
                manager, seen, embed_fn, device, args.batch_size
            )
            preds = ncm_predict(query.to(device), {k: v.to(device) for k, v in exemplars.items()})
            acc = float((preds.cpu() == y).float().mean().item() * 100.0)
            results[task_id, seen] = acc
            print(f"  Task {seen + 1} after task {task_id + 1}: {acc:.2f}%")

    metrics = continual_metrics(results)
    print(f"Average Accuracy (%): {metrics['avg_acc']:.2f}")
    print(f"Backward Transfer (BWT): {metrics['bwt']:.2f}")
    write_result(
        {
            "variant": f"icarl_{args.backbone}",
            "dataset": "imagenet",
            "protocol": "continual",
            "seed": args.seed,
            "k_shot": args.k_shot,
            "backbone": args.backbone,
            **metrics,
        },
        args.output_json,
        args.results_dir,
    )


if __name__ == "__main__":
    main()
