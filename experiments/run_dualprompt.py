"""
DualPrompt on the CALM ImageNet 10-task val split.

Frozen ViT-B/16 (ImageNet-21k if timm can download it), trainable G-prompts
and E-prompts, query-key task selection. Evaluation uses the same val images
and task order as CALM.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib.cli import add_common_flags
from lib.devices import get_device, set_seed
from lib.imagenet_protocol import IMAGENET_TRANSFORM, ImageNetContinualManager, continual_metrics
from lib.results_io import add_output_flag, write_result


def parse_args():
    parser = argparse.ArgumentParser(description="DualPrompt on the CALM ImageNet split.")
    add_common_flags(parser)
    add_output_flag(parser)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--g_length", type=int, default=5)
    parser.add_argument("--e_length", type=int, default=20)
    parser.add_argument("--g_layers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--e_layers", type=int, nargs="+", default=[2, 3, 4])
    return parser.parse_args()


def load_vit(device: str):
    try:
        import timm
    except ImportError as exc:
        raise SystemExit("DualPrompt needs timm. Install with: pip install timm") from exc

    names = [
        "vit_base_patch16_224.augreg_in21k",
        "vit_base_patch16_224",
    ]
    last_error = None
    for name in names:
        try:
            model = timm.create_model(name, pretrained=True)
            print(f"Loaded DualPrompt backbone: {name}")
            return model.to(device)
        except Exception as error:
            last_error = error
            print(f"Could not load {name}: {error}")
    raise RuntimeError(f"Could not load a ViT backbone: {last_error}")


class DualPromptNet(nn.Module):
    """Frozen ViT with shared G-prompts, task E-prompts, and query-key routing."""

    def __init__(self, vit, num_tasks, g_layers, e_layers, g_length, e_length, num_classes=1000):
        super().__init__()
        self.vit = vit
        for param in self.vit.parameters():
            param.requires_grad = False
        self.embed_dim = vit.embed_dim
        self.g_length = int(g_length) * max(1, len(g_layers))
        self.e_length = int(e_length) * max(1, len(e_layers))
        self.num_tasks = num_tasks
        self.g_prompt = nn.Parameter(torch.randn(self.g_length, self.embed_dim) * 0.02)
        self.e_prompt = nn.Parameter(torch.randn(num_tasks, self.e_length, self.embed_dim) * 0.02)
        self.prompt_pos = nn.Parameter(torch.randn(self.g_length + self.e_length, self.embed_dim) * 0.02)
        self.task_keys = nn.Parameter(torch.randn(num_tasks, self.embed_dim) * 0.02)
        self.head = nn.Linear(self.embed_dim, num_classes)
        self._current_task = 0
        self._use_query = False

    def _query_task(self, tokens):
        scores = F.normalize(tokens, dim=-1) @ F.normalize(self.task_keys, dim=-1).T
        return int(scores.mean(dim=0).argmax().item())

    def set_task(self, task_id: int, use_query: bool = False):
        self._current_task = task_id
        self._use_query = use_query

    def forward(self, images):
        x = self.vit.patch_embed(images)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        batch = x.size(0)
        cls = self.vit.cls_token.expand(batch, -1, -1)
        if self._use_query:
            task_id = self._query_task(F.normalize(x.mean(dim=1), dim=-1))
        else:
            task_id = self._current_task
        g = self.g_prompt.unsqueeze(0).expand(batch, -1, -1)
        e = self.e_prompt[task_id].unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat([cls, g, e, x], dim=1)
        extra = self.prompt_pos.unsqueeze(0).expand(batch, -1, -1)
        cls_pos = self.vit.pos_embed[:, :1].expand(batch, -1, -1)
        patch_pos = self.vit.pos_embed[:, 1:].expand(batch, -1, -1)
        tokens = tokens + torch.cat([cls_pos, extra, patch_pos], dim=1)
        tokens = self.vit.pos_drop(tokens)
        tokens = self.vit.blocks(tokens)
        tokens = self.vit.norm(tokens)
        return self.head(tokens[:, 0])

    def trainable_parameters(self, task_id: int):
        return [
            self.g_prompt,
            self.e_prompt,
            self.prompt_pos,
            self.task_keys,
            *self.head.parameters(),
        ]


def imagenet_normalize(images, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images.to(device) - mean) / std


@torch.no_grad()
def evaluate_task(net, manager, task_id, device, batch_size):
    loader = DataLoader(manager.get_task_datasets(task_id), batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    net.eval()
    net.set_task(task_id, use_query=True)
    for images, labels in loader:
        labels = labels.to(device)
        logits = net(imagenet_normalize(images, device))
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += len(labels)
    return (correct / total) * 100.0 if total else 0.0


def train_task(net, manager, task_id, epochs, lr, device, batch_size):
    loader = DataLoader(manager.get_task_datasets(task_id), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(net.trainable_parameters(task_id), lr=lr)
    net.train()
    net.set_task(task_id, use_query=False)
    for epoch in range(epochs):
        running = 0.0
        count = 0
        for images, labels in tqdm(loader, desc=f"DualPrompt train task {task_id + 1} epoch {epoch + 1}"):
            labels = labels.to(device)
            logits = net(imagenet_normalize(images, device))
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            count += 1
        print(f"  epoch {epoch + 1}/{epochs} loss={running / max(count, 1):.4f}")


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
    vit = load_vit(device)
    net = DualPromptNet(
        vit,
        num_tasks=args.num_tasks,
        g_layers=args.g_layers,
        e_layers=args.e_layers,
        g_length=args.g_length,
        e_length=args.e_length,
    ).to(device)

    results = np.zeros((args.num_tasks, args.num_tasks), dtype=np.float64)
    for task_id in range(args.num_tasks):
        print(f"\n--- DualPrompt task {task_id + 1}/{args.num_tasks} ---")
        train_task(net, manager, task_id, args.epochs, args.lr, device, args.batch_size)
        for seen in range(task_id + 1):
            acc = evaluate_task(net, manager, seen, device, args.batch_size)
            results[task_id, seen] = acc
            print(f"  Task {seen + 1} after task {task_id + 1}: {acc:.2f}%")

    metrics = continual_metrics(results)
    print(f"Average Accuracy (%): {metrics['avg_acc']:.2f}")
    print(f"Backward Transfer (BWT): {metrics['bwt']:.2f}")
    write_result(
        {
            "variant": "dualprompt",
            "dataset": "imagenet",
            "protocol": "continual",
            "seed": args.seed,
            **metrics,
        },
        args.output_json,
        args.results_dir,
    )


if __name__ == "__main__":
    main()
