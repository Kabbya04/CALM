"""Measure labels requested vs recovered accuracy (review gap G13)."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, STL10, SVHN
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
V1_MEM = os.path.join(
    os.path.dirname(HERE),
    "CALM-v1",
    "CALM-Continual_Adaptive_Learning_Model",
    "final-evaluation",
)
sys.path.insert(0, HERE)
sys.path.insert(0, V1_MEM)

from memory import EpisodicMemory  # type: ignore

from lib.clip_backbone import embed_pils, load_clip
from lib.devices import get_device, set_seed
from lib.results_io import add_output_flag, write_result

DATASETS = {
    "cifar10": lambda root: CIFAR10(root=root, train=False, download=True),
    "cifar100": lambda root: CIFAR100(root=root, train=False, download=True),
    "stl10": lambda root: STL10(root=root, split="test", download=True),
    "svhn": lambda root: SVHN(root=root, split="test", download=True),
    "fashionmnist": lambda root: FashionMNIST(root=root, train=False, download=True),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Correction-cost experiment.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=2000)
    parser.add_argument("--rho", type=float, default=0.7, help="Target deploy rate for the threshold.")
    parser.add_argument(
        "--mode",
        choices=["threshold", "none", "oracle", "inject"],
        default="threshold",
    )
    parser.add_argument("--error_rate", type=float, default=0.1, help="Used with --mode inject.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    add_output_flag(parser)
    return parser.parse_args()


def get_image(dataset, index):
    image, label = dataset[index]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    image = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224)]
    )(image)
    return image, int(label)


def targets_of(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    raise AttributeError("dataset has no labels")


def choose_threshold(confidences, rho):
    if len(confidences) == 0:
        return 0.0
    return float(np.quantile(confidences, 1.0 - rho))


def prototype_scores(memory, features):
    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0:
        return None, None, None
    prototypes = F.normalize(prototypes.to(features.device), p=2, dim=1)
    sims = features @ prototypes.T
    probs = F.softmax(sims, dim=1)
    conf, pred_idx = probs.max(dim=1)
    pred = torch.tensor([class_ids[i] for i in pred_idx], device=features.device)
    return pred, conf, class_ids


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    data_root = args.data_root or os.path.join(HERE, "data")
    dataset = DATASETS[args.dataset](data_root)
    model, processor = load_clip(device)
    targets = targets_of(dataset)
    class_ids = sorted(int(x) for x in np.unique(targets))
    class_to_indices = {cid: np.flatnonzero(targets == cid) for cid in class_ids}

    memory = EpisodicMemory()
    used = set()
    for cid in class_ids:
        take = min(args.k_shot, len(class_to_indices[cid]))
        chosen = np.random.choice(class_to_indices[cid], take, replace=False)
        images, labels = [], []
        for idx in chosen:
            image, label = get_image(dataset, int(idx))
            images.append(image)
            labels.append(label)
            used.add(int(idx))
        feats = embed_pils(images, model, processor, device, args.batch_size)
        for emb, label in zip(feats, labels):
            memory.add_example(emb, label)

    query_pool = [i for i in range(len(dataset)) if i not in used]
    np.random.shuffle(query_pool)
    query_pool = query_pool[: args.max_queries]

    # Calibrate threshold on a held-out slice of the query pool.
    calib_n = min(300, max(50, len(query_pool) // 5))
    calib_idx, stream_idx = query_pool[:calib_n], query_pool[calib_n:]
    calib_conf = []
    for start in range(0, len(calib_idx), args.batch_size):
        batch = calib_idx[start : start + args.batch_size]
        images = [get_image(dataset, i)[0] for i in batch]
        feats = embed_pils(images, model, processor, device, args.batch_size)
        _, conf, _ = prototype_scores(memory, feats)
        if conf is not None:
            calib_conf.extend(conf.cpu().tolist())
    threshold = choose_threshold(calib_conf, args.rho)

    labels_used = 0
    correct = 0
    total = 0
    for index in tqdm(stream_idx, desc="Correction stream"):
        image, true = get_image(dataset, index)
        feat = embed_pils([image], model, processor, device, 1)
        pred, conf, _ = prototype_scores(memory, feat)
        if pred is None:
            pred_label = true
            confidence = 0.0
        else:
            pred_label = int(pred.item())
            confidence = float(conf.item())

        request = False
        if args.mode == "none":
            request = False
        elif args.mode == "oracle":
            request = pred_label != true
        elif args.mode == "inject":
            if pred_label == true and np.random.rand() < args.error_rate:
                pred_label = int(np.random.choice([c for c in class_ids if c != true]))
            request = confidence < threshold
        else:
            request = confidence < threshold

        if request:
            memory.add_example(feat.squeeze(0), true)
            labels_used += 1
            pred_label = true

        correct += int(pred_label == true)
        total += 1

    accuracy = 100.0 * correct / total if total else 0.0
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Labels requested: {labels_used}")
    print(f"Labels per query: {labels_used / total if total else 0:.3f}")
    write_result(
        {
            "variant": f"correction_{args.mode}",
            "dataset": args.dataset,
            "protocol": "correction_cost",
            "seed": args.seed,
            "k_shot": args.k_shot,
            "rho": args.rho,
            "error_rate": args.error_rate if args.mode == "inject" else 0.0,
            "accuracy": accuracy,
            "labels_used": labels_used,
            "queries": total,
            "labels_per_query": labels_used / total if total else 0.0,
            "threshold": threshold,
        },
        args.output_json,
        args.results_dir,
    )


if __name__ == "__main__":
    main()
