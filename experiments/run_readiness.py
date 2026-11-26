"""
Protocol-matched readiness evaluation for CALM-v1 and CALM-v2.

Reports coverage, deferral rate, and deployed error at several target
deploy rates. Disagreement gating is optional and off by default.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, STL10, SVHN

import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
CALM = os.path.dirname(HERE)
V1_MEM = os.path.join(CALM, "CALM-v1", "CALM-Continual_Adaptive_Learning_Model", "final-evaluation")
V2_MEM = os.path.join(CALM, "CALM-v2", "episodic-memory")
V2_AGENT = os.path.join(CALM, "CALM-v2", "readiness-agent")
sys.path.insert(0, HERE)
sys.path.insert(0, V2_MEM)
sys.path.insert(0, V2_AGENT)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from ensemble_calibration import EnsembleCLIPModels  # type: ignore
from agent import ReadinessAgent  # type: ignore

FlatMemory = _load("calm_v1_memory", os.path.join(V1_MEM, "memory.py")).EpisodicMemory
HierarchicalEpisodicMemory = _load(
    "calm_v2_memory", os.path.join(V2_MEM, "memory.py")
).HierarchicalEpisodicMemory

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
    parser = argparse.ArgumentParser(description="Matched readiness evaluation.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--variant", choices=["v1", "v2"], default="v1")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--rho",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.85, 1.0],
        help="Target deploy rates. One run reports all of them.",
    )
    parser.add_argument(
        "--use_disagreement",
        action="store_true",
        help="v2 only: use ensemble agreement instead of the calibrated threshold.",
    )
    add_output_flag(parser)
    return parser.parse_args()


def get_image(dataset, index):
    image, label = dataset[index]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    image = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)
    return image, int(label)


def targets_of(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    raise AttributeError("dataset has no labels")


def build_memory(dataset, embed_fn, k_shot, device, ensemble=None):
    targets = targets_of(dataset)
    class_ids = sorted(int(x) for x in np.unique(targets))
    memory = HierarchicalEpisodicMemory(device=device) if embed_fn.kind == "v2" else FlatMemory()
    used = set()
    for cid in class_ids:
        indices = np.flatnonzero(targets == cid)
        take = min(k_shot, len(indices))
        chosen = np.random.choice(indices, take, replace=False)
        images, labels = [], []
        for idx in chosen:
            image, label = get_image(dataset, int(idx))
            images.append(image)
            labels.append(label)
            used.add(int(idx))
        if embed_fn.kind == "v2":
            all_features = ensemble.get_image_features(images)
            for j, label in enumerate(labels):
                payload = {name: tensor[j].cpu() for name, tensor in all_features.items()}
                memory.add_example(payload, label, importance=1.0)
            del all_features
        else:
            feats = embed_fn(images)
            for emb, label in zip(feats, labels):
                memory.add_example(emb.cpu(), label)
    return memory, used


def scores_for(memory, images, embed_fn, ensemble, temperature):
    if embed_fn.kind == "v1":
        feats = embed_fn(images)
        prototypes, class_ids = memory.get_prototypes()
        prototypes = F.normalize(prototypes.to(feats.device), p=2, dim=1)
        sims = feats @ prototypes.T
        probs = F.softmax(sims / temperature, dim=1)
        conf, pred_idx = probs.max(dim=1)
        pred = torch.tensor([class_ids[i] for i in pred_idx], device=feats.device)
        return pred, conf, None

    banks, class_ids = memory.get_prototypes_by_variant()
    if not class_ids:
        empty = torch.empty(0, dtype=torch.long, device=embed_fn.device)
        return empty, empty.float(), None
    all_features = ensemble.get_image_features(images)
    similarities = ensemble.compute_prototype_similarities(all_features, banks, class_ids)
    probs, stats = ensemble.ensemble_confidence(similarities, temperature=temperature)
    conf, pred_idx = probs.max(dim=1)
    pred = torch.tensor([class_ids[i] for i in pred_idx], device=embed_fn.device)
    return pred, conf, stats.get("agreement")


def curve_at_rho(confidences, predictions, labels, agreements, rho, use_disagreement):
    if use_disagreement and agreements is not None:
        agent = ReadinessAgent(threshold=max(min(rho, 0.99), 0.01), adaptive=False)
        decisions = [
            agent.decide(float(c), agreement=float(a))
            for c, a in zip(confidences, agreements)
        ]
    else:
        threshold = float(np.quantile(confidences, 1.0 - rho)) if rho < 1.0 else -1.0
        decisions = ["Deploy" if c >= threshold else "Request Feedback" for c in confidences]
    metrics = ReadinessAgent.get_metrics(predictions, labels, decisions)
    deployed_error = 1.0 - metrics["Deployed Accuracy"]
    return {
        "rho": rho,
        "coverage": metrics["Total Deployed"],
        "deferral_rate": metrics["Total Requested"],
        "deployed_accuracy": metrics["Deployed Accuracy"],
        "deployed_error": deployed_error,
        "model_accuracy": metrics["Total Model Accuracy"],
        "fpr": metrics["False Positive Rate (FPR)"],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    data_root = args.data_root or os.path.join(HERE, "data")
    dataset = DATASETS[args.dataset](data_root)
    ensemble = None
    if args.variant == "v1":
        model, processor = load_clip(device)

        def embed_fn(images):
            return embed_pils(images, model, processor, device, args.batch_size)

        embed_fn.kind = "v1"
        embed_fn.device = device
    else:
        ensemble = EnsembleCLIPModels(device=device)
        first = list(ensemble.models.keys())[0]

        def embed_fn(images):
            return ensemble.get_image_features(images, variant=first)

        embed_fn.kind = "v2"
        embed_fn.device = device

    memory, used = build_memory(dataset, embed_fn, args.k_shot, device, ensemble=ensemble)
    query_idx = [i for i in range(len(dataset)) if i not in used]
    preds, confs, labels, agrees = [], [], [], []
    for start in range(0, len(query_idx), args.batch_size):
        batch = query_idx[start : start + args.batch_size]
        images, y = zip(*(get_image(dataset, i) for i in batch))
        pred, conf, agree = scores_for(memory, list(images), embed_fn, ensemble, 1.0)
        preds.extend(pred.cpu().tolist())
        confs.extend(conf.cpu().tolist())
        labels.extend(y)
        if agree is None:
            agrees.extend([None] * len(y))
        else:
            agrees.extend(agree.detach().cpu().tolist())

    agreements = None if all(a is None for a in agrees) else np.array(agrees, dtype=np.float64)
    rows = [
        curve_at_rho(
            np.array(confs),
            np.array(preds),
            np.array(labels),
            agreements,
            rho,
            args.use_disagreement and args.variant == "v2",
        )
        for rho in args.rho
    ]
    for row in rows:
        print(
            f"rho={row['rho']:.2f}  coverage={row['coverage']:.3f}  "
            f"deferral={row['deferral_rate']:.3f}  deployed_err={row['deployed_error']:.3f}"
        )
    variant_name = (
        "readiness_v2pe" if args.variant == "v2" else f"readiness_{args.variant}"
    )
    write_result(
        {
            "variant": variant_name,
            "dataset": args.dataset,
            "protocol": "readiness",
            "seed": args.seed,
            "k_shot": args.k_shot,
            "use_disagreement": bool(args.use_disagreement),
            "curve": rows,
        },
        args.output_json,
        args.results_dir,
    )


if __name__ == "__main__":
    main()
