"""
Correction-loop experiment (advice S0–S4).

Headline metrics never credit a just-labeled query as model accuracy.
Reports autonomous accuracy, held-out accuracy after writes, and label cost.
"""

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

from lib.clip_backbone import embed_pils, embed_text, load_clip
from lib.devices import get_device, set_seed
from lib.results_io import add_output_flag, write_result

DATASETS = {
    "cifar10": lambda root: CIFAR10(root=root, train=False, download=True),
    "cifar100": lambda root: CIFAR100(root=root, train=False, download=True),
    "stl10": lambda root: STL10(root=root, split="test", download=True),
    "svhn": lambda root: SVHN(root=root, split="test", download=True),
    "fashionmnist": lambda root: FashionMNIST(root=root, train=False, download=True),
}

CLASS_NAMES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Correction-loop S0–S4 experiment.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=2000)
    parser.add_argument("--heldout", type=int, default=500)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument(
        "--system",
        choices=["S0", "S1", "S2", "S3", "S4", "all"],
        default="S3",
        help="S0 CLIP-ZS, S1 Tip static, S2 Tip+writes, S3 CALM-v1, S4 oracle.",
    )
    parser.add_argument(
        "--mode",
        choices=["threshold", "none", "oracle", "inject"],
        default=None,
        help="Legacy alias: maps to S3/S1/S4/S3+inject.",
    )
    parser.add_argument("--error_rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tip_alpha", type=float, default=1.0)
    parser.add_argument("--tip_beta", type=float, default=5.5)
    parser.add_argument("--prompt", default="a photo of a {}")
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


def class_name_list(dataset_name, class_ids, dataset):
    if dataset_name in CLASS_NAMES:
        return [CLASS_NAMES[dataset_name][i] for i in class_ids]
    if hasattr(dataset, "classes"):
        return [str(dataset.classes[i]).replace("_", " ") for i in class_ids]
    return [f"class {i}" for i in class_ids]


class TipCache:
    def __init__(self, text_features, alpha, beta):
        self.text_features = text_features
        self.alpha = alpha
        self.beta = beta
        self.keys = []
        self.values = []

    def add(self, embeddings, labels, num_classes):
        labels = torch.as_tensor(labels, device=embeddings.device, dtype=torch.long)
        onehot = F.one_hot(labels, num_classes=num_classes).float()
        self.keys.append(F.normalize(embeddings, p=2, dim=1).detach().cpu())
        self.values.append(onehot.detach().cpu())

    def predict(self, query, device):
        query = F.normalize(query, p=2, dim=1)
        clip_logits = 100.0 * query @ self.text_features.to(device).T
        if not self.keys:
            probs = F.softmax(clip_logits, dim=1)
            conf, pred = probs.max(dim=1)
            return pred, conf
        keys = torch.cat(self.keys, dim=0).to(device)
        values = torch.cat(self.values, dim=0).to(device)
        affinity = torch.exp(-self.beta * (1.0 - query @ keys.T))
        cache_logits = affinity @ values
        logits = clip_logits + self.alpha * cache_logits
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        return pred, conf


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


def resolve_systems(args):
    if args.mode is not None:
        mapping = {
            "none": ["S1"],
            "threshold": ["S3"],
            "oracle": ["S4"],
            "inject": ["S3"],
        }
        return mapping[args.mode]
    if args.system == "all":
        return ["S0", "S1", "S2", "S3", "S4"]
    return [args.system]


def evaluate_heldout(predict_fn, heldout_idx, dataset, model, processor, device, batch_size):
    correct = 0
    total = 0
    with torch.inference_mode():
        for start in range(0, len(heldout_idx), batch_size):
            batch = heldout_idx[start : start + batch_size]
            images, labels = zip(*(get_image(dataset, i) for i in batch))
            feats = embed_pils(list(images), model, processor, device, batch_size)
            pred, _ = predict_fn(feats)
            if pred is None:
                continue
            label_t = torch.tensor(labels, device=device)
            correct += int((pred == label_t).sum().item())
            total += len(labels)
    return 100.0 * correct / total if total else 0.0


def run_system(
    system,
    args,
    dataset,
    model,
    processor,
    device,
    class_ids,
    class_to_indices,
    text_features,
):
    set_seed(args.seed)
    used = set()
    seed_images = []
    seed_labels = []
    for cid in class_ids:
        take = min(args.k_shot, len(class_to_indices[cid]))
        chosen = np.random.choice(class_to_indices[cid], take, replace=False)
        for idx in chosen:
            image, label = get_image(dataset, int(idx))
            seed_images.append(image)
            seed_labels.append(label)
            used.add(int(idx))

    remaining = [i for i in range(len(dataset)) if i not in used]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(remaining)
    heldout_n = min(args.heldout, max(50, len(remaining) // 5))
    heldout_idx = remaining[:heldout_n]
    stream_pool = remaining[heldout_n:]
    stream_pool = stream_pool[: args.max_queries]
    calib_n = min(300, max(50, len(stream_pool) // 5))
    calib_idx, stream_idx = stream_pool[:calib_n], stream_pool[calib_n:]

    num_classes = len(class_ids)
    id_to_row = {cid: i for i, cid in enumerate(class_ids)}
    seed_feats = embed_pils(seed_images, model, processor, device, args.batch_size)

    memory = EpisodicMemory()
    tip = TipCache(text_features, args.tip_alpha, args.tip_beta)
    tip.add(seed_feats, [id_to_row[l] for l in seed_labels], num_classes)
    for emb, label in zip(seed_feats, seed_labels):
        memory.add_example(emb.cpu(), label)

    def tip_predict(feats):
        pred_rows, conf = tip.predict(feats, device)
        pred = torch.tensor(
            [class_ids[int(i)] for i in pred_rows.cpu().tolist()], device=device
        )
        return pred, conf

    def calm_predict(feats):
        pred, conf, _ = prototype_scores(memory, feats)
        if pred is None:
            return None, None
        return pred, conf

    def zs_predict(feats):
        logits = 100.0 * F.normalize(feats, p=2, dim=1) @ text_features.T
        probs = F.softmax(logits, dim=1)
        conf, pred_rows = probs.max(dim=1)
        pred = torch.tensor(
            [class_ids[int(i)] for i in pred_rows.cpu().tolist()], device=device
        )
        return pred, conf

    if system in ("S1", "S2"):
        predict_fn = tip_predict
    elif system == "S0":
        predict_fn = zs_predict
    else:
        predict_fn = calm_predict

    # Calibrate threshold on calm/tip confidence (S0 has no gate writes).
    calib_conf = []
    with torch.inference_mode():
        for start in range(0, len(calib_idx), args.batch_size):
            batch = calib_idx[start : start + args.batch_size]
            images = [get_image(dataset, i)[0] for i in batch]
            feats = embed_pils(images, model, processor, device, args.batch_size)
            _, conf = predict_fn(feats)
            if conf is not None:
                calib_conf.extend(conf.detach().cpu().tolist())
    threshold = choose_threshold(calib_conf, args.rho)

    labels_used = 0
    auto_correct = 0
    auto_total = 0
    request_flags = []

    for index in tqdm(stream_idx, desc=f"{system} stream", leave=False):
        image, true = get_image(dataset, index)
        with torch.inference_mode():
            feat = embed_pils([image], model, processor, device, 1)
            pred, conf = predict_fn(feat)
        if pred is None:
            pred_label = true
            confidence = 0.0
        else:
            pred_label = int(pred.item())
            confidence = float(conf.item())

        if system == "S0" or system == "S1":
            request = False
        elif system == "S4":
            request = pred_label != true
        else:
            # S2 / S3 (+ optional inject)
            if args.error_rate > 0 and pred_label == true and np.random.rand() < args.error_rate:
                pred_label = int(np.random.choice([c for c in class_ids if c != true]))
            request = confidence < threshold

        request_flags.append(int(request))

        if request:
            labels_used += 1
            if system in ("S2",):
                tip.add(feat.cpu(), [id_to_row[true]], num_classes)
            if system in ("S3", "S4"):
                memory.add_example(feat.squeeze(0).cpu(), true)
            # Do NOT count this query as autonomous correct.
            continue

        auto_total += 1
        auto_correct += int(pred_label == true)

    autonomous_acc = 100.0 * auto_correct / auto_total if auto_total else 0.0
    heldout_acc = evaluate_heldout(
        predict_fn, heldout_idx, dataset, model, processor, device, args.batch_size
    )
    baseline_heldout = None
    # Static Tip / ZS held-out for reference when system writes.
    if system in ("S2", "S3", "S4"):
        tip_static = TipCache(text_features, args.tip_alpha, args.tip_beta)
        tip_static.add(seed_feats, [id_to_row[l] for l in seed_labels], num_classes)

        def static_predict(feats):
            pred_rows, conf = tip_static.predict(feats, device)
            pred = torch.tensor(
                [class_ids[int(i)] for i in pred_rows.cpu().tolist()], device=device
            )
            return pred, conf

        baseline_heldout = evaluate_heldout(
            static_predict, heldout_idx, dataset, model, processor, device, args.batch_size
        )

    recovered = 0.0
    if baseline_heldout is not None:
        recovered = max(0.0, (heldout_acc - baseline_heldout) / 100.0 * len(heldout_idx))
    labels_per_recovered = (
        float(labels_used / recovered) if recovered > 1e-8 else None
    )

    record = {
        "variant": f"correction_{system}",
        "dataset": args.dataset,
        "protocol": "correction_loop",
        "seed": args.seed,
        "k_shot": args.k_shot,
        "rho": args.rho,
        "error_rate": args.error_rate,
        "system": system,
        "autonomous_accuracy": autonomous_acc,
        "heldout_accuracy_after_stream": heldout_acc,
        "heldout_accuracy_static_tip": baseline_heldout,
        "labels_used": labels_used,
        "queries_stream": len(stream_idx),
        "autonomous_queries": auto_total,
        "labels_per_query": labels_used / len(stream_idx) if stream_idx else 0.0,
        "realized_request_rate": float(np.mean(request_flags)) if request_flags else 0.0,
        "labels_per_recovered_heldout_error": labels_per_recovered,
        "threshold": threshold,
        "heldout_n": len(heldout_idx),
    }
    return record


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
    names = class_name_list(args.dataset, class_ids, dataset)
    prompts = [args.prompt.format(name) for name in names]
    text_features = embed_text(prompts, model, processor, device)

    systems = resolve_systems(args)
    for system in systems:
        record = run_system(
            system,
            args,
            dataset,
            model,
            processor,
            device,
            class_ids,
            class_to_indices,
            text_features,
        )
        print(
            f"{system}: autonomous={record['autonomous_accuracy']:.2f}% "
            f"heldout={record['heldout_accuracy_after_stream']:.2f}% "
            f"labels={record['labels_used']} "
            f"req_rate={record['realized_request_rate']:.3f}"
        )
        out = args.output_json
        if out and len(systems) > 1:
            root, ext = os.path.splitext(out)
            out = f"{root}_{system}{ext}"
        elif out is None:
            out = os.path.join(
                args.results_dir,
                f"corr_{system}_{args.dataset}_seed{args.seed}.json",
            )
        write_result(record, out, args.results_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
