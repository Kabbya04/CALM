"""
Generic class-index helper used by the CALM-v2 multi-dataset runner.
"""

import numpy as np


def get_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    raise AttributeError("Dataset does not expose targets or labels.")


def build_class_index(dataset) -> dict[int, np.ndarray]:
    targets = get_targets(dataset)
    return {
        int(class_id): np.flatnonzero(targets == class_id)
        for class_id in np.unique(targets)
    }

