"""Flat episodic memory used by the CALM-v1 final evaluation."""

from collections import defaultdict

import torch


class EpisodicMemory:
    """Store labeled embeddings and compute one prototype per class."""

    def __init__(self):
        self.storage = defaultdict(list)

    def add_example(self, embedding: torch.Tensor, label: int):
        self.storage[int(label)].append(embedding.detach().cpu())

    def get_prototypes(self):
        class_ids = sorted(self.storage)
        if not class_ids:
            return torch.empty(0), []

        prototypes = torch.stack(
            [torch.stack(self.storage[class_id]).mean(dim=0) for class_id in class_ids]
        )
        return prototypes, class_ids

    def __len__(self):
        return sum(len(items) for items in self.storage.values())

