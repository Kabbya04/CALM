import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageNet


class ImageNetSampler:
    """
    Minimal copy of the CALM-v1 ImageNet episodic sampler.

    Used for N-way K-shot episodes in the single-backbone ImageNet
    evaluation for CALM-v2.
    """

    def __init__(self, root_path, split, transform):
        """
        Initialize the sampler by loading the ImageNet dataset and creating a
        robust index by verifying the existence of each file on disk.
        """
        print(f"Loading ImageNet '{split}' split from disk...")
        self.dataset = ImageNet(root=root_path, split=split, transform=transform)

        print("Creating robust class index by verifying files... (This may take a minute)")
        self.class_index = defaultdict(list)

        # self.dataset.samples is a list of tuples: (full_path_to_image, class_index)
        for master_idx, (path, class_idx) in enumerate(tqdm(self.dataset.samples, desc="Verifying samples")):
            if os.path.exists(path):
                self.class_index[class_idx].append(master_idx)

        self.all_class_ids = sorted(self.class_index.keys())
        total_samples = sum(len(v) for v in self.class_index.values())
        print(f"Robust index complete. Found {len(self.all_class_ids)} classes and {total_samples} actual samples.")

        if total_samples == 0:
            raise RuntimeError(
                "The indexer found 0 valid images. This is likely due to a mismatch between the "
                "ImageNet root path and the directory structure. Please ensure the 'val' folder with "
                "class subdirectories is directly inside the provided root path."
            )

    def sample_episode(self, n_way, k_shot, num_query):
        """
        Samples a single N-way, K-shot episode with a query set.
        Returns support and query sets as lists of (PIL_Image, label).
        """
        episode_classes = np.random.choice(self.all_class_ids, n_way, replace=False)

        support_set, query_set = [], []

        for class_id in episode_classes:
            all_indices_for_class = self.class_index[class_id]

            if len(all_indices_for_class) < k_shot + num_query:
                # Not enough images in this class; signal caller to resample episode
                return None, None

            sampled_indices = np.random.choice(
                all_indices_for_class,
                k_shot + num_query,
                replace=False,
            )

            support_indices = sampled_indices[:k_shot]
            query_indices = sampled_indices[k_shot:]

            for idx in support_indices:
                support_set.append(self.dataset[idx])
            for idx in query_indices:
                query_set.append(self.dataset[idx])

        return support_set, query_set

