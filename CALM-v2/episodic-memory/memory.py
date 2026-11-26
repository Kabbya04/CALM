# File: memory.py
# Enhanced Episodic Memory with DeepSeek OCR-Inspired Compression
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from typing import Tuple, List, Optional
import time

class CompressionNetwork(nn.Module):
    """
    Learned embedding compression network inspired by DeepSeek OCR.
    Compresses 512-dim CLIP embeddings to smaller dimensions (128, 256).
    """
    def __init__(self, input_dim=512, compressed_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        
        # Encoder: 512 -> 256 -> compressed_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, compressed_dim)
        )
        
        # Decoder: compressed_dim -> 256 -> 512
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compress embeddings."""
        return self.encoder(embeddings)
    
    def decode(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress embeddings."""
        return self.decoder(compressed)
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: compress and reconstruct."""
        compressed = self.encode(embeddings)
        reconstructed = self.decode(compressed)
        return compressed, reconstructed


class HierarchicalEpisodicMemory:
    """
    Hierarchical episodic memory with optional compression.

    Tiers:
    - recent: full 512-d embeddings while age <= tau_recent
    - medium: 256-d codes while age <= tau_medium
    - old: 128-d codes after that

    Migration only happens when increment_episode() advances current_episode
    far enough past an example's timestamp. Use age_mode='per_example' in the
    runner if a 5-10 task stream should reach medium/old.
    """
    def __init__(
        self,
        device='cpu',
        compression_ratios=None,
        tau_recent=100,
        tau_medium=500,
        flat_memory=False,
        max_bytes=None,
    ):
        self.device = device
        self.flat_memory = bool(flat_memory)
        self.max_bytes = max_bytes
        self.last_migration_moved = 0

        if compression_ratios is None:
            compression_ratios = {
                'recent': (512, 1),
                'medium': (256, 2),
                'old': (128, 4),
            }
        self.compression_ratios = compression_ratios

        self.memory_tiers = {
            'recent': defaultdict(list),
            'medium': defaultdict(list),
            'old': defaultdict(list),
        }

        self.compression_networks = {}
        if not self.flat_memory:
            for tier, (compressed_dim, _) in compression_ratios.items():
                if compressed_dim < 512:
                    self.compression_networks[tier] = CompressionNetwork(
                        input_dim=512, compressed_dim=compressed_dim
                    ).to(device)

        self.current_episode = 0
        self.tau_recent = int(tau_recent)
        self.tau_medium = int(tau_medium)
        self.prototypes = defaultdict(dict)
        mode = "flat 512-d" if self.flat_memory else "hierarchical"
        print(
            f"Hierarchical Episodic Memory initialized ({mode}, "
            f"tau_recent={self.tau_recent}, tau_medium={self.tau_medium})."
        )
    
    def _get_tier_for_episode(self, episode: int) -> str:
        """Determine which tier an embedding should be stored in based on age."""
        age = self.current_episode - episode
        if age <= self.tau_recent:
            return 'recent'
        elif age <= self.tau_medium:
            return 'medium'
        else:
            return 'old'
    
    @staticmethod
    def _to_cpu_payload(embedding):
        """Detach and move a tensor or per-variant dict to CPU for storage."""
        if isinstance(embedding, dict):
            return {
                str(name): tensor.detach().float().cpu()
                for name, tensor in embedding.items()
            }
        return embedding.detach().float().cpu()

    def _compress_embedding(self, embedding, tier: str):
        """Compress a tensor or per-variant dict based on tier."""
        if isinstance(embedding, dict):
            return {
                name: self._compress_embedding(tensor, tier)
                for name, tensor in embedding.items()
            }
        if tier == 'recent' or self.flat_memory:
            return embedding
        if tier in self.compression_networks:
            network = self.compression_networks[tier]
            network.eval()
            embedding = embedding.to(self.device)
            with torch.no_grad():
                compressed = network.encode(embedding.unsqueeze(0))
            return compressed.squeeze(0).cpu()
        return embedding

    def _decompress_embedding(self, compressed, tier: str):
        """Decompress a tensor or per-variant dict based on tier."""
        if isinstance(compressed, dict):
            return {
                name: self._decompress_embedding(tensor, tier)
                for name, tensor in compressed.items()
            }
        if tier == 'recent' or self.flat_memory:
            return compressed
        if tier in self.compression_networks:
            network = self.compression_networks[tier]
            network.eval()
            compressed = compressed.to(self.device)
            with torch.no_grad():
                decompressed = network.decode(compressed.unsqueeze(0))
            return decompressed.squeeze(0)
        return compressed

    def add_example(self, embedding, label: int, importance: float = 1.0,
                    task_id: Optional[int] = None):
        """
        Add one labeled example.

        Args:
            embedding: A single 512-d CLIP tensor, or a dict
                ``{variant_name: tensor}`` for per-encoder banks (B/32 + B/16).
            label: Class label
            importance: Importance weight
            task_id: Optional task id
        """
        embedding = self._to_cpu_payload(embedding)
        tier = 'recent'
        compressed = self._compress_embedding(embedding, tier)
        timestamp = self.current_episode
        self.memory_tiers[tier][label].append((compressed, timestamp, importance, task_id))
        self.enforce_budget()
    
    def _migrate_memory(self):
        """Migrate old examples to appropriate tiers based on age."""
        self.last_migration_moved = 0
        if self.flat_memory:
            return
        for tier in ['recent', 'medium']:
            for class_id in list(self.memory_tiers[tier].keys()):
                examples = self.memory_tiers[tier][class_id]
                new_examples = []
                migrated = []
                
                for emb, timestamp, importance, task_id in examples:
                    age = self.current_episode - timestamp
                    new_tier = self._get_tier_for_episode(timestamp)
                    
                    if new_tier != tier:
                        # Need to migrate
                        # Decompress if needed
                        if tier != 'recent':
                            emb = self._decompress_embedding(emb, tier)
                        # Recompress for new tier
                        emb = self._compress_embedding(emb, new_tier)
                        migrated.append((emb, timestamp, importance, task_id))
                    else:
                        new_examples.append((emb, timestamp, importance, task_id))
                
                # Update current tier
                self.memory_tiers[tier][class_id] = new_examples
                
                # Add to new tier
                if migrated:
                    self.last_migration_moved += len(migrated)
                    target_tier = self._get_tier_for_episode(migrated[0][1])
                    for item in migrated:
                        self.memory_tiers[target_tier][class_id].append(item)
    
    def _weighted_mean(self, items):
        """items: list of (tensor_or_dict, importance)."""
        if not items:
            return None
        sample = items[0][0]
        weights = torch.tensor(
            [w for _, w in items], dtype=torch.float32, device=self.device
        )
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)

        if isinstance(sample, dict):
            out = {}
            for name in sample.keys():
                stacked = torch.stack([pair[0][name].to(self.device) for pair in items])
                out[name] = (stacked * weights.unsqueeze(1)).sum(dim=0).cpu()
            return out

        stacked = torch.stack([pair[0].to(self.device) for pair in items])
        return (stacked * weights.unsqueeze(1)).sum(dim=0).cpu()

    def _consolidate_old_memory(self):
        """Consolidate very old examples into a single weighted code per class."""
        for class_id in list(self.memory_tiers['old'].keys()):
            examples = self.memory_tiers['old'][class_id]
            if len(examples) <= 5:
                continue
            decompressed = []
            for emb, timestamp, importance, task_id in examples:
                decompressed.append((self._decompress_embedding(emb, 'old'), importance))
            prototype = self._weighted_mean(decompressed)
            prototype_compressed = self._compress_embedding(prototype, 'old')
            task_id = examples[0][3] if examples[0][3] is not None else None
            self.memory_tiers['old'][class_id] = [
                (prototype_compressed, self.current_episode, 1.0, task_id)
            ]
            if task_id is not None:
                self.prototypes[task_id][class_id] = prototype_compressed
    
    def increment_episode(self, steps=1):
        """Advance the age clock and migrate examples that crossed a tier."""
        for _ in range(max(1, int(steps))):
            self.current_episode += 1
            if not self.flat_memory:
                self._migrate_memory()
                if self.current_episode % 200 == 0:
                    self._consolidate_old_memory()
            self.enforce_budget()

    def _tier_dim(self, tier: str) -> int:
        return int(self.compression_ratios.get(tier, (512, 1))[0])

    def _payload_floats(self, payload, tier: str) -> int:
        """Number of float32 values stored for one example payload."""
        dim = self._tier_dim(tier)
        if isinstance(payload, dict):
            return len(payload) * dim
        return dim

    def tier_counts(self) -> dict:
        counts = {}
        for tier, by_class in self.memory_tiers.items():
            counts[tier] = sum(len(examples) for examples in by_class.values())
        return counts

    def encoder_names(self) -> List[str]:
        """Return sorted encoder keys if multi-encoder storage is in use."""
        for tier in self.memory_tiers.values():
            for examples in tier.values():
                if examples:
                    payload = examples[0][0]
                    if isinstance(payload, dict):
                        return sorted(payload.keys())
                    return []
        return []

    def stored_bytes(self, include_codecs: bool = False) -> int:
        total = 0
        for tier, by_class in self.memory_tiers.items():
            for examples in by_class.values():
                for payload, *_ in examples:
                    total += self._payload_floats(payload, tier) * 4
        if include_codecs:
            for network in self.compression_networks.values():
                total += sum(int(p.numel()) * 4 for p in network.parameters())
        return int(total)

    def reconstruction_mse(self, embeddings: torch.Tensor) -> dict:
        """MSE of each trained codec on a batch of 512-d embeddings."""
        if embeddings is None or embeddings.numel() == 0:
            return {}
        embeddings = embeddings.to(self.device)
        scores = {}
        for tier, network in self.compression_networks.items():
            network.eval()
            with torch.no_grad():
                _, reconstructed = network(embeddings)
                scores[tier] = float(F.mse_loss(reconstructed, embeddings).item())
        return scores

    def memory_stats(self) -> dict:
        return {
            "episode": self.current_episode,
            "examples": len(self),
            "tier_counts": self.tier_counts(),
            "stored_bytes": self.stored_bytes(),
            "last_migration_moved": self.last_migration_moved,
            "flat_memory": self.flat_memory,
            "tau_recent": self.tau_recent,
            "tau_medium": self.tau_medium,
            "encoder_names": self.encoder_names(),
            "per_encoder": bool(self.encoder_names()),
        }

    def enforce_budget(self):
        """Drop oldest examples until stored codes fit in max_bytes."""
        if self.max_bytes is None:
            return
        while self.stored_bytes() > self.max_bytes and len(self) > 0:
            oldest = None
            for tier in ("recent", "medium", "old"):
                for class_id, examples in self.memory_tiers[tier].items():
                    for idx, item in enumerate(examples):
                        timestamp = item[1]
                        if oldest is None or timestamp < oldest[0]:
                            oldest = (timestamp, tier, class_id, idx)
            if oldest is None:
                break
            _, tier, class_id, idx = oldest
            del self.memory_tiers[tier][class_id][idx]

    def save_codecs(self, path: str):
        payload = {
            name: network.state_dict()
            for name, network in self.compression_networks.items()
        }
        torch.save(payload, path)

    def load_codecs(self, path: str):
        payload = torch.load(path, map_location=self.device)
        for name, state in payload.items():
            if name in self.compression_networks:
                self.compression_networks[name].load_state_dict(state)
                self.compression_networks[name].eval()
    
    def _collect_class_examples(self, class_id: int):
        all_examples = []
        for tier in ['recent', 'medium', 'old']:
            if class_id not in self.memory_tiers[tier]:
                continue
            for emb, timestamp, importance, task_id in self.memory_tiers[tier][class_id]:
                all_examples.append((self._decompress_embedding(emb, tier), importance))
        return all_examples

    def get_prototypes_by_variant(self) -> Tuple[dict, List[int]]:
        """
        Per-encoder class means.

        Returns:
            prototypes: {variant: Tensor[num_classes, dim]} on self.device
            class_ids: shared class order
        """
        all_class_ids = set()
        for tier in self.memory_tiers.values():
            all_class_ids.update(tier.keys())
        if not all_class_ids:
            return {}, []

        class_ids = sorted(all_class_ids)
        names = self.encoder_names()
        if not names:
            # Single-tensor storage: expose as one anonymous bank.
            bank, ids = self.get_prototypes(variant=None)
            return {"__single__": bank}, ids

        banks = {name: [] for name in names}
        kept_ids = []
        for class_id in class_ids:
            examples = self._collect_class_examples(class_id)
            if not examples:
                continue
            mean = self._weighted_mean(examples)
            kept_ids.append(class_id)
            for name in names:
                banks[name].append(mean[name].to(self.device))

        if not kept_ids:
            return {}, []
        return {
            name: torch.stack(tensors)
            for name, tensors in banks.items()
        }, kept_ids

    def get_prototypes(self, variant: Optional[str] = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Class-mean prototypes.

        - Multi-encoder memory + variant set: that encoder's bank.
        - Multi-encoder memory + variant None: first encoder bank (legacy callers);
          prefer get_prototypes_by_variant() for ensemble scoring.
        - Single-tensor memory: the only bank.
        """
        names = self.encoder_names()
        if names:
            banks, class_ids = self.get_prototypes_by_variant()
            if not class_ids:
                return torch.empty(0), []
            key = variant if variant in banks else names[0]
            return banks[key], class_ids

        all_class_ids = set()
        for tier in self.memory_tiers.values():
            all_class_ids.update(tier.keys())
        if not all_class_ids:
            return torch.empty(0), []

        class_ids = sorted(all_class_ids)
        prototypes = []
        kept_ids = []
        for class_id in class_ids:
            examples = self._collect_class_examples(class_id)
            if not examples:
                continue
            mean = self._weighted_mean(examples)
            prototypes.append(mean.to(self.device))
            kept_ids.append(class_id)

        if not prototypes:
            return torch.empty(0), []
        return torch.stack(prototypes), kept_ids

    def get_seen_classes(self) -> List[int]:
        """Returns list of unique class labels seen so far."""
        all_classes = set()
        for tier in self.memory_tiers.values():
            all_classes.update(tier.keys())
        return sorted(all_classes)
    
    def __len__(self):
        """Returns total number of examples stored in memory."""
        total = 0
        for tier in self.memory_tiers.values():
            for examples in tier.values():
                total += len(examples)
        return total
    
    def train_compression_networks(self, embeddings: torch.Tensor, epochs=50, lr=0.001):
        """
        Train compression networks on a set of embeddings.
        
        Args:
            embeddings: Tensor of shape [N, 512] with sample embeddings
            epochs: Number of training epochs
            lr: Learning rate
        """
        embeddings = embeddings.to(self.device)
        
        for tier, network in self.compression_networks.items():
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            network.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                compressed, reconstructed = network(embeddings)
                loss = criterion(reconstructed, embeddings)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Tier {tier} compression network - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            network.eval()
            print(f"  Compression network for tier '{tier}' trained successfully.")
