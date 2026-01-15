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
    Enhanced episodic memory with hierarchical compression inspired by DeepSeek OCR.
    
    Three-tier memory architecture:
    - M_recent: Full 512-dim embeddings (0-100 episodes ago)
    - M_medium: 2× compressed 256-dim embeddings (100-500 episodes ago)
    - M_old: 4× compressed 128-dim embeddings or prototypes (>500 episodes ago)
    """
    def __init__(self, device='cpu', compression_ratios=None):
        """
        Initialize hierarchical memory.
        
        Args:
            device: Device to store tensors on
            compression_ratios: Dict with compression ratios for each tier
        """
        self.device = device
        
        # Compression ratios: {tier: (compressed_dim, compression_factor)}
        if compression_ratios is None:
            compression_ratios = {
                'recent': (512, 1),      # No compression
                'medium': (256, 2),      # 2× compression
                'old': (128, 4)          # 4× compression
            }
        self.compression_ratios = compression_ratios
        
        # Memory tiers: {tier: {class_id: [(embedding, timestamp, importance)]}}
        self.memory_tiers = {
            'recent': defaultdict(list),
            'medium': defaultdict(list),
            'old': defaultdict(list)
        }
        
        # Compression networks for each tier
        self.compression_networks = {}
        for tier, (compressed_dim, _) in compression_ratios.items():
            if compressed_dim < 512:
                self.compression_networks[tier] = CompressionNetwork(
                    input_dim=512, compressed_dim=compressed_dim
                ).to(device)
        
        # Episode counter
        self.current_episode = 0
        
        # Compression thresholds (in episodes)
        self.tau_recent = 100  # Recent memory threshold
        self.tau_medium = 500  # Medium memory threshold
        
        # Prototype storage for old tasks
        self.prototypes = defaultdict(dict)  # {task_id: {class_id: prototype}}
        
        print("Hierarchical Episodic Memory initialized with compression support.")
    
    def _get_tier_for_episode(self, episode: int) -> str:
        """Determine which tier an embedding should be stored in based on age."""
        age = self.current_episode - episode
        if age <= self.tau_recent:
            return 'recent'
        elif age <= self.tau_medium:
            return 'medium'
        else:
            return 'old'
    
    def _compress_embedding(self, embedding: torch.Tensor, tier: str) -> torch.Tensor:
        """Compress embedding based on tier."""
        if tier == 'recent':
            return embedding  # No compression
        elif tier in self.compression_networks:
            network = self.compression_networks[tier]
            network.eval()
            # Ensure embedding is on correct device
            embedding = embedding.to(self.device)
            with torch.no_grad():
                compressed = network.encode(embedding.unsqueeze(0))
            return compressed.squeeze(0).cpu()  # Return to CPU for storage
        else:
            return embedding
    
    def _decompress_embedding(self, compressed: torch.Tensor, tier: str) -> torch.Tensor:
        """Decompress embedding based on tier."""
        if tier == 'recent':
            return compressed  # No compression
        elif tier in self.compression_networks:
            network = self.compression_networks[tier]
            network.eval()
            # Ensure compressed is on correct device
            compressed = compressed.to(self.device)
            with torch.no_grad():
                decompressed = network.decode(compressed.unsqueeze(0))
            return decompressed.squeeze(0)
        else:
            return compressed
    
    def add_example(self, embedding: torch.Tensor, label: int, importance: float = 1.0, 
                   task_id: Optional[int] = None):
        """
        Add example to memory with appropriate compression.
        
        Args:
            embedding: Full 512-dim CLIP embedding
            label: Class label
            importance: Importance score (for compression decisions)
            task_id: Optional task identifier
        """
        # Ensure embedding is on CPU and detached
        embedding = embedding.detach().cpu()
        
        # Determine tier based on current episode
        tier = 'recent'  # New examples always go to recent tier
        
        # Compress if needed
        compressed = self._compress_embedding(embedding, tier)
        
        # Store with timestamp and importance
        timestamp = self.current_episode
        self.memory_tiers[tier][label].append((compressed, timestamp, importance, task_id))
    
    def _migrate_memory(self):
        """Migrate old examples to appropriate tiers based on age."""
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
                    target_tier = self._get_tier_for_episode(migrated[0][1])
                    for item in migrated:
                        self.memory_tiers[target_tier][class_id].append(item)
    
    def _consolidate_old_memory(self):
        """Consolidate very old examples into prototypes."""
        for class_id in list(self.memory_tiers['old'].keys()):
            examples = self.memory_tiers['old'][class_id]
            
            if len(examples) > 5:  # Only consolidate if enough examples
                # Decompress all embeddings
                decompressed = []
                for emb, timestamp, importance, task_id in examples:
                    decompressed_emb = self._decompress_embedding(emb, 'old')
                    decompressed.append((decompressed_emb, importance))
                
                # Compute weighted prototype
                embeddings = torch.stack([e for e, _ in decompressed])
                weights = torch.tensor([w for _, w in decompressed], dtype=torch.float32)
                if weights.sum() > 0:
                    weights = weights / weights.sum()  # Normalize
                else:
                    weights = torch.ones_like(weights) / len(weights)
                
                prototype = (embeddings * weights.unsqueeze(1)).sum(dim=0)
                
                # Store prototype (compressed)
                prototype_compressed = self._compress_embedding(prototype, 'old')
                
                # Replace examples with prototype
                # Store task_id from first example
                task_id = examples[0][3] if examples[0][3] is not None else None
                self.memory_tiers['old'][class_id] = [
                    (prototype_compressed, self.current_episode, 1.0, task_id)
                ]
                
                # Also store in prototype dict
                if task_id is not None:
                    self.prototypes[task_id][class_id] = prototype_compressed
    
    def increment_episode(self):
        """Increment episode counter and migrate/consolidate memory."""
        self.current_episode += 1
        
        # Periodically migrate memory
        if self.current_episode % 50 == 0:
            self._migrate_memory()
        
        # Periodically consolidate old memory
        if self.current_episode % 200 == 0:
            self._consolidate_old_memory()
    
    def get_prototypes(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Get prototypes for all classes, decompressing as needed.
        
        Returns:
            prototypes: Tensor of shape [num_classes, 512]
            class_ids: List of class labels
        """
        all_class_ids = set()
        for tier in self.memory_tiers.values():
            all_class_ids.update(tier.keys())
        
        if not all_class_ids:
            return torch.empty(0), []
        
        class_ids = sorted(all_class_ids)
        prototypes = []
        
        for class_id in class_ids:
            # Collect examples from all tiers
            all_examples = []
            for tier in ['recent', 'medium', 'old']:
                if class_id in self.memory_tiers[tier]:
                    for emb, timestamp, importance, task_id in self.memory_tiers[tier][class_id]:
                        # Decompress if needed
                        decompressed = self._decompress_embedding(emb, tier)
                        all_examples.append((decompressed, importance))
            
            if not all_examples:
                continue
            
            # Compute weighted prototype
            embeddings = torch.stack([e for e, _ in all_examples])
            weights = torch.tensor([w for _, w in all_examples], dtype=torch.float32)
            if weights.sum() > 0:
                weights = weights / weights.sum()  # Normalize
            else:
                weights = torch.ones_like(weights) / len(weights)
            
            prototype = (embeddings * weights.unsqueeze(1)).sum(dim=0)
            prototypes.append(prototype)
        
        if not prototypes:
            return torch.empty(0), []
        
        prototypes_tensor = torch.stack(prototypes)
        return prototypes_tensor, class_ids
    
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
