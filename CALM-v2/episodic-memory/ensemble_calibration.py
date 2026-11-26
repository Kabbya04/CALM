# File: ensemble_calibration.py
# Ensemble Confidence Calibration with Multiple CLIP Variants (Approach 2.2)
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from typing import List, Tuple, Optional
import numpy as np

class EnsembleCLIPModels:
    """
    Manages multiple CLIP model variants for ensemble confidence estimation.
    Supports: ViT-B/32, ViT-B/16, ViT-L/14 (if available)
    """
    def __init__(self, device='cpu', model_variants=None):
        """
        Initialize ensemble of CLIP models.
        
        Args:
            device: Device to run models on
            model_variants: List of model names to use, e.g., ['ViT-B/32', 'ViT-B/16']
        """
        self.device = device
        
        # Default model variants (in order of preference)
        if model_variants is None:
            model_variants = ['ViT-B/32', 'ViT-B/16']  # ViT-L/14 is large, optional
        
        self.model_variants = model_variants
        self.models = {}
        self.processors = {}
        
        # Model name mapping
        model_map = {
            'ViT-B/32': 'openai/clip-vit-base-patch32',
            'ViT-B/16': 'openai/clip-vit-base-patch16',
            'ViT-L/14': 'openai/clip-vit-large-patch14'
        }
        
        # Load models
        print(f"Loading CLIP ensemble models on {device}...")
        for variant in model_variants:
            if variant in model_map:
                try:
                    model_name = model_map[variant]
                    print(f"  Loading {variant} ({model_name})...")
                    # Use the same loader as the working CALM-v1 evaluator.
                    # This allows Transformers to use the available cached
                    # checkpoint format (safetensors or pytorch_model.bin).
                    self.models[variant] = CLIPModel.from_pretrained(model_name).to(device)
                    self.processors[variant] = CLIPProcessor.from_pretrained(model_name)
                    self.models[variant].eval()
                    print(f"  ✓ {variant} loaded successfully")
                except Exception as e:
                    print(f"  ✗ Failed to load {variant}: {e}")
                    print(f"    Continuing with available models...")
            else:
                print(f"  ✗ Unknown variant: {variant}")
        
        if not self.models:
            raise RuntimeError("No CLIP models could be loaded!")
        
        print(f"Ensemble initialized with {len(self.models)} model(s): {list(self.models.keys())}")
        
        # Ensemble weights (learned or uniform)
        self.weights = {variant: 1.0 / len(self.models) for variant in self.models.keys()}
    
    def get_image_features(self, images, variant: Optional[str] = None) -> torch.Tensor:
        """
        Get image features from specified variant or all variants.
        
        Args:
            images: PIL images or list of PIL images
            variant: Specific variant to use, or None for all
        
        Returns:
            If variant specified: [batch_size, embedding_dim]
            If None: Dict of {variant: [batch_size, embedding_dim]}
        """
        if not isinstance(images, list):
            images = [images]

        if variant is not None:
            if variant not in self.models:
                raise ValueError(f"Variant {variant} not available")
            processor = self.processors[variant]
            model = self.models[variant]
            inputs = processor(images=images, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
            del inputs
            return features

        all_features = {}
        for name, model in self.models.items():
            processor = self.processors[name]
            inputs = processor(images=images, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
            all_features[name] = features
            del inputs
        return all_features
    
    def compute_prototype_similarities(self, query_features: dict, prototypes,
                                       class_ids: List[int]) -> dict:
        """
        Cosine similarities per CLIP variant against in-space prototype banks.

        Args:
            query_features: {variant: [batch, dim]}
            prototypes: either a shared Tensor [num_classes, dim] (legacy) or
                a dict {variant: Tensor[num_classes, dim]} for per-encoder banks
            class_ids: class label order matching the prototype rows

        Returns:
            {variant: [batch, num_classes]} similarity scores
        """
        per_encoder = isinstance(prototypes, dict)
        shared = None if per_encoder else F.normalize(prototypes, p=2, dim=1)

        similarities = {}
        for variant, features in query_features.items():
            features_normalized = F.normalize(features, p=2, dim=1)
            if per_encoder:
                if variant not in prototypes:
                    continue
                bank = F.normalize(prototypes[variant].to(features.device), p=2, dim=1)
            else:
                bank = shared.to(features.device)
                if features_normalized.shape[1] != bank.shape[1]:
                    if features_normalized.shape[1] > bank.shape[1]:
                        features_normalized = features_normalized[:, : bank.shape[1]]
                    else:
                        pad = torch.zeros(
                            features_normalized.shape[0],
                            bank.shape[1] - features_normalized.shape[1],
                            device=features_normalized.device,
                            dtype=features_normalized.dtype,
                        )
                        features_normalized = torch.cat([features_normalized, pad], dim=1)
            similarities[variant] = features_normalized @ bank.T
        return similarities
    
    def ensemble_confidence(self, similarities: dict, temperature: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """
        Compute ensemble confidence from multiple model similarities.
        
        Args:
            similarities: Dict of {variant: [batch_size, num_classes]} similarity scores
            temperature: Temperature for softmax
        
        Returns:
            ensemble_probs: [batch_size, num_classes] ensemble probabilities
            stats: Dict with disagreement metrics and per-variant confidences
        """
        # Weighted average of similarities
        weighted_sim = None
        total_weight = 0.0
        
        for variant, sim in similarities.items():
            weight = self.weights.get(variant, 1.0 / len(similarities))
            if weighted_sim is None:
                weighted_sim = weight * sim
            else:
                weighted_sim += weight * sim
            total_weight += weight
        
        if total_weight > 0:
            weighted_sim = weighted_sim / total_weight
        
        # Apply temperature scaling
        ensemble_logits = weighted_sim / temperature
        ensemble_probs = F.softmax(ensemble_logits, dim=1)
        
        # Compute statistics
        stats = {}
        
        # Per-variant confidences (max probability)
        for variant, sim in similarities.items():
            variant_logits = sim / temperature
            variant_probs = F.softmax(variant_logits, dim=1)
            stats[f'{variant}_confidence'] = variant_probs.max(dim=1)[0]
        
        # Ensemble confidence
        stats['ensemble_confidence'] = ensemble_probs.max(dim=1)[0]
        
        # Disagreement metric (std of max probabilities across variants)
        max_probs = torch.stack([stats[f'{v}_confidence'] for v in similarities.keys()])
        stats['disagreement'] = max_probs.std(dim=0)
        
        # Agreement metric (1 - normalized disagreement)
        stats['agreement'] = 1.0 - (stats['disagreement'] / (stats['ensemble_confidence'] + 1e-8))
        
        return ensemble_probs, stats
    
    def calibrate_confidence(self, similarities: dict, prototypes: torch.Tensor, 
                            class_ids: List[int], temperature: float = 1.0,
                            delta_low: float = 0.55, delta_high: float = 0.85) -> dict:
        """
        Calibrate confidence with ensemble and make tiered decision.
        
        Args:
            similarities: Dict of {variant: [batch_size, num_classes]} similarity scores
            prototypes: [num_classes, 512] prototypes
            class_ids: List of class labels
            temperature: Temperature for softmax
            delta_low: Low confidence threshold
            delta_high: High confidence threshold
        
        Returns:
            Dict with predictions, confidence, tier, and statistics
        """
        batch_size = list(similarities.values())[0].shape[0]
        
        # Compute ensemble confidence
        ensemble_probs, stats = self.ensemble_confidence(similarities, temperature)
        
        # Get predictions
        pred_indices = ensemble_probs.argmax(dim=1)
        predicted_classes = [class_ids[idx] for idx in pred_indices]
        
        # Confidence scores
        confidences = stats['ensemble_confidence']
        disagreements = stats['disagreement']
        agreements = stats['agreement']
        
        # Determine tier based on confidence and agreement
        tiers = []
        for i in range(batch_size):
            conf = confidences[i].item()
            agree = agreements[i].item()
            
            # High confidence: deploy immediately
            if conf >= delta_high and agree > 0.7:
                tiers.append('high')
            # Medium confidence: may need validation
            elif conf >= delta_low:
                # If high agreement, can deploy autonomously
                if agree > 0.8:
                    tiers.append('medium_autonomous')
                else:
                    tiers.append('medium_uncertain')
            # Low confidence: request human feedback
            else:
                tiers.append('low')
        
        return {
            'predictions': predicted_classes,
            'probabilities': ensemble_probs,
            'confidence': confidences,
            'disagreement': disagreements,
            'agreement': agreements,
            'tiers': tiers,
            'stats': stats
        }
    
    def update_weights(self, validation_performance: dict):
        """
        Update ensemble weights based on validation performance.
        
        Args:
            validation_performance: Dict of {variant: accuracy_score}
        """
        total_perf = sum(validation_performance.values())
        if total_perf > 0:
            for variant in self.weights:
                if variant in validation_performance:
                    self.weights[variant] = validation_performance[variant] / total_perf
                else:
                    self.weights[variant] = 1.0 / len(self.weights)
            
            # Normalize
            total_weight = sum(self.weights.values())
            for variant in self.weights:
                self.weights[variant] /= total_weight
            
            print(f"Updated ensemble weights: {self.weights}")
