# File: calibration.py (CALM-v2 Enhanced Version)
# Calibration wrapper for ensemble CLIP models and hierarchical memory

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Import ensemble calibration and hierarchical memory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'episodic-memory'))
from ensemble_calibration import EnsembleCLIPModels
from memory import HierarchicalEpisodicMemory

class CalibratedEnsembleModel:
    """
    Enhanced calibration wrapper for CALM-v2 ensemble models.
    Works with EnsembleCLIPModels and HierarchicalEpisodicMemory.
    """
    def __init__(self, ensemble_models: EnsembleCLIPModels, 
                 memory: HierarchicalEpisodicMemory, device):
        """
        Initialize calibrated ensemble model.
        
        Args:
            ensemble_models: EnsembleCLIPModels instance
            memory: HierarchicalEpisodicMemory instance
            device: Device to run on
        """
        self.ensemble_models = ensemble_models
        self.memory = memory
        self.device = device
        
        # Initialize temperature to 1 (no change)
        self.temperature = nn.Parameter(torch.ones(1, device=device))
        
        # Alternative calibration methods
        self.isotonic_calibrator = None
        self.platt_calibrator = None
        self.calibration_method = "temperature"  # Default method

    def forward(self, images):
        """
        Calculate logits for a batch of images using ensemble models and memory prototypes.
        
        Args:
            images: List of PIL images or PIL image
        
        Returns:
            logits: [batch_size, num_classes] similarity scores
        """
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]
        
        # Get features from all models in ensemble
        all_features = self.ensemble_models.get_image_features(images)
        
        # Get prototypes from hierarchical memory
        prototypes, class_ids = self.memory.get_prototypes()
        if prototypes.nelement() == 0:
            # No prototypes yet, return zeros
            return torch.zeros(len(images), 1, device=self.device)
        
        prototypes = prototypes.to(self.device)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Create a mapping from original class_id to its index in the prototype tensor
        self.class_id_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        
        # Compute similarities for each variant and ensemble them
        similarities = self.ensemble_models.compute_prototype_similarities(
            all_features, prototypes, class_ids
        )
        
        # Get ensemble confidence (weighted average of similarities)
        ensemble_probs, stats = self.ensemble_models.ensemble_confidence(
            similarities, temperature=1.0
        )
        
        # Convert probabilities to logits (for calibration)
        # Use log probabilities as logits
        logits = torch.log(ensemble_probs + 1e-8)
        
        return logits

    def get_calibrated_confidences(self, logits):
        """Get calibrated confidence scores based on the selected method."""
        if self.calibration_method == "temperature":
            scaled_logits = logits / self.temperature
            probabilities = torch.softmax(scaled_logits, dim=1)
            confidences, _ = torch.max(probabilities, dim=1)
            return confidences
        
        elif self.calibration_method == "isotonic":
            probabilities = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            max_probs_np = max_probs.detach().cpu().numpy().astype(np.float64)
            if self.isotonic_calibrator is not None:
                calibrated_confidences = self.isotonic_calibrator.predict(max_probs_np)
                return torch.tensor(calibrated_confidences, device=self.device, dtype=torch.float32)
            return max_probs
        
        elif self.calibration_method == "platt":
            probabilities = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            max_probs_np = max_probs.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
            if self.platt_calibrator is not None:
                calibrated_probs = self.platt_calibrator.predict_proba(max_probs_np)[:, 1]
                return torch.tensor(calibrated_probs, device=self.device, dtype=torch.float32)
            return max_probs
        
        else:  # No calibration
            probabilities = torch.softmax(logits, dim=1)
            confidences, _ = torch.max(probabilities, dim=1)
            return confidences

    def get_ensemble_stats(self, images):
        """
        Get ensemble statistics including agreement and disagreement.
        
        Args:
            images: List of PIL images
        
        Returns:
            dict with ensemble statistics
        """
        if not isinstance(images, list):
            images = [images]
        
        prototypes, class_ids = self.memory.get_prototypes()
        if prototypes.nelement() == 0:
            return {}
        
        prototypes = prototypes.to(self.device)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        all_features = self.ensemble_models.get_image_features(images)
        similarities = self.ensemble_models.compute_prototype_similarities(
            all_features, prototypes, class_ids
        )
        
        ensemble_probs, stats = self.ensemble_models.ensemble_confidence(
            similarities, temperature=self.temperature.item()
        )
        
        return stats

    def temperature_scale(self, logits):
        """Applies temperature scaling to logits."""
        return logits / self.temperature

    def set_temperature_robust(self, validation_loader, method="temperature"):
        """
        Enhanced temperature setting with multiple methods.
        """
        self.calibration_method = method
        self.ensemble_models.models[list(self.ensemble_models.models.keys())[0]].eval()
        
        all_logits = []
        all_labels = []
        all_correct = []

        # Collect all logits and labels from the validation set
        print(f"Collecting logits from validation set for {method} calibration...")
        with torch.no_grad():
            for images, labels in validation_loader:
                pil_images = [transforms.ToPILImage()(img) for img in images]
                logits = self.forward(pil_images)
                
                # Get predictions for accuracy calculation
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(probs, dim=1)
                class_ids = sorted(self.memory.get_seen_classes())
                mapped_preds = torch.tensor([class_ids[p] for p in preds], device=self.device)
                correct = (mapped_preds.cpu() == labels).float()
                
                all_logits.append(logits)
                all_labels.append(labels)
                all_correct.append(correct)
        
        all_logits = torch.cat(all_logits).to(self.device)
        all_labels = torch.cat(all_labels).to(self.device)
        all_correct = torch.cat(all_correct).to(self.device)
        
        # Map global labels to the local indices of the prototypes
        mapped_labels = torch.tensor(
            [self.class_id_to_idx[lbl.item()] for lbl in all_labels], 
            device=self.device
        )

        if method == "temperature":
            self._calibrate_temperature(all_logits, mapped_labels)
        elif method == "isotonic":
            self._calibrate_isotonic(all_logits, all_correct)
        elif method == "platt":
            self._calibrate_platt(all_logits, all_correct)
        
        return self

    def _calibrate_temperature(self, all_logits, mapped_labels):
        """Calibrate using temperature scaling."""
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        
        initial_loss = nll_criterion(all_logits, mapped_labels)
        print(f"Initial NLL loss: {initial_loss.item():.4f}")
        
        best_temperature = 1.0
        best_loss = float('inf')
        
        for lr in [0.01, 0.001, 0.1]:
            for max_iter in [50, 100, 200]:
                self.temperature.data.fill_(1.0)
                optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

                def eval_fn():
                    optimizer.zero_grad()
                    loss = nll_criterion(self.temperature_scale(all_logits), mapped_labels)
                    loss.backward()
                    return loss

                try:
                    optimizer.step(eval_fn)
                    final_loss = nll_criterion(self.temperature_scale(all_logits), mapped_labels)
                    
                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_temperature = self.temperature.item()
                except:
                    continue
        
        self.temperature.data.fill_(best_temperature)
        print(f"Best temperature found: {best_temperature:.4f}")
        print(f"Final NLL loss: {best_loss:.4f}")

    def _calibrate_isotonic(self, all_logits, all_correct):
        """Calibrate using Isotonic Regression."""
        probs = torch.softmax(all_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        
        max_probs_np = max_probs.detach().cpu().numpy().astype(np.float64)
        correct_np = all_correct.cpu().numpy().astype(np.float64)
        
        self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_calibrator.fit(max_probs_np, correct_np)
        
        print("Isotonic calibration completed.")

    def _calibrate_platt(self, all_logits, all_correct):
        """Calibrate using Platt Scaling (Logistic Regression)."""
        probs = torch.softmax(all_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        
        max_probs_np = max_probs.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
        correct_np = all_correct.cpu().numpy().astype(np.int32)
        
        self.platt_calibrator = LogisticRegression()
        self.platt_calibrator.fit(max_probs_np, correct_np)
        
        print("Platt scaling calibration completed.")

    # Backward compatibility
    def set_temperature(self, validation_loader):
        """Backward compatibility wrapper."""
        return self.set_temperature_robust(validation_loader, method="temperature")
