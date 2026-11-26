# File: evaluate_agent.py
# Enhanced evaluation script for CALM-v2 readiness agent
# Integrates with ensemble calibration and hierarchical memory

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, STL10
from torchvision import transforms
import numpy as np
import argparse
from tqdm import tqdm
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'episodic-memory'))

from ensemble_calibration import EnsembleCLIPModels
from memory import HierarchicalEpisodicMemory
from agent import ReadinessAgent
from calibration import CalibratedEnsembleModel

def parse_args():
    parser = argparse.ArgumentParser(description="CALM-v2 Readiness Agent Evaluation")
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'fashionmnist', 'stl10'],
                        help='Dataset to use for the experiment.')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of examples per class in memory. Paper protocol is 5.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.7, 
                        help='Initial confidence threshold for the agent.')
    parser.add_argument('--adaptive', action='store_true', default=True,
                        help='Use adaptive threshold calibration.')
    parser.add_argument('--calibration_method', type=str, default='temperature',
                        choices=['temperature', 'isotonic', 'platt', 'none'],
                        help='Confidence calibration method to use.')
    parser.add_argument('--target_deployment', type=float, default=0.75,
                        help='Target deployment rate for adaptive thresholding.')
    parser.add_argument('--delta_low', type=float, default=0.55,
                        help='Low confidence threshold for ensemble tiered decisions.')
    parser.add_argument('--delta_high', type=float, default=0.85,
                        help='High confidence threshold for ensemble tiered decisions.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debugging output.')
    return parser.parse_args()

def get_best_device():
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_dataset(name):
    """Load dataset."""
    print(f"Loading {name.upper()} dataset...")
    
    if name == 'cifar10':
        transform = transforms.ToTensor()
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif name == 'cifar100':
        transform = transforms.ToTensor()
        train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif name == 'fashionmnist':
        # FashionMNIST is grayscale, convert to RGB for CLIP
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        train_dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root="./data", train=False, download=True, transform=transform)
    elif name == 'stl10':
        transform = transforms.ToTensor()
        train_dataset = STL10(root="./data", split='train', download=True, transform=transform)
        test_dataset = STL10(root="./data", split='test', download=True, transform=transform)
        # STL-10 uses 'labels' instead of 'targets', set it for compatibility
        train_dataset.targets = train_dataset.labels
        test_dataset.targets = test_dataset.labels
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    print(f"Number of classes: {len(train_dataset.classes)}")
    return train_dataset, test_dataset

def build_memory(train_dataset, ensemble_models, device, k_shot, debug=False):
    """
    Build hierarchical memory with examples from training set.
    """
    print(f"\n=== Building {k_shot}-shot hierarchical episodic memory ===")
    memory = HierarchicalEpisodicMemory(device=device)
    
    # Group samples by class
    class_samples = {}
    for idx, (_, label) in enumerate(train_dataset):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(idx)
    
    # Select diverse examples for each class
    for class_idx in range(len(train_dataset.classes)):
        if class_idx not in class_samples:
            print(f"Warning: No samples found for class {class_idx}")
            continue
            
        indices = class_samples[class_idx]
        
        if len(indices) < k_shot:
            print(f"Warning: Class {class_idx} has only {len(indices)} samples")
            selected_indices = indices
        else:
            selected_indices = np.random.choice(indices, k_shot, replace=False)
        
        # Add selected examples to memory
        for idx in selected_indices:
            image, label = train_dataset[idx]
            pil_image = transforms.ToPILImage()(image)
            
            with torch.inference_mode():
                feats = ensemble_models.get_image_features([pil_image])
                payload = {
                    name: tensor.squeeze(0).cpu() for name, tensor in feats.items()
                }
                memory.add_example(payload, label, importance=1.0)
                del feats
    
    print(f"Memory built with {len(memory)} total examples across {len(memory.get_seen_classes())} classes")
    return memory

def run_evaluation(args):
    """Main evaluation function for CALM-v2 readiness agent."""
    device = get_best_device()
    print(f"Using device: {device}")
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  K-shot: {args.k_shot}")
    print(f"  Initial threshold: {args.threshold:.2%}")
    print(f"  Adaptive: {args.adaptive}")
    print(f"  Calibration: {args.calibration_method}")
    print(f"  Target deployment: {args.target_deployment:.2%}")
    print(f"  Ensemble thresholds: δ_low={args.delta_low}, δ_high={args.delta_high}")

    # 1. Load Data and Models
    print("\n" + "="*50)
    print("STEP 1: Loading ensemble models and data")
    print("="*50)
    
    ensemble_models = EnsembleCLIPModels(device=device)
    train_dataset, test_dataset = load_dataset(args.dataset)

    # 2. Create train/validation/test splits
    val_size = min(2000, len(test_dataset) // 3 * 2)
    test_size = len(test_dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    val_dataset, final_test_dataset = random_split(test_dataset, [val_size, test_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(final_test_dataset, batch_size=32, shuffle=False)
    
    print(f"Data splits: {len(train_dataset)} train, {len(val_dataset)} validation, {len(final_test_dataset)} test")

    # 3. Build Hierarchical Memory
    print("\n" + "="*50)
    print("STEP 2: Building hierarchical episodic memory")
    print("="*50)
    
    memory = build_memory(train_dataset, ensemble_models, device, args.k_shot, args.debug)

    # 4. Model Calibration
    print("\n" + "="*50)
    print("STEP 3: Ensemble model calibration")
    print("="*50)
    
    calibrated_model = CalibratedEnsembleModel(ensemble_models, memory, device)
    
    # Apply calibration
    if args.calibration_method != "none":
        calibrated_model.set_temperature_robust(val_loader, method=args.calibration_method)

    # 5. Agent Calibration
    print("\n" + "="*50)
    print("STEP 4: Agent threshold calibration")
    print("="*50)
    
    agent = ReadinessAgent(
        threshold=args.threshold, 
        adaptive=args.adaptive, 
        target_deployment_rate=args.target_deployment
    )
    
    if args.adaptive:
        # Collect validation data for agent calibration
        val_confidences, val_predictions, val_labels = [], [], []
        val_agreements = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Collecting validation data"):
                pil_images = [transforms.ToPILImage()(img) for img in images]
                logits = calibrated_model.forward(pil_images)
                confidences = calibrated_model.get_calibrated_confidences(logits)
                
                # Get ensemble stats for agreement
                stats = calibrated_model.get_ensemble_stats(pil_images)
                if 'agreement' in stats:
                    val_agreements.extend(stats['agreement'].cpu().numpy())
                
                _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)
                
                # Map predictions back to original class IDs
                class_ids = sorted(memory.get_seen_classes())
                mapped_preds = torch.tensor([class_ids[p] for p in predictions], device=device)
                
                val_confidences.extend(confidences.cpu().numpy())
                val_predictions.extend(mapped_preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        # Calibrate agent threshold
        agent.calibrate_threshold(val_confidences, val_predictions, val_labels, verbose=args.debug)

    # 6. Final Evaluation
    print("\n" + "="*50)
    print("STEP 5: Final evaluation on test set")
    print("="*50)
    
    all_confidences, all_preds, all_labels, all_decisions = [], [], [], []
    all_agreements = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final evaluation"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            logits = calibrated_model.forward(pil_images)
            confidences = calibrated_model.get_calibrated_confidences(logits)
            
            # Get ensemble stats for agreement
            stats = calibrated_model.get_ensemble_stats(pil_images)
            agreements = stats.get('agreement', None)
            if agreements is not None:
                agreements = agreements.cpu().numpy()
            else:
                agreements = [None] * len(confidences)
            
            _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)
            
            # Map predictions back to original class IDs
            class_ids = sorted(memory.get_seen_classes())
            mapped_preds = torch.tensor([class_ids[p] for p in predictions], device=device)

            # Agent decisions (with ensemble agreement if available)
            decisions = []
            for i, conf in enumerate(confidences):
                agreement = agreements[i] if isinstance(agreements, np.ndarray) else None
                decisions.append(agent.decide(conf.item(), agreement=agreement))
            
            all_confidences.extend(confidences.cpu().numpy())
            all_preds.extend(mapped_preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_decisions.extend(decisions)

    # 7. Results Analysis
    print("\n" + "="*50)
    print(f"FINAL RESULTS: {args.dataset.upper()}")
    print("="*50)
    
    metrics = agent.get_metrics(np.array(all_preds), np.array(all_labels), np.array(all_decisions))
    
    # Main results table
    print(f"\n{'Metric':<35} | {'Value':<12}")
    print("-" * 50)
    print(f"{'Dataset':<35} | {args.dataset.upper():<12}")
    print(f"{'Calibration Method':<35} | {args.calibration_method.capitalize():<12}")
    print(f"{'K-shot Memory':<35} | {args.k_shot:<12}")
    print(f"{'Ensemble Models':<35} | {len(ensemble_models.models):<12}")
    print("-" * 50)
    print(f"{'Base Model Accuracy':<35} | {metrics['Total Model Accuracy']:.2%}")
    print(f"{'Agent Threshold (Calibrated)':<35} | {agent.calibrated_threshold:.3f}")
    print(f"{'Deployment Rate':<35} | {metrics['Total Deployed']:.2%}")
    print(f"{'Feedback Request Rate':<35} | {metrics['Total Requested']:.2%}")
    print("-" * 50)
    print(f"{'Deployment Decision Accuracy':<35} | {metrics['Deployment Decision Accuracy']:.2%}")
    print(f"{'False Positive Rate (FPR)':<35} | {metrics['False Positive Rate (FPR)']:.2%}")
    print(f"{'True Negative Rate (TNR)':<35} | {metrics['True Negative Rate (TNR)']:.2%}")
    print(f"{'Deployed Predictions Accuracy':<35} | {metrics['Deployed Accuracy']:.2%}")
    print("-" * 50)
    
    # Detailed analysis if requested
    if args.debug:
        agent.detailed_analysis(all_confidences, all_preds, all_labels, all_decisions)
    
    return metrics

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        run_evaluation(args)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
