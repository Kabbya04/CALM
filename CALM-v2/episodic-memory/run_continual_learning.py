# File: run_continual_learning.py
# Enhanced CALM Framework with Ensemble Calibration and Memory Compression
# Implements:
# - Approach 2.2: Ensemble Confidence Calibration with Multiple CLIP Variants
# - Approach 2.4: DeepSeek OCR-Inspired Episodic Memory Compression

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import os

# Import our enhanced components
from memory import HierarchicalEpisodicMemory
from dataset_manager import DatasetManager
from ensemble_calibration import EnsembleCLIPModels

# --- 1. Setup & Helper Functions ---

def get_best_device():
    """
    Automatically detect and return the best available device.
    Priority: CUDA > MPS (Mac) > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        return device
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("✓ MPS (Metal Performance Shaders) available")
        return device
    
    device = "cpu"
    print("⚠ Using CPU (CUDA and MPS not available)")
    return device

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced CALM Framework with Ensemble Calibration and Memory Compression")
    
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar100', 'cifar10', 'fashionmnist', 'stl10'],
                        help='Dataset to use for the experiment.')
                        
    parser.add_argument('--num_tasks', type=int, default=None,
                        help='Number of sequential tasks. If not set, a smart default will be used.')
    
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of shots per class for few-shot learning.')
    
    parser.add_argument('--train_compression', action='store_true',
                        help='Train compression networks before running experiments.')
    
    parser.add_argument('--compression_epochs', type=int, default=50,
                        help='Number of epochs to train compression networks.')
    
    parser.add_argument('--delta_low', type=float, default=0.55,
                        help='Low confidence threshold for tiered decision making.')
    
    parser.add_argument('--delta_high', type=float, default=0.85,
                        help='High confidence threshold for tiered decision making.')
    
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softmax confidence calibration.')
    
    return parser.parse_args()

# --- 2. Evaluation Logic Functions ---

def evaluate_zero_shot_continual(ensemble_models, data_manager, device):
    """Evaluates Zero-Shot performance across all sequential tasks using ensemble."""
    ensemble_models.models[list(ensemble_models.models.keys())[0]].eval()
    task_accuracies = []
    
    # Using all class names for zero-shot evaluation
    all_class_names = data_manager.class_names
    text_prompts = [f"a photo of a {c}" for c in all_class_names]
    
    # Use first model's processor for text (they're compatible)
    first_variant = list(ensemble_models.models.keys())[0]
    processor = ensemble_models.processors[first_variant]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    for task_id in range(data_manager.num_tasks):
        test_loader = DataLoader(data_manager.get_task_datasets('test', task_id), batch_size=128)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                pil_images = [transforms.ToPILImage()(img) for img in images]
                
                # Get features from all models
                all_features = ensemble_models.get_image_features(pil_images)
                
                # Use first model for text-image matching (simplified)
                first_model = ensemble_models.models[first_variant]
                image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
                
                outputs = first_model(**image_inputs, **text_inputs)
                logits_per_image = outputs.logits_per_image
                
                preds_global = torch.argmax(logits_per_image, dim=1)
                
                correct += (preds_global == labels.to(device)).sum().item()
                total += len(labels)
        
        task_acc = (correct / total) * 100 if total > 0 else 0
        task_accuracies.append(task_acc)
        print(f"  Zero-Shot Accuracy on Task {task_id+1}: {task_acc:.2f}%")
        
    return np.mean(task_accuracies)

def run_continual_task_sequence(ensemble_models, data_manager, k_shot, device, 
                                delta_low=0.55, delta_high=0.85, temperature=1.0,
                                train_compression=False, compression_epochs=50):
    """
    Runs a full continual learning sequence with enhanced memory and ensemble calibration.
    """
    # Initialize hierarchical memory
    memory = HierarchicalEpisodicMemory(device=device)
    
    # Train compression networks if requested
    if train_compression:
        print("\n--- Training Compression Networks ---")
        # Collect sample embeddings for training
        sample_embeddings = []
        sample_count = 0
        max_samples = 1000  # Limit samples for training
        
        for task_id in range(min(2, data_manager.num_tasks)):  # Use first 2 tasks
            train_subset = data_manager.get_task_datasets('train', task_id)
            for idx in range(min(100, len(train_subset))):  # Sample from each task
                image, label = train_subset[idx]
                pil_image = transforms.ToPILImage()(image)
                
                # Get embedding from first model
                first_variant = list(ensemble_models.models.keys())[0]
                features = ensemble_models.get_image_features(pil_image, variant=first_variant)
                sample_embeddings.append(features.squeeze().cpu())
                sample_count += 1
                
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
        
        if sample_embeddings:
            sample_tensor = torch.stack(sample_embeddings)
            memory.train_compression_networks(sample_tensor, epochs=compression_epochs)
    
    num_tasks = data_manager.num_tasks
    results_matrix = np.zeros((num_tasks, num_tasks))
    
    # Track statistics
    tier_counts = {'high': 0, 'medium_autonomous': 0, 'medium_uncertain': 0, 'low': 0}
    human_feedback_count = 0
    total_predictions = 0

    for task_id in range(num_tasks):
        class_range = data_manager.get_class_range_for_task(task_id)
        print(f"\n--- Learning Task {task_id+1}/{num_tasks} (Classes {class_range.start}-{class_range.stop-1}) ---")
        
        train_subset = data_manager.get_task_datasets('train', task_id)
        # Handle STL-10 which uses 'labels' instead of 'targets'
        if data_manager.dataset_name == 'stl10':
            original_targets = np.array(data_manager.train_dataset.labels)[train_subset.indices]
        else:
            original_targets = np.array(data_manager.train_dataset.targets)[train_subset.indices]
        
        # Add examples to memory
        for class_id in class_range:
            class_indices = np.where(original_targets == class_id)[0]
            if len(class_indices) < k_shot:
                print(f"Warning: Not enough samples for class {class_id}. Using {len(class_indices)} samples.")
                support_indices = class_indices
            else:
                support_indices = np.random.choice(class_indices, k_shot, replace=False)

            for idx in support_indices:
                image, label = train_subset[idx]
                pil_image = transforms.ToPILImage()(image)
                
                # Get embedding from first model (for memory storage)
                first_variant = list(ensemble_models.models.keys())[0]
                with torch.no_grad():
                    embedding = ensemble_models.get_image_features(pil_image, variant=first_variant).squeeze()
                
                memory.add_example(embedding, label, importance=1.0, task_id=task_id)
        
        # Increment episode after adding examples
        memory.increment_episode()
        
        # Evaluate on all seen tasks
        for j in range(task_id + 1):
            test_loader = DataLoader(data_manager.get_task_datasets('test', j), batch_size=128)
            acc, stats = evaluate_on_memory_ensemble(memory, ensemble_models, test_loader, device,
                                                     delta_low, delta_high, temperature)
            results_matrix[task_id, j] = acc
            
            # Update statistics
            if stats:
                for tier in stats.get('tier_counts', {}):
                    tier_counts[tier] += stats['tier_counts'][tier]
                human_feedback_count += stats.get('human_feedback', 0)
                total_predictions += stats.get('total_predictions', 0)
    
    # Calculate metrics
    avg_accuracy = results_matrix.diagonal().mean()
    bwt = 0.0
    if num_tasks > 1:
        for i in range(1, num_tasks):
            bwt += results_matrix[num_tasks-1, i-1] - results_matrix[i-1, i-1]
        bwt /= (num_tasks - 1)
    
    # Calculate human feedback rate
    human_feedback_rate = (human_feedback_count / total_predictions * 100) if total_predictions > 0 else 0.0
    autonomy_rate = 100.0 - human_feedback_rate
    
    return {
        "avg_acc": avg_accuracy,
        "bwt": bwt,
        "human_feedback_rate": human_feedback_rate,
        "autonomy_rate": autonomy_rate,
        "tier_counts": tier_counts
    }

def evaluate_on_memory_ensemble(memory, ensemble_models, test_loader, device,
                                delta_low=0.55, delta_high=0.85, temperature=1.0):
    """
    Evaluate using ensemble models and hierarchical memory with tiered decision making.
    """
    # Set models to eval mode
    for model in ensemble_models.models.values():
        model.eval()
    
    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0:
        return 0.0, {}
    
    prototypes = prototypes.to(device)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    correct = 0
    total = 0
    tier_counts = {'high': 0, 'medium_autonomous': 0, 'medium_uncertain': 0, 'low': 0}
    human_feedback = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            pil_images = [transforms.ToPILImage()(img) for img in images]
            
            # Get features from all models
            all_features = ensemble_models.get_image_features(pil_images)
            
            # Compute similarities for each variant
            similarities = ensemble_models.compute_prototype_similarities(
                all_features, prototypes, class_ids
            )
            
            # Calibrate confidence with ensemble
            results = ensemble_models.calibrate_confidence(
                similarities, prototypes, class_ids, temperature,
                delta_low, delta_high
            )
            
            # Get predictions
            predicted_classes = results['predictions']
            tiers = results['tiers']
            
            # Count tiers
            for tier in tiers:
                tier_counts[tier] += 1
                if tier in ['medium_uncertain', 'low']:
                    human_feedback += 1
            
            # Evaluate accuracy (assuming autonomous deployment for high/medium_autonomous)
            for i, (pred_class, true_label, tier) in enumerate(zip(predicted_classes, labels, tiers)):
                total += 1
                # Count accuracy for all predictions (for evaluation purposes)
                # In real deployment, only high/medium_autonomous would be deployed
                if pred_class == true_label.item():
                    correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    stats = {
        'tier_counts': tier_counts,
        'human_feedback': human_feedback,
        'total_predictions': total
    }
    
    return accuracy, stats

# --- 3. Main Orchestrator ---

if __name__ == "__main__":
    args = parse_args()
    device = get_best_device()
    
    print(f"\n{'='*70}")
    print(f"Enhanced CALM Framework - Ensemble Calibration + Memory Compression")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Temperature: {args.temperature}")
    print(f"Confidence Thresholds: δ_low={args.delta_low}, δ_high={args.delta_high}")
    print(f"{'='*70}\n")
    
    # Initialize dataset manager
    data_manager = DatasetManager(dataset_name=args.dataset, num_tasks=args.num_tasks, k_shot=args.k_shot)
    
    # Initialize ensemble CLIP models
    print("\n--- Initializing Ensemble CLIP Models ---")
    ensemble_models = EnsembleCLIPModels(device=device)
    
    print(f"\n{'='*70}\nStarting Continual Learning Experiment\n{'='*70}")
    
    # Phase 1: Zero-shot baseline
    print("\n--- Phase 1: Evaluating Zero-Shot Baseline ---")
    zero_shot_avg_acc = evaluate_zero_shot_continual(ensemble_models, data_manager, device)
    
    # Phase 2: 1-shot with enhanced framework
    print("\n\n--- Phase 2: Running 1-Shot Enhanced Framework ---")
    one_shot_results = run_continual_task_sequence(
        ensemble_models, data_manager, k_shot=1, device=device,
        delta_low=args.delta_low, delta_high=args.delta_high, temperature=args.temperature,
        train_compression=args.train_compression, compression_epochs=args.compression_epochs
    )
    
    # Phase 3: 5-shot with enhanced framework
    print("\n\n--- Phase 3: Running 5-Shot Enhanced Framework ---")
    five_shot_results = run_continual_task_sequence(
        ensemble_models, data_manager, k_shot=5, device=device,
        delta_low=args.delta_low, delta_high=args.delta_high, temperature=args.temperature,
        train_compression=args.train_compression, compression_epochs=args.compression_epochs
    )
    
    # Print results
    print(f"\n\n{'='*70}")
    print(f"    Final Results on {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"{'Method':<30} | {'Avg. Accuracy (%)':<20} | {'BWT (%)':<15} | {'Autonomy (%)':<15} | {'Human Feedback (%)':<20}")
    print(f"-"*100)
    print(f"{'Zero-Shot Baseline':<30} | {zero_shot_avg_acc:<20.2f} | {'N/A':<15} | {'N/A':<15} | {'N/A':<20}")
    print(f"{'1-Shot Enhanced':<30} | {one_shot_results['avg_acc']:<20.2f} | {one_shot_results['bwt']:<15.2f} | {one_shot_results['autonomy_rate']:<15.2f} | {one_shot_results['human_feedback_rate']:<20.2f}")
    print(f"{'5-Shot Enhanced':<30} | {five_shot_results['avg_acc']:<20.2f} | {five_shot_results['bwt']:<15.2f} | {five_shot_results['autonomy_rate']:<15.2f} | {five_shot_results['human_feedback_rate']:<20.2f}")
    print(f"-"*100)
    
    # Print tier statistics
    print(f"\n--- Tier Statistics (5-Shot) ---")
    for tier, count in five_shot_results['tier_counts'].items():
        percentage = (count / sum(five_shot_results['tier_counts'].values()) * 100) if sum(five_shot_results['tier_counts'].values()) > 0 else 0
        print(f"  {tier}: {count} ({percentage:.2f}%)")
    
    print(f"\n{'='*70}")
