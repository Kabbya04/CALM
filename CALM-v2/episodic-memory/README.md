# CALM-v2: Enhanced Continual Adaptive Learning Model

This is an enhanced version of the CALM framework implementing two key improvements:

1. **Approach 2.2: Ensemble Confidence Calibration with Multiple CLIP Variants**
   - Uses multiple CLIP models (ViT-B/32, ViT-B/16) for ensemble confidence estimation
   - Detects model disagreement to improve confidence calibration
   - Adaptive threshold adjustment based on ensemble agreement

2. **Approach 2.4: DeepSeek OCR-Inspired Episodic Memory Compression**
   - Hierarchical memory architecture with three tiers (recent/medium/old)
   - Learned embedding compression networks (autoencoders)
   - Prototype consolidation for old tasks
   - Multi-resolution memory replay

## Features

- **Automatic Device Detection**: Automatically selects the best available device (CUDA > MPS > CPU)
- **Ensemble Calibration**: Multiple CLIP variants provide robust confidence estimation
- **Memory Compression**: Hierarchical compression enables 2.5× more storage capacity
- **Tiered Decision Making**: Three-tier system (high/medium/low confidence) for autonomous deployment
- **Multi-Dataset Support**: Ready-to-use configurations for CIFAR-10, CIFAR-100, FashionMNIST, and STL-10

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Supported Datasets

The framework supports the following datasets:
- **CIFAR-10**: 10 classes, default 5 tasks (2 classes per task)
- **CIFAR-100**: 100 classes, default 10 tasks (10 classes per task)
- **FashionMNIST**: 10 classes, default 5 tasks (2 classes per task)
- **STL-10**: 10 classes, default 5 tasks (2 classes per task)

### Basic Usage

Run on CIFAR-10 with default settings:
```bash
python run_continual_learning.py --dataset cifar10
```

Run on CIFAR-100:
```bash
python run_continual_learning.py --dataset cifar100
```

Run on FashionMNIST:
```bash
python run_continual_learning.py --dataset fashionmnist
```

Run on STL-10:
```bash
python run_continual_learning.py --dataset stl10
```

### Advanced Options

Train compression networks before running:
```bash
python run_continual_learning.py --dataset cifar10 --train_compression --compression_epochs 50
```

Custom confidence thresholds:
```bash
python run_continual_learning.py --dataset cifar10 --delta_low 0.55 --delta_high 0.85
```

Custom number of tasks:
```bash
python run_continual_learning.py --dataset cifar10 --num_tasks 5
```

### Dataset-Specific Examples

**CIFAR-10 (5 tasks, 2 classes per task):**
```bash
python run_continual_learning.py --dataset cifar10 --k_shot 5
```

**CIFAR-100 (10 tasks, 10 classes per task):**
```bash
python run_continual_learning.py --dataset cifar100 --k_shot 5
```

**FashionMNIST (5 tasks, 2 classes per task):**
```bash
python run_continual_learning.py --dataset fashionmnist --k_shot 5
```

**STL-10 (5 tasks, 2 classes per task):**
```bash
python run_continual_learning.py --dataset stl10 --k_shot 5
```

### Full Example

```bash
python run_continual_learning.py \
    --dataset cifar10 \
    --num_tasks 5 \
    --k_shot 5 \
    --train_compression \
    --compression_epochs 50 \
    --delta_low 0.55 \
    --delta_high 0.85 \
    --temperature 1.0
```

## Architecture

### Ensemble Calibration
- **Multiple CLIP Variants**: ViT-B/32 (base), ViT-B/16 (higher resolution)
- **Weighted Ensemble**: Combines predictions from multiple models
- **Disagreement Detection**: Flags uncertain predictions when models disagree
- **Adaptive Thresholds**: Adjusts confidence thresholds based on ensemble agreement

### Hierarchical Memory
- **Recent Tier** (0-100 episodes): Full 512-dim embeddings, no compression
- **Medium Tier** (100-500 episodes): 2× compression (512 → 256 dim)
- **Old Tier** (>500 episodes): 4× compression (512 → 128 dim) or prototypes

### Compression Networks
- Autoencoder architecture: 512 → 256 → compressed_dim → 256 → 512
- Trained with reconstruction loss to minimize information loss
- Separate networks for each compression tier

## Output Metrics

The framework reports:
- **Average Accuracy**: Mean accuracy across all tasks
- **Backward Transfer (BWT)**: Forgetting measure (lower is better)
- **Autonomy Rate**: Percentage of predictions deployed autonomously
- **Human Feedback Rate**: Percentage requiring human intervention
- **Tier Statistics**: Breakdown of predictions by confidence tier

## Expected Improvements

Based on the alternatives discussion report:
- **Human Feedback Reduction**: 15.14% → 8-10% (35-40% reduction)
- **Memory Capacity**: 2.5× increase through compression
- **Confidence Calibration**: Better handling of edge cases through ensemble
- **Backward Transfer**: Improved from -5.52% to target -3.0% to -2.0%

## File Structure

```
CALM-v2/
├── run_continual_learning.py    # Main experiment script
├── memory.py                     # Hierarchical episodic memory with compression
├── ensemble_calibration.py       # Ensemble CLIP models and calibration
├── dataset_manager.py            # Multi-dataset handling (CIFAR-10/100, FashionMNIST, STL-10)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Notes

- The framework automatically detects and uses the best available device
- Compression networks can be trained on-the-fly or pre-trained
- Memory migration and consolidation happen automatically during training
- Ensemble models are loaded sequentially to manage memory usage

## References

- CALM Framework: Original continual learning framework
- DeepSeek OCR: Inspiration for hierarchical compression
- CLIP: Vision-language model for feature extraction
