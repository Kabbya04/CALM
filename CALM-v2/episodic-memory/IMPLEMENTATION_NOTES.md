# Implementation Notes: CALM-v2 Enhanced Framework

## Overview

This implementation enhances the original CALM framework with two key approaches from the alternatives discussion report:

1. **Approach 2.2: Ensemble Confidence Calibration with Multiple CLIP Variants**
2. **Approach 2.4: DeepSeek OCR-Inspired Episodic Memory Compression**

## Key Differences from Original CALM

### 1. Ensemble Calibration (`ensemble_calibration.py`)

**Original**: Single CLIP ViT-B/32 model for feature extraction

**Enhanced**: 
- Multiple CLIP variants (ViT-B/32, ViT-B/16) for ensemble
- Weighted combination of predictions from multiple models
- Disagreement detection to identify uncertain predictions
- Adaptive confidence thresholds based on ensemble agreement

**Benefits**:
- More robust confidence estimation through model diversity
- Better handling of edge cases where models disagree
- Reduced false low-confidence predictions

### 2. Hierarchical Memory (`memory.py`)

**Original**: Simple episodic memory storing full 512-dim embeddings uniformly

**Enhanced**:
- Three-tier hierarchical architecture:
  - **Recent** (0-100 episodes): Full 512-dim, no compression
  - **Medium** (100-500 episodes): 2× compression (512 → 256 dim)
  - **Old** (>500 episodes): 4× compression (512 → 128 dim) or prototypes
- Learned compression networks (autoencoders) for each tier
- Automatic memory migration based on age
- Prototype consolidation for very old tasks

**Benefits**:
- 2.5× storage capacity increase
- Better memory utilization
- Progressive compression mimics biological forgetting
- Maintains accuracy while reducing storage

### 3. Device Detection (`run_continual_learning.py`)

**Original**: Manual device specification

**Enhanced**:
- Automatic device detection (CUDA > MPS > CPU)
- Intelligent fallback to best available device
- Clear device status reporting

### 4. Tiered Decision Making

**Original**: Binary decision (deploy vs. human feedback)

**Enhanced**:
- Three-tier system:
  - **High confidence** (≥δ_high): Deploy immediately
  - **Medium confidence** (δ_low ≤ conf < δ_high): 
    - High agreement → Deploy autonomously
    - Low agreement → Request human feedback
  - **Low confidence** (<δ_low): Request human feedback

**Benefits**:
- More nuanced decision making
- Better autonomy rate
- Reduced human dependency

## Implementation Details

### Compression Networks

The compression networks use a simple autoencoder architecture:
- Encoder: 512 → 256 → compressed_dim
- Decoder: compressed_dim → 256 → 512
- Training: Reconstruction loss (MSE)
- Separate networks for medium (256-dim) and old (128-dim) tiers

### Ensemble Weights

Currently uses uniform weights (1/N for N models). Can be updated based on validation performance using `update_weights()` method.

### Memory Migration

Memory automatically migrates examples between tiers:
- Triggered every 50 episodes
- Examples move from recent → medium → old based on age
- Compression applied during migration

### Prototype Consolidation

Very old examples (>500 episodes) are consolidated:
- K-means clustering within each class
- Weighted prototype computation
- Replaces multiple examples with single prototype
- Maintains class representation with reduced storage

## Usage Examples

### Basic Usage
```bash
python run_continual_learning.py --dataset cifar10
```

### With Compression Training
```bash
python run_continual_learning.py --dataset cifar10 --train_compression --compression_epochs 50
```

### Custom Thresholds
```bash
python run_continual_learning.py --dataset cifar10 --delta_low 0.55 --delta_high 0.85
```

## Expected Performance

Based on the alternatives discussion report:

| Metric | Original | Target | Implementation |
|--------|----------|--------|----------------|
| Human Feedback Rate | 15.14% | 8-10% | ~10-12% (with ensemble + compression) |
| Autonomy Rate | 84.86% | 90-92% | ~88-90% |
| Memory Capacity | 1× | 2.5× | 2.5× (through compression) |
| BWT | -5.52% | -3.0% to -2.0% | Improved (better memory retention) |

## Limitations and Future Improvements

1. **Compression Network Training**: Currently trains on sample embeddings. Could be improved with task-specific training.

2. **Ensemble Dimension Handling**: Simple truncation/padding for different CLIP variants. Could use learned projection layers.

3. **Prototype Consolidation**: Uses simple weighted mean. Could use more sophisticated methods (e.g., mixture of prototypes).

4. **Memory Migration Frequency**: Fixed intervals. Could be adaptive based on memory pressure.

5. **Ensemble Weights**: Currently uniform. Could be learned from validation data.

## Testing

The framework is configured to test on:
- **CIFAR-10**: 5 tasks of 2 classes each (default)
- **CIFAR-100**: 10 tasks of 10 classes each (default)

Both datasets are automatically downloaded on first run.

## File Structure

```
CALM-v2/
├── run_continual_learning.py    # Main experiment orchestrator
├── memory.py                     # Hierarchical memory with compression
├── ensemble_calibration.py       # Ensemble CLIP models
├── dataset_manager.py            # CIFAR dataset handling
├── requirements.txt              # Dependencies
├── README.md                     # User guide
└── IMPLEMENTATION_NOTES.md       # This file
```

## References

- Original CALM Framework: `SEM11-DEEP-LEARNING-PROJECT/CALM-Continual_Adaptive_Learning_Model/`
- Alternatives Discussion: `CALM_Alternatives_Discussion_Report.md`
- SAM3-DeepSeek Integration: `CALM-SAM3-DeepSeek OCR.md`
