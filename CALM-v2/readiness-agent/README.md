# CALM-v2 Readiness Agent

This is the readiness agent component for the CALM-v2 enhanced framework. It integrates with:

1. **Ensemble CLIP Models**: Uses multiple CLIP variants (ViT-B/32, ViT-B/16) for robust confidence estimation
2. **Hierarchical Episodic Memory**: Works with the enhanced hierarchical memory with compression
3. **Ensemble Calibration**: Leverages ensemble agreement for better confidence calibration

## Features

- **Ensemble-Based Confidence**: Uses ensemble agreement to improve confidence estimation
- **Tiered Decision Making**: Three-tier system (high/medium/low confidence) with ensemble agreement
- **Adaptive Threshold Calibration**: Automatically calibrates thresholds based on validation data
- **Multiple Calibration Methods**: Supports temperature scaling, isotonic regression, and Platt scaling
- **Hierarchical Memory Integration**: Works seamlessly with compressed hierarchical memory

## Architecture

The readiness agent makes deployment decisions based on:

1. **Ensemble Confidence**: Weighted average of confidences from multiple CLIP models
2. **Ensemble Agreement**: Measures how much models agree (disagreement = uncertainty)
3. **Calibrated Thresholds**: Thresholds calibrated on validation data
4. **Tiered Logic**:
   - High confidence (≥0.85) + High agreement (>0.7) → Deploy
   - Medium confidence (≥0.55) + High agreement (>0.8) → Deploy
   - Otherwise → Request Feedback

## Usage

### Supported Datasets

The readiness agent supports the following datasets:
- **CIFAR-10**: 10 classes
- **CIFAR-100**: 100 classes
- **FashionMNIST**: 10 classes (automatically converted from grayscale to RGB)
- **STL-10**: 10 classes

### Basic Evaluation

**CIFAR-10:**
```bash
python evaluate_agent.py --dataset cifar10 --k_shot 10
```

**CIFAR-100:**
```bash
python evaluate_agent.py --dataset cifar100 --k_shot 10
```

**FashionMNIST:**
```bash
python evaluate_agent.py --dataset fashionmnist --k_shot 10
```

**STL-10:**
```bash
python evaluate_agent.py --dataset stl10 --k_shot 10
```

### With Custom Thresholds

```bash
python evaluate_agent.py --dataset cifar10 --k_shot 10 --threshold 0.75 --target_deployment 0.8
```

### With Different Calibration Methods

**Temperature scaling (default):**
```bash
python evaluate_agent.py --dataset cifar10 --calibration_method temperature
```

**Isotonic regression:**
```bash
python evaluate_agent.py --dataset cifar10 --calibration_method isotonic
```

**Platt scaling:**
```bash
python evaluate_agent.py --dataset cifar10 --calibration_method platt
```

### Dataset-Specific Examples

**CIFAR-10 with full options:**
```bash
python evaluate_agent.py \
    --dataset cifar10 \
    --k_shot 10 \
    --threshold 0.7 \
    --adaptive \
    --calibration_method temperature \
    --target_deployment 0.75 \
    --delta_low 0.55 \
    --delta_high 0.85 \
    --debug
```

**FashionMNIST evaluation:**
```bash
python evaluate_agent.py \
    --dataset fashionmnist \
    --k_shot 10 \
    --threshold 0.7 \
    --adaptive \
    --calibration_method temperature
```

**STL-10 evaluation:**
```bash
python evaluate_agent.py \
    --dataset stl10 \
    --k_shot 10 \
    --threshold 0.7 \
    --adaptive \
    --calibration_method isotonic
```

### Full Example

```bash
python evaluate_agent.py \
    --dataset cifar10 \
    --k_shot 10 \
    --threshold 0.7 \
    --adaptive \
    --calibration_method temperature \
    --target_deployment 0.75 \
    --delta_low 0.55 \
    --delta_high 0.85 \
    --debug
```

## Components

### `agent.py`
- `ReadinessAgent`: Main agent class with adaptive thresholding
- Supports ensemble agreement in decision making
- Comprehensive metrics calculation

### `calibration.py`
- `CalibratedEnsembleModel`: Wrapper for ensemble models with calibration
- Integrates with `EnsembleCLIPModels` and `HierarchicalEpisodicMemory`
- Multiple calibration methods (temperature, isotonic, Platt)

### `memory.py`
- Re-exports `HierarchicalEpisodicMemory` from episodic-memory module
- Provides convenient access to hierarchical memory

### `evaluate_agent.py`
- Main evaluation script
- Integrates all components for end-to-end evaluation
- Comprehensive metrics and analysis

## Expected Performance

Based on CALM-v2 report:

| Metric | Target | Notes |
|--------|--------|-------|
| Deployment Rate | 90-92% | Autonomous deployment |
| Human Feedback Rate | 8-10% | Reduced from 15.14% |
| False Positive Rate | <10% | Safety metric |
| Deployed Accuracy | ≥99% | Accuracy of deployed predictions |

## Integration with CALM-v2

The readiness agent is designed to work with:

1. **Episodic Memory Module**: Uses `HierarchicalEpisodicMemory` from `../episodic-memory/`
2. **Ensemble Calibration**: Uses `EnsembleCLIPModels` from `../episodic-memory/`
3. **Hierarchical Compression**: Automatically handles compressed embeddings

## File Structure

```
readiness-agent/
├── agent.py              # ReadinessAgent class
├── calibration.py        # CalibratedEnsembleModel wrapper
├── memory.py             # Memory import wrapper
├── evaluate_agent.py     # Main evaluation script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Notes

- The agent automatically detects and uses the best available device (CUDA > MPS > CPU)
- Ensemble models are loaded sequentially to manage memory
- Hierarchical memory compression is handled transparently
- Calibration can be performed on validation set or skipped for faster evaluation
