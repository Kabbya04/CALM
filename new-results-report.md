# CALM Final Evaluation: New Results Report

This report records the results produced by the CALM-v1 and CALM-v2
multi-dataset final evaluation runners. It is designed to be extended as
results for both frameworks and the remaining datasets become available.

## 1. Evaluation implementation

The experiment uses the CALM-v1 final evaluation runner:

```text
CALM-v1/CALM-Continual_Adaptive_Learning_Model/final-evaluation/final-eval-other/final-others.py
```

The evaluated framework consists of:

- Frozen CLIP ViT-B/32 image encoder.
- Flat episodic memory.
- Class prototypes computed as the mean of stored support embeddings.
- Cosine-similarity prototype classification.
- No gradient-based fine-tuning during evaluation.

The runner performs two evaluations for each selected dataset:

1. Few-shot episodic evaluation.
2. Sequential continual learning evaluation.

## 2. Current completed run: CIFAR-10

### 2.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v1 |
| Dataset | CIFAR-10 |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| PyTorch | CUDA-enabled build |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 5 |
| Classes per continual task | 2 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The test split is used for both support sampling and evaluation, following the
same evaluation convention as the ImageNet runner, which uses the ImageNet
validation split. Within each few-shot episode, support and query images are
sampled separately.

### 2.2 Reproduction command

Run from:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation\final-eval-other
```

Command:

```powershell
python final-others.py --datasetname cifar10 --device cuda
```

## 3. Few-shot results

### 3.1 Aggregate result

| Metric | CALM-v1 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Mean accuracy | **85.74%** |
| 95% confidence interval | **$\pm$ 0.49 percentage points** |

The reported confidence interval is computed across episode accuracies:

```text
95% CI = 1.96 × episode-accuracy standard deviation / sqrt(number of episodes)
```

### 3.2 Interpretation

CALM-v1 achieved **85.74%** mean accuracy on the CIFAR-10 5-way 5-shot
evaluation. The confidence interval of **$\pm$ 0.49 percentage points**
indicates moderate variation across the 600 randomly sampled episodes.

This result should be compared with results from other datasets only after
considering differences in image resolution, domain, class semantics, and
dataset difficulty. It is directly useful as the CIFAR-10 baseline for the
corresponding CALM-v2 evaluation.

## 4. Continual learning results

### 4.1 Task organization

CIFAR-10 was divided into five sequential tasks:

| Task | Classes |
|---|---|
| Task 1 | Classes 0--1 |
| Task 2 | Classes 2--3 |
| Task 3 | Classes 4--5 |
| Task 4 | Classes 6--7 |
| Task 5 | Classes 8--9 |

At each stage, five support examples per class were added to the episodic
memory. The system was then evaluated on every task seen so far.

### 4.2 Accuracy matrix

Each row represents the point in the continual sequence after learning the
corresponding task. Each value is accuracy on the indicated task.

| After learning | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| Task 1 | **97.70** | -- | -- | -- | -- |
| Task 2 | 87.85 | **92.00** | -- | -- | -- |
| Task 3 | 87.75 | 75.90 | **79.30** | -- | -- |
| Task 4 | 87.60 | 71.85 | 76.95 | **81.50** | -- |
| Task 5 | 83.75 | 71.85 | 76.90 | 81.45 | **87.50** |

### 4.3 Aggregate continual metrics

| Metric | CALM-v1 result |
|---|---:|
| Number of tasks | 5 |
| Memory examples per class | 5 |
| Average accuracy | **87.60%** |
| Backward transfer (BWT) | **-9.14 percentage points** |

### 4.4 Average accuracy calculation

The runner defines average accuracy as the mean of the diagonal values, where
each diagonal entry is performance on a task immediately after that task is
learned:

```text
Average accuracy
= (97.70 + 92.00 + 79.30 + 81.50 + 87.50) / 5
= 87.60%
```

### 4.5 BWT calculation

BWT is computed over previously learned tasks:

```text
BWT = mean(final accuracy on previous task - accuracy immediately after learning it)
```

For this run:

| Previous task | Accuracy after learning | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 97.70 | 83.75 | -13.95 |
| Task 2 | 92.00 | 71.85 | -20.15 |
| Task 3 | 79.30 | 76.90 | -2.40 |
| Task 4 | 81.50 | 81.45 | -0.05 |

```text
BWT = (-13.95 - 20.15 - 2.40 - 0.05) / 4
BWT = -9.14 percentage points
```

### 4.6 Interpretation

The CIFAR-10 continual run produced **87.60% average accuracy** and
**-9.14 BWT**. The negative BWT indicates forgetting of earlier tasks after
later tasks are introduced. Forgetting is most pronounced for Task 2, whose
accuracy decreases from 92.00% to 71.85%.

Because CALM-v1 uses a flat memory and does not update the CLIP parameters,
the observed forgetting is not caused by neural network weight interference.
It is more plausibly associated with prototype competition as additional
classes are added, limited support coverage, and the fixed prototype
representation of each class.

The continual metric should not be interpreted as perfect retention simply
because the backbone is frozen. A frozen representation prevents parameter
drift, but classification boundaries can still change when new class
prototypes are added.

## 5. Comparison with existing CALM-v1 ImageNet results

The original CALM-v1 ImageNet results reported:

- 96.88% on 5-way 5-shot ImageNet.
- 59.61% average accuracy on 10-task ImageNet continual evaluation.
- -5.52 BWT on 10-task ImageNet continual evaluation.

The present CIFAR-10 results are not a replacement for those ImageNet
numbers. They provide a dataset-specific result for the new multi-dataset
evaluation suite:

| Dataset | Few-shot accuracy | Continual average accuracy | BWT |
|---|---:|---:|---:|
| ImageNet | 96.88% | 59.61% | -5.52 |
| CIFAR-10 | **85.74% $\pm$ 0.49** | **87.60%** | **-9.14** |
| CIFAR-100 | **85.44% $\pm$ 0.64** | **60.08%** | **-11.11** |
| STL-10 | **96.75% $\pm$ 0.21** | **98.14%** | **-5.58** |
| SVHN | **26.77% $\pm$ 0.47** | **27.64%** | **-13.62** |
| FashionMNIST | **78.25% $\pm$ 0.76** | **76.85%** | **-11.79** |

The different continual task structures explain why the accuracy values should
not be compared as if they were the same benchmark:

- ImageNet uses 10 tasks with 100 classes per task.
- CIFAR-10 uses 5 tasks with 2 classes per task.
- The datasets have different image sizes and visual complexity.

## 6. Reproducibility and reporting notes

- The GPU was correctly used: `NVIDIA GeForce RTX 3060 Ti`.
- The runner was invoked with `--device cuda`, preventing silent CPU fallback.
- The experiment used random seed 42.
- The model remained frozen.
- Support examples were selected randomly without replacement.
- Continual evaluation was performed on all official test examples for each
  seen task.
- Few-shot confidence intervals describe variation across episodes, not
  variation across independent training runs.
- The continual result is currently a single seeded run and does not include a
  multi-seed confidence interval.

## 7. Results to be added

Run the following commands from the same directory and add their outputs to
this report:

```powershell
python final-others.py --datasetname cifar100 --device cuda
python final-others.py --datasetname stl10 --device cuda
python final-others.py --datasetname svhn --device cuda
python final-others.py --datasetname fashionmnist --device cuda
```

### Results table template

| Dataset | Few-shot accuracy | 95% CI | Continual tasks | Average accuracy | BWT |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 85.74% | $\pm$ 0.49 | 5 | 87.60% | -9.14 |
| CIFAR-100 | **85.44%** | **$\pm$ 0.64** | 10 | **60.08%** | **-11.11** |
| STL-10 | **96.75%** | **$\pm$ 0.21** | 5 | **98.14%** | **-5.58** |
| SVHN | **26.77%** | **$\pm$ 0.47** | 5 | **27.64%** | **-13.62** |
| FashionMNIST | **78.25%** | **$\pm$ 0.76** | 5 | **76.85%** | **-11.79** |

## 13. Completed run: CIFAR-10 (CALM-v2)

### 13.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v2 |
| Dataset | CIFAR-10 |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Evaluation mode | Ensemble |
| CLIP backbones | ViT-B/32 and ViT-B/16 |
| Episodic memory | Hierarchical episodic memory |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 5 |
| Classes per continual task | 2 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v2\final-eval-other
python final-others.py --datasetname cifar10 --device cuda
```

### 13.2 Few-shot result

| Metric | CALM-v2 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Evaluation mode | Ensemble |
| Mean accuracy | **82.10%** |
| 95% confidence interval | **$\pm$ 0.51 percentage points** |

CALM-v2 achieved **82.10%** mean accuracy on the CIFAR-10 few-shot
evaluation. Under the same protocol, this is 3.64 percentage points below the
CALM-v1 result of 85.74%.

### 13.3 Continual task organization

CIFAR-10 was divided into five sequential tasks, with two classes per task.
At each stage, five support examples per class were added to hierarchical
episodic memory. Evaluation was then performed on every task seen so far.

### 13.4 Continual accuracy matrix

| After learning | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| Task 1 | **97.50** | -- | -- | -- | -- |
| Task 2 | 90.70 | **87.20** | -- | -- | -- |
| Task 3 | 90.60 | 68.05 | **81.20** | -- | -- |
| Task 4 | 90.10 | 63.85 | 79.25 | **77.60** | -- |
| Task 5 | 87.25 | 63.85 | 79.20 | 77.60 | **61.60** |

### 13.5 Aggregate continual metrics

| Metric | CALM-v2 result |
|---|---:|
| Number of tasks | 5 |
| Classes per task | 2 |
| Memory examples per class | 5 |
| Average accuracy | **81.02%** |
| Backward transfer (BWT) | **-8.90 percentage points** |

### 13.6 Average accuracy calculation

Average accuracy is the mean of the diagonal entries:

```text
Average accuracy
= (97.50 + 87.20 + 81.20 + 77.60 + 61.60) / 5
= 81.02%
```

### 13.7 BWT calculation

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 97.50 | 87.25 | -10.25 |
| Task 2 | 87.20 | 63.85 | -23.35 |
| Task 3 | 81.20 | 79.20 | -2.00 |
| Task 4 | 77.60 | 77.60 | 0.00 |

```text
BWT = (-10.25 - 23.35 - 2.00 + 0.00) / 4
BWT = -8.90 percentage points
```

### 13.8 Comparison with CALM-v1 on CIFAR-10

| Metric | CALM-v1 | CALM-v2 | Difference (v2 - v1) |
|---|---:|---:|---:|
| Few-shot accuracy | 85.74% | **82.10%** | **-3.64 pp** |
| Continual average accuracy | 87.60% | **81.02%** | **-6.58 pp** |
| BWT | -9.14 | **-8.90** | **+0.24 pp** |

On this first multi-dataset comparison, CALM-v2 has lower few-shot and
continual average accuracy than CALM-v1, while its BWT is 0.24 percentage
points less negative. Thus, the ensemble and hierarchical memory show a
small retention improvement in this run, but they do not yet produce a
performance improvement in absolute accuracy.

### 13.9 Interpretation

The CALM-v2 CIFAR-10 run achieved **81.02% average accuracy** and
**-8.90 BWT**. The largest forgetting occurs for Task 2, which decreases by
23.35 percentage points. Task 4 is retained exactly at its initial measured
accuracy.

The result should be interpreted with the model and memory changes in mind:
CALM-v2 uses the ViT-B/32 and ViT-B/16 ensemble together with hierarchical
episodic memory, whereas CALM-v1 uses a single ViT-B/32 backbone and flat
memory. The lower v2 accuracy does not show that the ensemble is intrinsically
ineffective; it indicates that the current ensemble aggregation, prototype
space alignment, and hierarchical-memory configuration are not yet optimized
for this dataset. The modestly improved BWT suggests that the v2 configuration
may provide some retention benefit despite its lower absolute accuracy.

## 12. Completed run: FashionMNIST (CALM-v1)

### 12.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v1 |
| Dataset | FashionMNIST |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 5 |
| Classes per continual task | 2 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation\final-eval-other
python final-others.py --datasetname fashionmnist --device cuda
```

### 12.2 Few-shot result

| Metric | CALM-v1 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Mean accuracy | **78.25%** |
| 95% confidence interval | **$\pm$ 0.76 percentage points** |

CALM-v1 achieved **78.25%** mean accuracy on the FashionMNIST few-shot
evaluation. This is below the STL-10 result but substantially above the SVHN
result under the same episodic protocol.

### 12.3 Continual task organization

FashionMNIST was divided into five sequential tasks:

| Task | Classes |
|---|---|
| Task 1 | Classes 0--1 |
| Task 2 | Classes 2--3 |
| Task 3 | Classes 4--5 |
| Task 4 | Classes 6--7 |
| Task 5 | Classes 8--9 |

At each stage, five randomly selected support examples per class were added to
flat episodic memory. Evaluation was then performed on every task seen so far.

### 12.4 Continual accuracy matrix

| After learning | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| Task 1 | **91.85** | -- | -- | -- | -- |
| Task 2 | 82.75 | **72.85** | -- | -- | -- |
| Task 3 | 80.55 | 58.95 | **78.35** | -- | -- |
| Task 4 | 79.95 | 56.05 | 69.20 | **56.65** | -- |
| Task 5 | 79.85 | 55.95 | 68.05 | 48.70 | **84.55** |

### 12.5 Aggregate continual metrics

| Metric | CALM-v1 result |
|---|---:|
| Number of tasks | 5 |
| Classes per task | 2 |
| Memory examples per class | 5 |
| Average accuracy | **76.85%** |
| Backward transfer (BWT) | **-11.79 percentage points** |

### 12.6 Average accuracy calculation

Average accuracy is the mean of the diagonal entries:

```text
Average accuracy
= (91.85 + 72.85 + 78.35 + 56.65 + 84.55) / 5
= 76.85%
```

### 12.7 BWT calculation

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 91.85 | 79.85 | -12.00 |
| Task 2 | 72.85 | 55.95 | -16.90 |
| Task 3 | 78.35 | 68.05 | -10.30 |
| Task 4 | 56.65 | 48.70 | -7.95 |

```text
BWT = (-12.00 - 16.90 - 10.30 - 7.95) / 4
BWT = -11.79 percentage points
```

### 12.8 Interpretation

The FashionMNIST continual run produced **76.85% average accuracy** and
**-11.79 BWT**. The model retains useful performance across the sequence,
but all previously learned tasks experience negative transfer as new classes
are added. Task 2 shows the largest decline, falling by 16.90 percentage
points, while Task 4 declines by 7.95 points.

The result demonstrates a middle-ground behavior relative to the other
datasets: FashionMNIST is considerably easier for this frozen CLIP and
prototype memory setup than SVHN, but it is less compatible than STL-10.
Because the CLIP backbone is frozen, the observed degradation reflects
prototype competition, limited five-shot support, and representation-domain
alignment rather than learned-weight interference.

### 12.9 Updated CALM-v1 results summary

| Dataset | Few-shot accuracy | 95% CI | Continual tasks | Average accuracy | BWT |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 85.74% | $\pm$ 0.49 | 5 | 87.60% | -9.14 |
| CIFAR-100 | 85.44% | $\pm$ 0.64 | 10 | 60.08% | -11.11 |
| STL-10 | 96.75% | $\pm$ 0.21 | 5 | 98.14% | -5.58 |
| SVHN | 26.77% | $\pm$ 0.47 | 5 | 27.64% | -13.62 |
| FashionMNIST | **78.25%** | **$\pm$ 0.76** | 5 | **76.85%** | **-11.79** |

## 8. CALM-v2 comparison placeholder

After the corresponding CALM-v2 runs are completed, add a version comparison
for each dataset:

| Dataset | Metric | CALM-v1 | CALM-v2 | Difference |
|---|---|---:|---:|---:|
| CIFAR-10 | Few-shot accuracy | 85.74% | **82.10%** | **-3.64 pp** |
| CIFAR-10 | Continual average accuracy | 87.60% | **81.02%** | **-6.58 pp** |
| CIFAR-10 | BWT | -9.14 | **-8.90** | **+0.24 pp** |
| CIFAR-100 | Few-shot accuracy | 85.44% | **82.12%** | **-3.32 pp** |
| CIFAR-100 | Continual average accuracy | 60.08% | **54.35%** | **-5.73 pp** |
| CIFAR-100 | BWT | -11.11 | **-11.70** | **-0.59 pp** |
| STL-10 | Few-shot accuracy | 96.75% | **94.57%** | **-2.18 pp** |
| STL-10 | Continual average accuracy | 98.14% | **95.10%** | **-3.04 pp** |
| STL-10 | BWT | -5.58 | **-9.12** | **-3.54 pp** |
| SVHN | Few-shot accuracy | 26.77% | **26.21%** | **-0.56 pp** |
| SVHN | Continual average accuracy | 27.64% | **26.35%** | **-1.29 pp** |
| SVHN | BWT | -13.62 | **-12.75** | **+0.87 pp** |
| FashionMNIST | Few-shot accuracy | 78.25% | **74.77%** | **-3.48 pp** |
| FashionMNIST | Continual average accuracy | 76.85% | **69.90%** | **-6.95 pp** |
| FashionMNIST | BWT | -11.79 | **-10.16** | **+1.63 pp** |

For CALM-v2, also record whether the run used:

- Default ensemble mode.
- `--single_backbone` mode.
- Temperature value.
- ImageNet-domain or dataset-domain compression training.

These details are necessary to make the v1/v2 comparison scientifically
interpretable.

## 9. Completed run: CIFAR-100 (CALM-v1)

### 9.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v1 |
| Dataset | CIFAR-100 |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 10 |
| Classes per continual task | 10 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation\final-eval-other
python final-others.py --datasetname cifar100 --device cuda
```

### 9.2 Few-shot result

| Metric | CALM-v1 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Mean accuracy | **85.44%** |
| 95% confidence interval | **$\pm$ 0.64 percentage points** |

CALM-v1 achieved **85.44%** on the CIFAR-100 few-shot evaluation. The
confidence interval was computed across episode accuracies using:

```text
95% CI = 1.96 × episode-accuracy standard deviation / sqrt(number of episodes)
```

The result is close to the CIFAR-10 few-shot score, although the class
taxonomy is substantially larger. The value should be interpreted as a
dataset-specific result rather than compared directly with ImageNet accuracy.

### 9.3 Continual task organization

CIFAR-100 was divided into ten sequential tasks, with ten classes per task:

| Task | Classes |
|---|---|
| Task 1 | Classes 0--9 |
| Task 2 | Classes 10--19 |
| Task 3 | Classes 20--29 |
| Task 4 | Classes 30--39 |
| Task 5 | Classes 40--49 |
| Task 6 | Classes 50--59 |
| Task 7 | Classes 60--69 |
| Task 8 | Classes 70--79 |
| Task 9 | Classes 80--89 |
| Task 10 | Classes 90--99 |

At each stage, five randomly selected support examples per class were added to
flat episodic memory. The model was then evaluated on every task observed so
far.

### 9.4 Continual accuracy matrix

Each row reports performance after learning the task named in the first
column. Blank entries were not evaluated because those tasks had not yet been
introduced.

| After learning | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task 1 | **81.20** | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Task 2 | 71.50 | **70.00** | -- | -- | -- | -- | -- | -- | -- | -- |
| Task 3 | 65.80 | 66.60 | **61.10** | -- | -- | -- | -- | -- | -- | -- |
| Task 4 | 62.30 | 62.90 | 58.90 | **62.10** | -- | -- | -- | -- | -- | -- |
| Task 5 | 58.50 | 59.80 | 56.40 | 59.70 | **57.80** | -- | -- | -- | -- | -- |
| Task 6 | 56.70 | 57.90 | 55.90 | 57.70 | 54.80 | **55.00** | -- | -- | -- | -- |
| Task 7 | 55.50 | 55.80 | 55.30 | 52.30 | 53.10 | 53.70 | **55.60** | -- | -- | -- |
| Task 8 | 54.80 | 54.20 | 53.80 | 50.00 | 51.90 | 51.60 | 53.10 | **45.80** | -- | -- |
| Task 9 | 54.00 | 50.30 | 53.10 | 48.60 | 50.00 | 50.20 | 52.20 | 45.80 | **59.10** | -- |
| Task 10 | 53.10 | 50.00 | 52.90 | 44.60 | 47.90 | 46.60 | 51.30 | 43.90 | 57.40 | **53.10** |

### 9.5 Aggregate continual metrics

| Metric | CALM-v1 result |
|---|---:|
| Number of tasks | 10 |
| Classes per task | 10 |
| Memory examples per class | 5 |
| Average accuracy | **60.08%** |
| Backward transfer (BWT) | **-11.11 percentage points** |

### 9.6 Average accuracy calculation

The runner calculates average accuracy as the mean of the diagonal entries:

```text
Average accuracy
= (81.20 + 70.00 + 61.10 + 62.10 + 57.80
   + 55.00 + 55.60 + 45.80 + 59.10 + 53.10) / 10
= 60.08%
```

The diagonal values represent each task's accuracy immediately after its
support examples are introduced.

### 9.7 BWT calculation

For each previously learned task, BWT compares final accuracy after Task 10
with accuracy immediately after that task was learned:

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 81.20 | 53.10 | -28.10 |
| Task 2 | 70.00 | 50.00 | -20.00 |
| Task 3 | 61.10 | 52.90 | -8.20 |
| Task 4 | 62.10 | 44.60 | -17.50 |
| Task 5 | 57.80 | 47.90 | -9.90 |
| Task 6 | 55.00 | 46.60 | -8.40 |
| Task 7 | 55.60 | 51.30 | -4.30 |
| Task 8 | 45.80 | 43.90 | -1.90 |
| Task 9 | 59.10 | 57.40 | -1.70 |

```text
BWT = (-28.10 - 20.00 - 8.20 - 17.50 - 9.90
       - 8.40 - 4.30 - 1.90 - 1.70) / 9
BWT = -11.11 percentage points
```

### 9.8 Interpretation

The CIFAR-100 continual run produced **60.08% average accuracy** and
**-11.11 BWT**. The result indicates measurable forgetting as new groups of
classes are added. The largest decreases occur for Task 1 and Task 2, while
later tasks show smaller final changes.

The lower average accuracy than CIFAR-10 is expected to be influenced by the
larger number of competing classes and the same five examples per class being
used to represent each class. Since the CLIP backbone remains frozen, the
degradation is not due to neural-network weight updates; it arises from the
non-parametric prototype decision problem as the memory's class set expands.

### 9.9 CIFAR-100 reporting status

The CIFAR-100 CALM-v1 result is now complete. The corresponding CALM-v2 result
should be added using the same dataset split, task partition, shot count,
number of episodes, seed, and GPU configuration. The v1/v2 comparison should
also record whether CALM-v2 used ensemble mode or `--single_backbone`.

## 10. Completed run: STL-10 (CALM-v1)

### 10.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v1 |
| Dataset | STL-10 |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 5 |
| Classes per continual task | 2 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation\final-eval-other
python final-others.py --datasetname stl10 --device cuda
```

### 10.2 Few-shot result

| Metric | CALM-v1 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Mean accuracy | **96.75%** |
| 95% confidence interval | **$\pm$ 0.21 percentage points** |

CALM-v1 achieved **96.75%** mean accuracy on the STL-10 few-shot
evaluation. The small confidence interval indicates relatively low variation
across the 600 sampled episodes.

### 10.3 Continual task organization

STL-10 was divided into five sequential tasks:

| Task | Classes |
|---|---|
| Task 1 | Classes 0--1 |
| Task 2 | Classes 2--3 |
| Task 3 | Classes 4--5 |
| Task 4 | Classes 6--7 |
| Task 5 | Classes 8--9 |

At each stage, five randomly selected support examples per class were added to
flat episodic memory. Evaluation was then performed on every task seen so far.

### 10.4 Continual accuracy matrix

| After learning | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| Task 1 | **99.94** | -- | -- | -- | -- |
| Task 2 | 98.81 | **97.31** | -- | -- | -- |
| Task 3 | 98.50 | 93.69 | **97.06** | -- | -- |
| Task 4 | 93.44 | 91.00 | 95.81 | **96.75** | -- |
| Task 5 | 92.69 | 83.69 | 95.81 | 96.56 | **99.62** |

### 10.5 Aggregate continual metrics

| Metric | CALM-v1 result |
|---|---:|
| Number of tasks | 5 |
| Classes per task | 2 |
| Memory examples per class | 5 |
| Average accuracy | **98.14%** |
| Backward transfer (BWT) | **-5.58 percentage points** |

### 10.6 Average accuracy calculation

Average accuracy is the mean of the diagonal entries:

```text
Average accuracy
= (99.94 + 97.31 + 97.06 + 96.75 + 99.62) / 5
= 98.14%
```

### 10.7 BWT calculation

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 99.94 | 92.69 | -7.25 |
| Task 2 | 97.31 | 83.69 | -13.62 |
| Task 3 | 97.06 | 95.81 | -1.25 |
| Task 4 | 96.75 | 96.56 | -0.19 |

```text
BWT = (-7.25 - 13.62 - 1.25 - 0.19) / 4
BWT = -5.58 percentage points
```

### 10.8 Interpretation

The STL-10 continual run produced **98.14% average accuracy** and
**-5.58 BWT**. The high average accuracy suggests that the frozen CLIP
representation is particularly effective for the STL-10 visual domain under
this protocol.

Forgetting is concentrated in Task 2, which decreases from 97.31% to 83.69%
after the complete sequence. Task 4 is almost unchanged, decreasing by only
0.19 percentage points. As with the other CALM-v1 results, the forgetting
comes from changes in the prototype classification problem as new classes are
added, rather than from updates to the frozen CLIP parameters.

### 10.9 Updated CALM-v1 results summary

| Dataset | Few-shot accuracy | 95% CI | Continual tasks | Average accuracy | BWT |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 85.74% | $\pm$ 0.49 | 5 | 87.60% | -9.14 |
| CIFAR-100 | 85.44% | $\pm$ 0.64 | 10 | 60.08% | -11.11 |
| STL-10 | **96.75%** | **$\pm$ 0.21** | 5 | **98.14%** | **-5.58** |
| SVHN | 26.77% | $\pm$ 0.47 | 5 | 27.64% | -13.62 |
| FashionMNIST | 78.25% | $\pm$ 0.76 | 5 | 76.85% | -11.79 |

## 11. Completed run: SVHN (CALM-v1)

### 11.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v1 |
| Dataset | SVHN |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 5 |
| Classes per continual task | 2 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation\final-eval-other
python final-others.py --datasetname svhn --device cuda
```

### 11.2 Few-shot result

| Metric | CALM-v1 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Mean accuracy | **26.77%** |
| 95% confidence interval | **$\pm$ 0.47 percentage points** |

CALM-v1 achieved **26.77%** mean accuracy on the SVHN few-shot evaluation.
The result is substantially lower than the CIFAR-10, CIFAR-100, and STL-10
results under the same episodic protocol, indicating a domain mismatch between
the frozen CLIP representation and the street-number images.

### 11.3 Continual task organization

SVHN was divided into five sequential tasks:

| Task | Classes |
|---|---|
| Task 1 | Classes 0--1 |
| Task 2 | Classes 2--3 |
| Task 3 | Classes 4--5 |
| Task 4 | Classes 6--7 |
| Task 5 | Classes 8--9 |

At each stage, five randomly selected support examples per class were added to
flat episodic memory. Evaluation was then performed on every task seen so far.

### 11.4 Continual accuracy matrix

| After learning | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| Task 1 | **45.67** | -- | -- | -- | -- |
| Task 2 | 33.74 | **31.55** | -- | -- | -- |
| Task 3 | 30.98 | 14.79 | **33.85** | -- | -- |
| Task 4 | 20.91 | 12.69 | 29.57 | **20.15** | -- |
| Task 5 | 20.23 | 11.79 | 26.55 | 18.17 | **6.97** |

### 11.5 Aggregate continual metrics

| Metric | CALM-v1 result |
|---|---:|
| Number of tasks | 5 |
| Classes per task | 2 |
| Memory examples per class | 5 |
| Average accuracy | **27.64%** |
| Backward transfer (BWT) | **-13.62 percentage points** |

### 11.6 Average accuracy calculation

Average accuracy is the mean of the diagonal entries:

```text
Average accuracy
= (45.67 + 31.55 + 33.85 + 20.15 + 6.97) / 5
= 27.64%
```

### 11.7 BWT calculation

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 45.67 | 20.23 | -25.44 |
| Task 2 | 31.55 | 11.79 | -19.76 |
| Task 3 | 33.85 | 26.55 | -7.30 |
| Task 4 | 20.15 | 18.17 | -1.98 |

```text
BWT = (-25.44 - 19.76 - 7.30 - 1.98) / 4
BWT = -13.62 percentage points
```

### 11.8 Interpretation

The SVHN continual run produced **27.64% average accuracy** and
**-13.62 BWT**. Performance declines substantially as additional digit
classes are introduced. The final accuracy for Task 5 is only 6.97%, while
Task 1 falls by 25.44 percentage points and Task 2 falls by 19.76 points.

These results should be interpreted as evidence that the frozen ViT-B/32
representation and flat prototype memory are poorly matched to SVHN under
this evaluation protocol. The weak few-shot result is already present before
continual learning begins, and the continual sequence compounds that
difficulty through prototype competition and limited support coverage. This
also makes SVHN a useful stress test for CALM-v2: improvements should be
reported separately from the stronger natural-image datasets rather than
treated as a simple aggregate gain.

### 11.9 Updated CALM-v1 results summary

| Dataset | Few-shot accuracy | 95% CI | Continual tasks | Average accuracy | BWT |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 85.74% | $\pm$ 0.49 | 5 | 87.60% | -9.14 |
| CIFAR-100 | 85.44% | $\pm$ 0.64 | 10 | 60.08% | -11.11 |
| STL-10 | 96.75% | $\pm$ 0.21 | 5 | 98.14% | -5.58 |
| SVHN | **26.77%** | **$\pm$ 0.47** | 5 | **27.64%** | **-13.62** |
| FashionMNIST | **78.25%** | **$\pm$ 0.76** | 5 | **76.85%** | **-11.79** |

## 14. Completed run: CIFAR-100 (CALM-v2)

### 14.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v2 |
| Dataset | CIFAR-100 |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Evaluation mode | Ensemble |
| CLIP backbones | ViT-B/32 and ViT-B/16 |
| Episodic memory | Hierarchical episodic memory |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 10 |
| Classes per continual task | 10 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v2\final-eval-other
python final-others.py --datasetname cifar100 --device cuda
```

### 14.2 Few-shot result

| Metric | CALM-v2 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Evaluation mode | Ensemble |
| Mean accuracy | **82.12%** |
| 95% confidence interval | **$\pm$ 0.74 percentage points** |

CALM-v2 achieved **82.12%** mean accuracy on the CIFAR-100 few-shot
evaluation. Under the same protocol, this is 3.32 percentage points below the
CALM-v1 result of 85.44%.

### 14.3 Continual task organization

CIFAR-100 was divided into ten sequential tasks, with ten classes per task.
At each stage, five support examples per class were added to hierarchical
episodic memory. Evaluation was then performed on every task seen so far.

### 14.4 Continual accuracy matrix

| After learning | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | **78.90** | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| T2 | 69.10 | **62.50** | -- | -- | -- | -- | -- | -- | -- | -- |
| T3 | 63.10 | 59.10 | **57.20** | -- | -- | -- | -- | -- | -- | -- |
| T4 | 56.80 | 53.30 | 55.20 | **59.50** | -- | -- | -- | -- | -- | -- |
| T5 | 53.60 | 51.40 | 53.20 | 57.70 | **48.10** | -- | -- | -- | -- | -- |
| T6 | 50.70 | 50.50 | 52.60 | 55.30 | 45.50 | **48.70** | -- | -- | -- | -- |
| T7 | 49.20 | 48.30 | 51.80 | 49.10 | 44.00 | 46.60 | **50.60** | -- | -- | -- |
| T8 | 48.60 | 47.10 | 51.20 | 48.10 | 42.70 | 45.30 | 49.20 | **34.40** | -- | -- |
| T9 | 47.80 | 41.10 | 50.00 | 47.20 | 42.10 | 43.60 | 48.70 | 34.30 | **55.00** | -- |
| T10 | 47.40 | 40.60 | 49.50 | 44.00 | 38.60 | 37.60 | 47.20 | 31.10 | 53.60 | **48.60** |

### 14.5 Aggregate continual metrics

| Metric | CALM-v2 result |
|---|---:|
| Number of tasks | 10 |
| Classes per task | 10 |
| Memory examples per class | 5 |
| Average accuracy | **54.35%** |
| Backward transfer (BWT) | **-11.70 percentage points** |

### 14.6 Average accuracy calculation

Average accuracy is the mean of the diagonal entries:

```text
Average accuracy
= (78.90 + 62.50 + 57.20 + 59.50 + 48.10
   + 48.70 + 50.60 + 34.40 + 55.00 + 48.60) / 10
= 54.35%
```

### 14.7 BWT calculation

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 78.90 | 47.40 | -31.50 |
| Task 2 | 62.50 | 40.60 | -21.90 |
| Task 3 | 57.20 | 49.50 | -7.70 |
| Task 4 | 59.50 | 44.00 | -15.50 |
| Task 5 | 48.10 | 38.60 | -9.50 |
| Task 6 | 48.70 | 37.60 | -11.10 |
| Task 7 | 50.60 | 47.20 | -3.40 |
| Task 8 | 34.40 | 31.10 | -3.30 |
| Task 9 | 55.00 | 53.60 | -1.40 |

```text
BWT = (-31.50 - 21.90 - 7.70 - 15.50 - 9.50
       - 11.10 - 3.40 - 3.30 - 1.40) / 9
BWT = -11.70 percentage points
```

### 14.8 Comparison with CALM-v1 on CIFAR-100

| Metric | CALM-v1 | CALM-v2 | Difference (v2 - v1) |
|---|---:|---:|---:|
| Few-shot accuracy | 85.44% | **82.12%** | **-3.32 pp** |
| Continual average accuracy | 60.08% | **54.35%** | **-5.73 pp** |
| BWT | -11.11 | **-11.70** | **-0.59 pp** |

On CIFAR-100, CALM-v2 is lower than CALM-v1 in both few-shot accuracy and
continual average accuracy. Its BWT is 0.59 percentage points more negative,
indicating slightly greater forgetting under the current ensemble and
hierarchical-memory configuration.

### 14.9 Interpretation

The CALM-v2 CIFAR-100 run achieved **54.35% average accuracy** and
**-11.70 BWT**. The largest forgetting occurs for Task 1, which decreases by
31.50 percentage points, followed by Task 2, which decreases by 21.90 points.
The final task has 48.60% accuracy after being learned.

The result is consistent with the CIFAR-10 comparison: the current CALM-v2
configuration does not yet improve absolute accuracy over CALM-v1. The
additional loss in BWT on CIFAR-100 also suggests that hierarchical memory and
ensemble aggregation are not sufficient by themselves to preserve performance
as the number of competing classes grows. Potential causes include prototype
space alignment between ensemble backbones, uncalibrated similarity fusion,
and compression settings that have not been optimized for CIFAR-100.

## 15. Completed run: STL-10 (CALM-v2)

### 15.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v2 |
| Dataset | STL-10 |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Evaluation mode | Ensemble |
| CLIP backbones | ViT-B/32 and ViT-B/16 |
| Episodic memory | Hierarchical episodic memory |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 5 |
| Classes per continual task | 2 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v2\final-eval-other
python final-others.py --datasetname stl10 --device cuda
```

### 15.2 Few-shot result

| Metric | CALM-v2 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Evaluation mode | Ensemble |
| Mean accuracy | **94.57%** |
| 95% confidence interval | **$\pm$ 0.30 percentage points** |

CALM-v2 achieved **94.57%** mean accuracy on the STL-10 few-shot
evaluation. Under the same protocol, this is 2.18 percentage points below the
CALM-v1 result of 96.75%.

### 15.3 Continual task organization

STL-10 was divided into five sequential tasks, with two classes per task. At
each stage, five support examples per class were added to hierarchical
episodic memory. Evaluation was then performed on every task seen so far.

### 15.4 Continual accuracy matrix

| After learning | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| Task 1 | **99.88** | -- | -- | -- | -- |
| Task 2 | 99.19 | **98.38** | -- | -- | -- |
| Task 3 | 98.12 | 84.62 | **98.56** | -- | -- |
| Task 4 | 95.75 | 81.88 | 97.44 | **93.81** | -- |
| Task 5 | 93.50 | 70.62 | 97.38 | 92.62 | **84.88** |

### 15.5 Aggregate continual metrics

| Metric | CALM-v2 result |
|---|---:|
| Number of tasks | 5 |
| Classes per task | 2 |
| Memory examples per class | 5 |
| Average accuracy | **95.10%** |
| Backward transfer (BWT) | **-9.12 percentage points** |

### 15.6 Average accuracy calculation

Average accuracy is the mean of the diagonal entries:

```text
Average accuracy
= (99.88 + 98.38 + 98.56 + 93.81 + 84.88) / 5
= 95.10%
```

### 15.7 BWT calculation

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 99.88 | 93.50 | -6.38 |
| Task 2 | 98.38 | 70.62 | -27.76 |
| Task 3 | 98.56 | 97.38 | -1.18 |
| Task 4 | 93.81 | 92.62 | -1.19 |

```text
BWT = (-6.38 - 27.76 - 1.18 - 1.19) / 4
BWT ≈ -9.12 percentage points (reported value; calculations use rounded accuracies)
```

### 15.8 Comparison with CALM-v1 on STL-10

| Metric | CALM-v1 | CALM-v2 | Difference (v2 - v1) |
|---|---:|---:|---:|
| Few-shot accuracy | 96.75% | **94.57%** | **-2.18 pp** |
| Continual average accuracy | 98.14% | **95.10%** | **-3.04 pp** |
| BWT | -5.58 | **-9.12** | **-3.54 pp** |

On STL-10, CALM-v2 is lower than CALM-v1 in few-shot accuracy and continual
average accuracy. Its BWT is 3.54 percentage points more negative, indicating
greater forgetting under the current ensemble and hierarchical-memory setup.

### 15.9 Interpretation

The CALM-v2 STL-10 run produced **95.10% average accuracy** and **-9.12 BWT**.
The largest forgetting occurs for Task 2, which decreases by 27.76 percentage
points. Task 3 is retained well, with a final decrease of only 1.18 points.

STL-10 remains a strong domain for both frameworks, but the current CALM-v2
configuration does not improve absolute performance over CALM-v1. Despite the
high accuracy, the ensemble and hierarchical memory introduce more forgetting
on this run. This suggests that the additional architecture is not yet
optimally calibrated for STL-10, particularly in prototype alignment and
similarity-score aggregation.

## 16. Completed run: SVHN (CALM-v2)

### 16.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v2 |
| Dataset | SVHN |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Evaluation mode | Ensemble |
| CLIP backbones | ViT-B/32 and ViT-B/16 |
| Episodic memory | Hierarchical episodic memory |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 5 |
| Classes per continual task | 2 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v2\final-eval-other
python final-others.py --datasetname svhn --device cuda
```

### 16.2 Few-shot result

| Metric | CALM-v2 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Evaluation mode | Ensemble |
| Mean accuracy | **26.21%** |
| 95% confidence interval | **$\pm$ 0.43 percentage points** |

CALM-v2 achieved **26.21%** mean accuracy on the SVHN few-shot evaluation.
This is 0.56 percentage points below the CALM-v1 result of 26.77%, confirming
that SVHN remains a difficult domain for both frozen CLIP configurations.

### 16.3 Continual task organization

SVHN was divided into five sequential tasks, with two digit classes per task.
At each stage, five support examples per class were added to hierarchical
episodic memory. Evaluation was then performed on every task seen so far.

### 16.4 Continual accuracy matrix

| After learning | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| Task 1 | **40.01** | -- | -- | -- | -- |
| Task 2 | 31.90 | **29.31** | -- | -- | -- |
| Task 3 | 24.96 | 10.45 | **41.43** | -- | -- |
| Task 4 | 17.42 | 8.55 | 33.97 | **20.17** | -- |
| Task 5 | 17.42 | 8.53 | 33.89 | 20.07 | **0.83** |

### 16.5 Aggregate continual metrics

| Metric | CALM-v2 result |
|---|---:|
| Number of tasks | 5 |
| Classes per task | 2 |
| Memory examples per class | 5 |
| Average accuracy | **26.35%** |
| Backward transfer (BWT) | **-12.75 percentage points** |

### 16.6 Average accuracy calculation

Average accuracy is the mean of the diagonal entries:

```text
Average accuracy
= (40.01 + 29.31 + 41.43 + 20.17 + 0.83) / 5
= 26.35%
```

### 16.7 BWT calculation

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 40.01 | 17.42 | -22.59 |
| Task 2 | 29.31 | 8.53 | -20.78 |
| Task 3 | 41.43 | 33.89 | -7.54 |
| Task 4 | 20.17 | 20.07 | -0.10 |

```text
BWT = (-22.59 - 20.78 - 7.54 - 0.10) / 4
BWT = -12.75 percentage points
```

### 16.8 Comparison with CALM-v1 on SVHN

| Metric | CALM-v1 | CALM-v2 | Difference (v2 - v1) |
|---|---:|---:|---:|
| Few-shot accuracy | 26.77% | **26.21%** | **-0.56 pp** |
| Continual average accuracy | 27.64% | **26.35%** | **-1.29 pp** |
| BWT | -13.62 | **-12.75** | **+0.87 pp** |

On SVHN, CALM-v2 is slightly lower than CALM-v1 in few-shot and continual
average accuracy. However, its BWT is 0.87 percentage points less negative,
indicating a modest retention improvement despite the lower absolute accuracy.

### 16.9 Interpretation

The CALM-v2 SVHN run produced **26.35% average accuracy** and **-12.75 BWT**.
The final Task 5 accuracy is only 0.83%, and the largest retention losses occur
for Task 1 and Task 2.

The low few-shot result indicates that the primary limitation is
representation-domain mismatch between the frozen CLIP ensemble and SVHN,
rather than continual learning alone. Relative to CALM-v1, the slightly
improved BWT suggests that hierarchical memory and ensemble aggregation may
help retain information, but they do not overcome the weak initial
representation quality on this dataset.

## 17. Completed run: FashionMNIST (CALM-v2)

### 17.1 Configuration

| Setting | Value |
|---|---|
| Framework | CALM-v2 |
| Dataset | FashionMNIST |
| Dataset split | Official test split |
| Device | NVIDIA GeForce RTX 3060 Ti |
| Evaluation mode | Ensemble |
| CLIP backbones | ViT-B/32 and ViT-B/16 |
| Episodic memory | Hierarchical episodic memory |
| Few-shot episodes | 600 |
| Few-shot configuration | 5-way, 5-shot |
| Query images | 15 per class |
| Continual tasks | 5 |
| Classes per continual task | 2 |
| Continual memory | 5 examples per class |
| Random seed | 42 |

The experiment was run with:

```powershell
cd E:\Uni\CSE498R\CALM\CALM-v2\final-eval-other
python final-others.py --datasetname fashionmnist --device cuda
```

### 17.2 Few-shot result

| Metric | CALM-v2 result |
|---|---:|
| Configuration | 5-way, 5-shot |
| Query images per class | 15 |
| Number of episodes | 600 |
| Evaluation mode | Ensemble |
| Mean accuracy | **74.77%** |
| 95% confidence interval | **$\pm$ 0.74 percentage points** |

CALM-v2 achieved **74.77%** mean accuracy on the FashionMNIST few-shot
evaluation. Under the same protocol, this is 3.48 percentage points below the
CALM-v1 result of 78.25%.

### 17.3 Continual task organization

FashionMNIST was divided into five sequential tasks, with two classes per task.
At each stage, five support examples per class were added to hierarchical
episodic memory. Evaluation was then performed on every task seen so far.

### 17.4 Continual accuracy matrix

| After learning | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|
| Task 1 | **93.00** | -- | -- | -- | -- |
| Task 2 | 88.20 | **55.10** | -- | -- | -- |
| Task 3 | 86.80 | 50.90 | **70.30** | -- | -- |
| Task 4 | 86.80 | 50.85 | 67.40 | **43.60** | -- |
| Task 5 | 86.35 | 50.65 | 64.70 | 19.65 | **87.50** |

### 17.5 Aggregate continual metrics

| Metric | CALM-v2 result |
|---|---:|
| Number of tasks | 5 |
| Classes per task | 2 |
| Memory examples per class | 5 |
| Average accuracy | **69.90%** |
| Backward transfer (BWT) | **-10.16 percentage points** |

### 17.6 Average accuracy calculation

Average accuracy is the mean of the diagonal entries:

```text
Average accuracy
= (93.00 + 55.10 + 70.30 + 43.60 + 87.50) / 5
= 69.90%
```

### 17.7 BWT calculation

| Previous task | Initial accuracy | Final accuracy | Change |
|---|---:|---:|---:|
| Task 1 | 93.00 | 86.35 | -6.65 |
| Task 2 | 55.10 | 50.65 | -4.45 |
| Task 3 | 70.30 | 64.70 | -5.60 |
| Task 4 | 43.60 | 19.65 | -23.95 |

```text
BWT = (-6.65 - 4.45 - 5.60 - 23.95) / 4
BWT = -10.16 percentage points
```

### 17.8 Comparison with CALM-v1 on FashionMNIST

| Metric | CALM-v1 | CALM-v2 | Difference (v2 - v1) |
|---|---:|---:|---:|
| Few-shot accuracy | 78.25% | **74.77%** | **-3.48 pp** |
| Continual average accuracy | 76.85% | **69.90%** | **-6.95 pp** |
| BWT | -11.79 | **-10.16** | **+1.63 pp** |

On FashionMNIST, CALM-v2 is lower than CALM-v1 in few-shot accuracy and
continual average accuracy. Its BWT is 1.63 percentage points less negative,
indicating improved retention despite the lower absolute accuracy.

### 17.9 Interpretation

The CALM-v2 FashionMNIST run produced **69.90% average accuracy** and
**-10.16 BWT**. The largest forgetting occurs for Task 4, which decreases by
23.95 percentage points, while the final Task 5 accuracy is 87.50%.

The result follows the pattern observed on the other datasets: the ensemble
and hierarchical memory provide a modest retention benefit relative to
CALM-v1, but do not improve absolute accuracy under the current configuration.
The lower few-shot accuracy suggests that representation alignment and
ensemble similarity calibration remain important limitations for FashionMNIST.
