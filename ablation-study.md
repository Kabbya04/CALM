# CALM Ablation Study

## Purpose and scope

This document consolidates the ablation-related evidence currently available in
the CALM repository. It is intended to serve as the working source for the
ablation section of the research paper, not as a replacement for raw
experiment logs. The central question is:

> Which parts of CALM are responsible for its observed few-shot accuracy,
> continual-learning behavior, memory scaling claims, and deployment decisions?

The repository contains two main implementations:

- **CALM-v1:** frozen CLIP ViT-B/32, flat episodic memory, class prototypes,
  cosine-similarity classification, and a binary readiness policy.
- **CALM-v2:** two CLIP variants (ViT-B/32 and ViT-B/16), ensemble score
  fusion, disagreement statistics, tiered readiness logic, and a hierarchical
  episodic-memory implementation with learned compression modules.

The available results do not yet provide a complete factorial ablation of those
components. The v1--v2 comparisons change several components at once. The
results below are therefore reported as **variant comparisons** or
**sensitivity studies**, not as causal evidence for any single component unless
the protocol isolates that component.

## Evidence policy

The following labels are used throughout this document:

- **Measured:** a numerical result is recorded in an experiment report or paper
  artifact and can be traced to an evaluation script.
- **Reported:** a numerical result appears in a report or paper, but the
  repository does not contain sufficient raw output or a fully matching command
  to independently reproduce it.
- **Diagnostic:** a code-path or implementation observation that explains a
  possible mechanism but is not itself a performance measurement.
- **Proposed:** a missing experiment or an experimental design recommended for
  the paper.

This distinction matters because several repository documents describe target
values as if they were achieved values. In particular, the following should
not be presented as measured outcomes without new runs:

- 2.5x empirical memory capacity;
- a causal benefit from compression;
- a causal benefit from the CLIP ensemble;
- ImageNet readiness-agent parity with CALM-v1;
- RLHF or policy-learning gains;
- SAM3 validation gains;
- a universal reduction in human feedback;
- a state-of-the-art claim across heterogeneous baselines.

## 1. Components and intended ablations

### 1.1 CALM-v1 baseline

The final v1 evaluation path is implemented in:

`CALM-v1/CALM-Continual_Adaptive_Learning_Model/final-evaluation/final-eval-other/`

Its core behavior is:

1. Load the frozen `openai/clip-vit-base-patch32` image encoder.
2. Encode support images into 512-dimensional embeddings.
3. Store embeddings by class in a flat memory.
4. Compute one arithmetic-mean prototype per class.
5. Normalize query embeddings and prototypes.
6. Predict the class with the largest cosine similarity.
7. In the separate readiness pipeline, calibrate confidence and either deploy
   or request feedback.

The v1 final-evaluation memory is deliberately simple:

```python
prototype_k = mean(stored_embeddings_for_class_k)
prediction = argmax(cosine(query_embedding, prototype_k))
```

No backbone parameters are updated during adaptation. The adaptation mechanism
is the addition of labeled embeddings to memory.

### 1.2 CALM-v2 changes

The v2 implementation is distributed across:

- `CALM-v2/episodic-memory/memory.py`
- `CALM-v2/episodic-memory/ensemble_calibration.py`
- `CALM-v2/episodic-memory/run_continual_learning.py`
- `CALM-v2/imagenet-eval/`
- `CALM-v2/final-eval-other/`
- `CALM-v2/readiness-agent/`

The intended changes are:

| Component | CALM-v1 | CALM-v2 |
|---|---|---|
| Image encoder | ViT-B/32 | ViT-B/32 + ViT-B/16 |
| Support memory | Flat full embeddings | Recent, medium, and old tiers |
| Prototype | Unweighted class mean | Weighted mean after decompression |
| Classifier | One cosine-similarity space | Fused similarities from two encoders |
| Confidence signal | Maximum calibrated probability | Ensemble probability and disagreement |
| Deployment rule | Binary threshold | Confidence/agreement tiers |
| Compression | None | 512→256 and 512→128 autoencoders |

These changes define the hypotheses to be tested:

1. **H1: More support examples improve prototype quality.**
2. **H2: A second frozen encoder improves confidence estimation or accuracy.**
3. **H3: Representation-consistent single-backbone scoring avoids the v2
   regression caused by cross-space similarity.**
4. **H4: Trained hierarchical compression preserves accuracy while reducing
   storage.**
5. **H5: More memory and/or better memory management reduce forgetting.**
6. **H6: Calibration changes the safety--autonomy trade-off, not merely the
   raw classifier accuracy.**

The current repository provides partial evidence for H1 and H6, and a
compound comparison relevant to H2--H4. It does not yet isolate H2, H3, or H4
with matched controls.

## 2. Evaluation protocols

### 2.1 ImageNet few-shot evaluation

The v1 and v2 ImageNet few-shot runners use the following nominal protocol:

- ImageNet validation split;
- 600 episodes;
- 5 classes per episode;
- 5 support images per class;
- 15 query images per class;
- support and query samples selected without replacement within an episode;
- frozen CLIP feature extraction;
- mean episode accuracy;
- interval computed as:

  `1.96 × std(episode accuracies) / sqrt(number of episodes)`.

The v1 runner is:

`CALM-v1/CALM-Continual_Adaptive_Learning_Model/final-evaluation/run_imagenet_evaluation.py`

The v2 runner is:

`CALM-v2/imagenet-eval/run_imagenet_evaluation.py`

The support and query images come from the same ImageNet validation
distribution. They are disjoint within an episode, but this is not a
train-support/test-query protocol. The paper should state this explicitly.

### 2.2 ImageNet continual evaluation

The nominal continual protocol is:

- ImageNet validation split;
- 10 sequential tasks;
- 100 classes per task;
- 5 support examples per class;
- evaluation on every previously introduced task after each new task;
- average accuracy computed from the diagonal of the task-by-task accuracy
  matrix;
- BWT computed as final accuracy on each previous task minus its accuracy
  immediately after introduction.

For a task accuracy matrix `A`, the implemented metrics are:

```text
Average accuracy = mean(A[i, i])

BWT = mean(A[T-1, i] - A[i, i]) for i = 0, ..., T-2
```

Here, a BWT value closer to zero is better. A negative value indicates
forgetting. The phrase “lower is better,” which appears in some v2 report
visualizations, is misleading for signed BWT.

The diagonal definition is important. Some paper prose describes average
accuracy as performance after all tasks, but the runners compute the mean of
the immediately post-task diagonal entries. The paper and ablation section
should use one definition consistently.

### 2.3 Other-dataset evaluation

The final v1 and v2 non-ImageNet runners support:

- CIFAR-10: 5 tasks, 2 classes per task;
- CIFAR-100: 10 tasks, 10 classes per task;
- STL-10: 5 tasks, 2 classes per task;
- SVHN: 5 tasks, 2 classes per task;
- FashionMNIST: 5 tasks, 2 classes per task.

The default final-other protocol is:

- official test split;
- 600 few-shot episodes;
- 5-way 5-shot;
- 15 queries per class;
- 5-shot continual memory;
- random seed 42.

Support and query data are sampled from the official test split for few-shot
evaluation. In continual evaluation, the support examples are also selected
from that split before the same split is used for evaluation. This is useful
as a controlled engineering comparison, but it is not a clean held-out
generalization protocol.

### 2.4 Readiness-agent evaluation

The v1 readiness path and v2 readiness path do not implement the same
experiment.

The v1 paper reports ImageNet deployment metrics, but the available v1
readiness evaluator primarily loads CIFAR-10, CIFAR-100, STL-10, or
FashionMNIST and uses a separate fixed-threshold pipeline.

The v2 readiness evaluator:

- builds memory from the training split;
- partitions the test split into validation and final-test subsets;
- calibrates a confidence model;
- calibrates a threshold on validation data;
- evaluates deployment decisions on the remaining test subset.

This is a useful diagnostic, but it is not directly comparable to the v1
ImageNet readiness result. The paper should not put those numbers in one
comparison table without identifying the dataset and protocol.

## 3. Results currently available

### 3.1 ImageNet v1 versus v2

The following values are reported in:

- `CALM-v2/imagenet-eval/report.md`;
- `CALM-v2/imagenet-eval/report.pdf`;
- `Paper/report.pdf`;
- `Paper/calm-neurips/neurips_2026.tex`;
- `Paper/final-report/CSE498R-Final-Report-CALM.tex`.

| Variant | Few-shot accuracy | Continual average accuracy | BWT |
|---|---:|---:|---:|
| CALM-v1 | 96.88% | 59.61% | -5.52 |
| CALM-v2, T=1.0 | 96.16% ± 0.32 | 55.16% | -5.97 |
| CALM-v2, T=0.7 | 95.94% ± 0.33 | 54.59% | -6.03 |

The corresponding differences are:

| Comparison | Few-shot change | Continual accuracy change | BWT change |
|---|---:|---:|---:|
| v2 T=1.0 minus v1 | -0.72 pp | -4.45 pp | -0.45 pp |
| v2 T=0.7 minus v1 | -0.94 pp | -5.02 pp | -0.51 pp |
| v2 T=0.7 minus v2 T=1.0 | -0.22 pp | -0.57 pp | -0.06 pp |

These are the strongest available v1--v2 comparisons, but they are
**compound variant comparisons**. The v2 result simultaneously changes the
encoder path, similarity fusion, memory class, and confidence machinery.
Therefore, the table supports the statement that the evaluated v2
configuration did not outperform v1 under this protocol. It does not support
the stronger statement that hierarchical compression caused the decrease.

#### Interpretation

The v2 few-shot result is close to v1, while the continual average is lower
and BWT is slightly more negative. A plausible implementation-level
explanation is that v2 stores support embeddings using the first available
encoder, normally ViT-B/32, but scores queries from both ViT-B/32 and
ViT-B/16 against those same prototypes.

In `CALM-v2/episodic-memory/ensemble_calibration.py`, each feature vector is
normalized and compared to the prototype tensor. Equal dimensionality does
not make two independently trained CLIP embedding spaces the same metric
space. The ViT-B/16 similarity can therefore dilute or overturn a
ViT-B/32-consistent decision. This is a testable explanation, not a measured
causal attribution.

### 3.2 Temperature sensitivity

The two ImageNet v2 settings provide a limited temperature sensitivity study:

- `T=1.0`: 96.16% ± 0.32 few-shot, 55.16% continual average, -5.97 BWT.
- `T=0.7`: 95.94% ± 0.33 few-shot, 54.59% continual average, -6.03 BWT.

The lower temperature did not help in the reported runs. The changes are
small relative to the larger v1--v2 configuration change, and only two
temperatures were tested. This should be described as a two-point sensitivity
comparison rather than a temperature ablation or an optimized calibration
study.

The temperature parameter is also used differently across code paths:

- in the final v2 runners, it changes the softmax applied to fused
  similarities;
- in the readiness-agent path, calibration models and threshold logic are
  applied separately;
- in the v2 readiness agent, passing `agreement` to `decide()` activates
  hard-coded 0.55 and 0.85 tier thresholds and bypasses
  `calibrated_threshold`.

That last behavior must be fixed or clearly documented before treating a v2
threshold value as the operating threshold of the final agent.

### 3.3 CIFAR-10 v1 versus v2 exploratory comparison

`CALM-v2/episodic-memory/report.md` reports the following comparison with the
original v1 episodic-memory runner:

| Setting | CALM-v1 | CALM-v2 | Change in v2 |
|---|---:|---:|---:|
| Zero-shot accuracy | 88.80% | 88.80% | 0.00 pp |
| 1-shot accuracy | 71.14% | 65.90% | -5.24 pp |
| 5-shot accuracy | 90.78% | 88.30% | -2.48 pp |
| 1-shot BWT | -24.25 | -21.86 | +2.39 pp |
| 5-shot BWT | -11.40 | -11.33 | +0.07 pp |

The report calls the 1-shot accuracy difference an improvement in one
visualization, but `65.90 - 71.14 = -5.24`; it is a decrease. The report
also describes BWT as “lower is better” in one place. Because less-negative
BWT is better, `-21.86` is better than `-24.25`, and `-11.33` is better than
`-11.40`.

This comparison is informative but not component-isolated. The v2 runner
uses the ensemble and hierarchical-memory classes together, while support
prototypes are still generated from the first encoder. It does not establish
that compression produced the BWT improvement.

### 3.4 New CALM-v1 multi-dataset results

`new-results-report.md` records a newer v1 final-evaluation suite. All rows use
the final-other protocol described above and a single seed (42).

| Dataset | Few-shot accuracy | 95% episode CI | Continual average | BWT |
|---|---:|---:|---:|---:|
| CIFAR-10 | 85.74% | ±0.49 pp | 87.60% | -9.14 |
| CIFAR-100 | 85.44% | ±0.64 pp | 60.08% | -11.11 |
| STL-10 | 96.75% | ±0.21 pp | 98.14% | -5.58 |
| SVHN | 26.77% | ±0.47 pp | 27.64% | -13.62 |
| FashionMNIST | 78.25% | ±0.76 pp | 76.85% | -11.79 |

These values are useful as v1 reference points for future v2 runs. They should
not be merged with the older exploratory values without a provenance note:

- the older v1 plotting script lists CIFAR-10 5-shot continual accuracy as
  90.78% and BWT as -11.40;
- it lists STL-10 5-shot continual accuracy as 98.21% and BWT as -2.33;
- the newer final-other report lists CIFAR-10 as 87.60% and -9.14, and STL-10
  as 98.14% and -5.58.

The likely explanation is that these values came from different runners or
protocols, but the repository does not preserve a complete manifest proving
that. The paper should select one finalized protocol and remove or label
legacy values.

### 3.5 New CALM-v2 multi-dataset results

The updated `new-results-report.md` records CALM-v2 ensemble runs for all five
non-ImageNet datasets. These runs use the final v2 `final-others.py` runner
with hierarchical episodic memory, ViT-B/32 plus ViT-B/16, five-shot support,
600 few-shot episodes, seed 42, and the official test split. The new results
are important for the ablation context because they test whether the ImageNet
pattern generalizes across domains.

| Dataset | Few-shot accuracy | 95% episode CI | Continual average | BWT |
|---|---:|---:|---:|---:|
| CIFAR-10 | 82.10% | ±0.51 pp | 81.02% | -8.90 |
| CIFAR-100 | 82.12% | ±0.74 pp | 54.35% | -11.70 |
| STL-10 | 94.57% | ±0.30 pp | 95.10% | -9.12 |
| SVHN | 26.21% | ±0.43 pp | 26.35% | -12.75 |
| FashionMNIST | 74.77% | ±0.74 pp | 69.90% | -10.16 |

Compared with the finalized v1 results, the differences are:

| Dataset | Few-shot change | Continual accuracy change | BWT change |
|---|---:|---:|---:|
| CIFAR-10 | -3.64 pp | -6.58 pp | +0.24 pp |
| CIFAR-100 | -3.32 pp | -5.73 pp | -0.59 pp |
| STL-10 | -2.18 pp | -3.04 pp | -3.54 pp |
| SVHN | -0.56 pp | -1.29 pp | +0.87 pp |
| FashionMNIST | -3.48 pp | -6.95 pp | +1.63 pp |

The pattern is clearer than the earlier CIFAR-10-only comparison:

- v2 is lower than v1 in few-shot accuracy on all five datasets;
- v2 is lower than v1 in continual average accuracy on all five datasets;
- BWT improves slightly on CIFAR-10, SVHN, and FashionMNIST;
- BWT becomes more negative on CIFAR-100 and STL-10.

This is useful evidence for a trade-off, but it is not evidence that either
the ensemble or compression independently caused the BWT changes. The runs
still change encoder count, similarity fusion, prototype implementation, and
memory class together. Moreover, the five-task and ten-task schedules remain
too short to trigger the current v2 memory migration thresholds. The results
should therefore be described as cross-dataset **compound variant
comparisons**.

The task matrices add a more specific failure picture:

- CIFAR-10: Task 2 loses 23.35 points; Task 4 is unchanged.
- CIFAR-100: Task 1 loses 31.50 points and Task 2 loses 21.90 points.
- STL-10: Task 2 loses 27.76 points.
- SVHN: the final Task 5 accuracy is 0.83%; the weak representation dominates
  the continual result.
- FashionMNIST: Task 4 loses 23.95 points.

These matrices are available in `new-results-report.md`, Sections 13--17.
They support per-task failure analysis, but they do not replace repeated-seed
statistics.

The new results also change the appropriate wording for the v2 conclusion.
The v2 configuration does not show a general accuracy gain. Its BWT behavior
is dataset-dependent, with the largest apparent retention benefit on
FashionMNIST (+1.63 pp) and the largest regression on STL-10 (-3.54 pp).
Because all runs use a single seed and shared official test support/query
source, these differences should be treated as directional observations until
the controlled ablations are complete.

### 3.6 CALM-v2 readiness calibration diagnostics

`CALM-v2/readiness-agent/report.md` reports a CIFAR-10 diagnostic with
10-shot memory, a ViT-B/32 plus ViT-B/16 ensemble, 2,000 validation samples,
and 8,000 final-test samples:

| Calibration | Deployment | Feedback | Deployed accuracy | FPR | TNR | Decision accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Temperature | 0.00% | 100.00% | N/A | 0.00% | 100.00% | 17.81% |
| Isotonic | 86.81% | 13.19% | 87.26% | 62.11% | 37.89% | 82.50% |
| Platt | 100.00% | 0.00% | 82.19% | 100.00% | 0.00% | 82.19% |

The v2 report identifies isotonic regression as the practical compromise, but
the FPR of 62.11% is high for a safety-oriented deployment claim. The
temperature and Platt results are useful failure modes: one rejects
everything, while the other deploys everything. Neither should be summarized
as evidence of reliable autonomous deployment.

There is also an implementation caveat. `evaluate_agent.py` passes ensemble
agreement to `ReadinessAgent.decide()`. In that branch, the agent applies
fixed tier thresholds and does not use the calibrated threshold selected by
`calibrate_threshold()`. The reported calibrated thresholds therefore do not
necessarily govern the reported final decisions.

### 3.7 Legacy v1 readiness and memory-size values

The older v1 paper source and `CALM-v1/Documentation/calm_viz_generator.py`
contain additional exploratory values. They include:

- ImageNet base accuracy: 96.88%;
- ImageNet deployment rate: 84.86%;
- ImageNet feedback rate: 15.14%;
- ImageNet deployed accuracy: 99.25%;
- ImageNet FPR: 12.01%;
- ImageNet TNR: 87.99%;
- ImageNet decision accuracy: 86.72%;
- reported error-rate reduction: 76%.

The plotting script also contains memory-size and calibration tables for
CIFAR-10, CIFAR-100, STL-10, and FashionMNIST. These values are not backed by
raw output files in the repository, and the current v1 readiness evaluator does
not reproduce all of them with the same defaults. They may be retained as
legacy reported results only after the authors recover their experiment
configuration and provenance.

The memory-size trend is still a useful hypothesis: increasing k-shot support
generally improves the prototype classifier and often improves
agent-filtered accuracy. For the paper, this should be rerun with fixed
support-selection seeds, the same held-out test set, and explicit memory
budgets.

## 4. What the code establishes about v2 compression

### 4.1 Implemented compression mechanism

`CALM-v2/episodic-memory/memory.py` defines:

- recent tier: 512 dimensions;
- medium tier: 256-dimensional code;
- old tier: 128-dimensional code;
- one autoencoder per compressed tier;
- decompression before prototype formation;
- optional importance weights;
- optional old-memory consolidation.

The autoencoders are ordinary trainable PyTorch modules:

```text
512 → 256 → compressed_dim → 256 → 512
```

The main v2 continual runner can train them with reconstruction loss when
`--train_compression` is provided. The final v2 ImageNet runners do not expose
that option. The final-other runner also constructs the hierarchical memory
but does not train its compression networks.

### 4.2 Why the reported short runs do not test compression

New examples are always inserted into the recent tier with the current episode
as their timestamp. Migration is checked every 50 calls to
`increment_episode()`:

- age ≤100: recent;
- age 101--500: medium;
- age >500: old.

The ImageNet continual runner calls `increment_episode()` once per task.
Consequently, a 10-task run reaches only episode 10. The CIFAR-style final
runner reaches only five or ten episodes depending on the dataset. In those
runs:

- no recent-to-medium migration occurs;
- no medium-to-old migration occurs;
- no old-memory consolidation occurs;
- the autoencoder is not used to reconstruct the stored examples.

Few-shot evaluation creates a fresh memory and does not advance the episode
counter, so it also does not exercise compression.

This is the most important ablation conclusion in the repository:

> The reported v2 ImageNet and final-other numbers are not measurements of
> compression-induced retention or compression-induced forgetting. They are
> primarily measurements of the v2 scoring path and its use of the
> hierarchical-memory interface before tier migration is activated.

The v2 episodic-memory report attributes performance changes to compression,
but the current scheduling code does not support that attribution for the
short final runs.

### 4.3 The 2.5x memory-capacity claim

The value “2.5x memory capacity” appears in the v2 reports, READMEs, and
implementation notes. The code does define lower-dimensional storage tiers,
but it does not:

- enforce a total byte budget;
- assign a fixed fraction of examples to each tier;
- measure serialized memory size;
- compare capacity at equal storage;
- measure reconstruction error by tier;
- sweep compression ratios;
- record how many examples are migrated;
- validate accuracy at the claimed capacity.

Therefore, 2.5x is an architectural estimate or design target, not an
empirical result. A paper-safe statement is:

> CALM-v2 implements tiered lower-dimensional storage intended to increase
> effective memory capacity; the current experiments do not quantify the
> resulting capacity under a fixed byte budget.

## 5. Recommended publication-grade ablation matrix

The minimum useful study should hold the dataset, split, task order,
support-selection seed, query set, shot count, episode count, and evaluation
code constant. The only changed variable should be the component named in the
row.

### 5.1 Accuracy and representation ablations

| ID | Encoder path | Memory path | Scoring path | Purpose |
|---|---|---|---|---|
| A0 | ViT-B/32 | Flat full memory | ViT-B/32 only | Final v1 baseline |
| A1 | ViT-B/32 | Hierarchical, migration disabled | ViT-B/32 only | Memory-class control without cross-encoder fusion |
| A2 | ViT-B/32 | Hierarchical, trained compression | ViT-B/32 only | Compression effect in one representation space |
| A3 | ViT-B/32 + ViT-B/16 | Flat or uncompressed paired memory | Per-backbone prototypes, then fused | Ensemble effect with metric consistency |
| A4 | ViT-B/32 + ViT-B/16 | Hierarchical paired memory | Per-backbone prototypes, then fused | Full representation-consistent v2 |
| A5 | ViT-B/32 + ViT-B/16 | Current shared B/32 memory | Current fused scoring | Reproduce the present v2 path |
| A6 | ViT-B/32 only | Hierarchical memory | Single-backbone flag | Existing non-ImageNet diagnostic |

The most important comparisons are:

- **A0 vs A1:** effect of changing the memory implementation while keeping
  scoring single-space and compression inactive;
- **A1 vs A2:** effect of trained hierarchical compression;
- **A1 vs A3:** effect of adding a second encoder under matched memory
  conditions;
- **A4 vs A5:** effect of maintaining per-backbone prototypes instead of
  comparing B/16 queries to B/32 prototypes;
- **A0 vs A4:** end-to-end v1 versus representation-consistent v2.

The current `--single_backbone` option in
`CALM-v2/final-eval-other/final-others.py` can implement a version of A6 for
the five non-ImageNet datasets. The ImageNet few-shot and continual runners do
not currently provide a corresponding single-backbone option, so an ImageNet
single-backbone result should not be implied by the current paper text.

### 5.2 Compression ablations

Compression must be tested separately from ensemble fusion:

1. Flat full-precision memory.
2. Hierarchical memory with migration disabled.
3. Hierarchical memory with migration and **trained** compression.
4. Hierarchical memory with migration and untrained compression.
5. 512→256 medium compression only.
6. 512→128 old compression only.
7. Optional prototype consolidation.

For every condition, report:

- few-shot accuracy;
- continual average accuracy;
- BWT;
- per-task accuracy matrix;
- compression reconstruction MSE;
- number of examples in each tier;
- stored bytes and bytes per example;
- decompression time;
- accuracy at equal byte budget;
- accuracy at equal number of retained examples.

The stream must be long enough to activate each tier. With the current
episode-based scheduler, a run needs at least approximately 150 episode
increments to observe medium-tier migration and more than 550 increments to
observe old-tier migration. A 5- or 10-task run is insufficient. There are two
reasonable choices:

- redefine the episode counter so each insertion or batch advances time;
- retain the current semantics and run an explicitly long synthetic or
  large-task stream.

The choice must be stated because it changes the meaning of “episode age.”

### 5.3 Memory-size ablations

Use the same encoder and scoring path while varying:

`k ∈ {1, 5, 10, 20}`

For each k:

- fix the support-selection seed;
- use the same query set;
- report few-shot accuracy;
- report continual average accuracy and BWT;
- report stored examples and bytes;
- report readiness metrics separately.

The old plotting script contains values for this direction, but those values
should be regenerated under the final runner before being used in the paper.

### 5.4 Temperature and fusion-weight ablations

The current two-point comparison is too small for a full temperature study.
Use a validation set to select or compare:

`T ∈ {0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0}`

For the ensemble, compare:

- B/32 only;
- B/16 only;
- uniform weights `(0.5, 0.5)`;
- validation-selected weights;
- per-class or per-dataset weights only if the paper has enough data to
  justify them.

The validation set must not be the final query set. Weight and temperature
selection should be performed once, then frozen for the test run.

### 5.5 Readiness-agent ablations

At minimum, compare:

- no calibration;
- temperature scaling;
- isotonic regression;
- Platt scaling;
- calibrated threshold without agreement;
- calibrated threshold plus agreement;
- fixed tier thresholds;
- a safety-constrained threshold selected for a target FPR.

Report:

- base accuracy;
- deployment rate;
- feedback rate;
- accuracy among deployed predictions;
- FPR among model errors;
- TNR among model errors;
- decision accuracy;
- coverage/risk curves;
- calibration error, preferably ECE or reliability diagrams;
- confidence intervals across repeated seeds.

The agent’s “decision accuracy” is not ordinary classification accuracy. It
counts a correct deployment and a correctly rejected model error as correct
decisions. This definition should be given in the paper.

## 6. Reproducibility requirements

The current repository stores most results in prose, LaTeX, PDFs, and plotting
scripts. It does not contain a complete set of raw CSV/JSON logs or
checkpoints for the reported tables. Before submission, every ablation row
should have a machine-readable record containing:

### Existing reproduction entry points

The following commands are the current entry points for the principal
comparisons. They require the corresponding datasets and CLIP checkpoints to
be available locally.

```powershell
# CALM-v1 final-other reference
cd E:\Uni\CSE498R\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation\final-eval-other
python final-others.py --datasetname cifar10 --device cuda

# CALM-v2 final-other ensemble comparison
cd E:\Uni\CSE498R\CALM\CALM-v2\final-eval-other
python final-others.py --datasetname cifar10 --device cuda --temperature 1.0

# CALM-v2 non-ImageNet single-backbone control
python final-others.py --datasetname cifar10 --device cuda --single_backbone

# CALM-v1 ImageNet few-shot reference
cd E:\Uni\CSE498R\CALM\CALM-v1\CALM-Continual_Adaptive_Learning_Model\final-evaluation
python run_imagenet_evaluation.py --imagenet_root <IMAGENET_ROOT> --k_shot 5 --n_way 5 --num_query 15 --num_episodes 600

# CALM-v2 ImageNet few-shot comparison
cd E:\Uni\CSE498R\CALM\CALM-v2\imagenet-eval
python run_imagenet_evaluation.py --imagenet_root <IMAGENET_ROOT> --k_shot 5 --n_way 5 --num_query 15 --num_episodes 600 --temperature 1.0

# CALM-v2 ImageNet continual comparison
python run_imagenet_continual.py --imagenet_root <IMAGENET_ROOT> --num_tasks 10 --k_shot 5 --temperature 1.0
```

The ImageNet commands do not set a random seed through their command-line
interfaces. A publication rerun should add explicit seed control or record
the seed in the launcher before comparing outputs. The final-other scripts do
accept `--seed 42`, which is the seed documented for the newer v1 results.

```text
experiment_id
git_commit
script_path
command
dataset
dataset_split
task_partition
model_checkpoint
model_variants
memory_type
compression_training_status
compression_dimensions
temperature
ensemble_weights
shot_count
episode_count
query_count
seed
device
few_shot_mean
few_shot_interval
continual_average
bwt
deployment_rate
feedback_rate
deployed_accuracy
fpr
tnr
decision_accuracy
memory_bytes
compression_mse
```

For each reported mean, use at least three independent seeds for continual
learning and report the mean and standard deviation or confidence interval.
The current “95% CI” in the episodic runners measures variation across
episodes in one run. It is not a multi-seed confidence interval.

The paper should also report:

- exact checkpoint identifiers;
- the actual number of loaded ensemble models;
- whether compression networks were trained;
- the data used to train compression networks;
- whether support and query samples share a dataset split;
- task order and class partition;
- whether the classifier and readiness agent use the same temperature;
- the exact definition of average accuracy and BWT.

## 7. Threats to validity

### 7.1 Shared support/query source split

The final-other experiments sample support and query images from the official
test split. Support and query are disjoint within an episode, but the protocol
does not measure transfer from training support to an untouched test set.
ImageNet uses the validation split in the same way. This can remain as the
chosen benchmark protocol, but it must not be described as a conventional
train/test generalization experiment.

### 7.2 Compound v1--v2 comparison

The v2 numbers change several variables simultaneously:

- encoder count;
- encoder representation space;
- similarity fusion;
- memory class;
- prototype weighting;
- readiness logic;
- potentially compression settings.

The observed v2 regression cannot be assigned to one of those changes without
the matrix in Section 5.

### 7.3 Compression is not activated in short runs

The current task runners increment memory age once per task. The reported
five- and ten-task runs do not reach the migration thresholds. Compression
claims must therefore be isolated in a longer controlled experiment.

### 7.4 Cross-encoder metric mismatch

The present v2 classifier stores prototypes from the first encoder and
compares them with query features from both encoders. ViT-B/32 and ViT-B/16
having 512 output dimensions is not evidence that their coordinates are
aligned. The v2 paper’s representation-consistency explanation is plausible,
but it remains a hypothesis until a per-backbone-prototype or single-backbone
control is run.

### 7.5 Readiness implementation mismatch

The v2 readiness agent has two decision modes:

- threshold mode uses `calibrated_threshold`;
- agreement mode uses hard-coded confidence and agreement cutoffs.

The current evaluator passes agreement, so the calibrated threshold reported
in the output may not control decisions. A paper result should be generated
after this path is made explicit and tested.

### 7.6 Baseline heterogeneity

The ImageNet paper table places CALM next to Prototypical Networks, CLIP
zero-shot, Tip-Adapter, iCaRL, and DualPrompt. These methods differ in
backbone, training data, trainable parameters, memory assumptions, and
evaluation protocol details. Those rows provide context, not a clean
component ablation or a universal ranking.

### 7.7 Legacy report drift

The repository contains multiple result generations, duplicated memory
implementations, and untracked final-evaluation directories. The older v1
plots, the current v1 final-other report, the v2 exploratory report, and the
NeurIPS source do not all describe the same experiment. Only values with a
recoverable protocol should enter the final paper table.

## 8. Paper-ready interpretation of the current evidence

The following paragraph is suitable as a conservative draft for the paper
until the missing ablations are run:

> We first compare the two implemented CALM variants under the ImageNet
> protocol. CALM-v1 obtains 96.88% few-shot accuracy, 59.61% continual
> average accuracy, and -5.52 BWT. The evaluated CALM-v2 ensemble obtains
> 96.16% ± 0.32% few-shot accuracy, 55.16% continual average accuracy, and
> -5.97 BWT at temperature 1.0. Lowering the temperature to 0.7 produces
> 95.94% ± 0.33%, 54.59%, and -6.03, respectively. Thus, the current v2
> configuration does not improve the reported task metrics over v1, although
> the few-shot gap is small. Because v2 changes the encoder path, score
> fusion, and memory implementation simultaneously, these results are a
> variant comparison rather than a component-isolated ablation. Code
> inspection identifies cross-encoder prototype/query inconsistency as a
> plausible explanation for part of the regression. In addition, the short
> continual runs do not reach the implemented memory-migration thresholds, so
> the reported results do not establish a causal benefit or cost from
> hierarchical compression.

If a complete matrix is run, replace the last four sentences with the measured
single-backbone, ensemble, and compression comparisons. Do not retain causal
language that is not supported by those controls.

## 9. Suggested final ablation section structure

For the conference paper, the ablation section should be organized as:

1. **Ablation setup:** dataset, split, task sequence, shots, query count,
   seeds, metrics, and fixed implementation details.
2. **Representation consistency:** v1, current v2 fusion, and per-backbone
   prototype/single-backbone controls.
3. **Memory design:** flat memory, hierarchical memory without compression,
   trained compression, compression-ratio sweep, and byte-level capacity.
4. **Memory size:** 1-, 5-, 10-, and 20-shot results under equal protocol.
5. **Confidence and readiness:** temperature, calibration method, agreement
   gating, deployment coverage, and risk.
6. **Failure analysis:** SVHN and any other domain where the frozen CLIP
   representation is poorly matched.
7. **Limitations:** shared evaluation split, seed count, missing raw logs, and
   any remaining implementation caveats.

The ablation narrative should emphasize trade-offs rather than claim that every
extension improves every metric. The existing evidence already supports a
useful systems result: frozen-feature prototype adaptation can be strong on
some domains, while adding ensemble and memory machinery without
representation-consistent controls can reduce continual performance.

## 10. Repository evidence index

### Core v1 implementation

- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/final-evaluation/final-eval-other/final-others.py`
- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/final-evaluation/final-eval-other/memory.py`
- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/final-evaluation/run_imagenet_evaluation.py`
- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/episodic-memory/run_continual_learning.py`
- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/dynamic-eval/clip-eval-knn.py`
- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/dynamic-eval/clip-eval-prot-net.py`

### Core v2 implementation

- `CALM-v2/episodic-memory/memory.py`
- `CALM-v2/episodic-memory/ensemble_calibration.py`
- `CALM-v2/episodic-memory/run_continual_learning.py`
- `CALM-v2/final-eval-other/final-others.py`
- `CALM-v2/imagenet-eval/run_imagenet_evaluation.py`
- `CALM-v2/imagenet-eval/run_imagenet_continual.py`

### Readiness and calibration

- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/readiness-agent/agent.py`
- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/readiness-agent/evaluate_agent.py`
- `CALM-v1/CALM-Continual_Adaptive_Learning_Model/readiness-agent-v2/`
- `CALM-v2/readiness-agent/agent.py`
- `CALM-v2/readiness-agent/evaluate_agent.py`
- `CALM-v2/readiness-agent/report.md`

### Reports and paper sources

- `new-results-report.md`
- `CALM-v2/imagenet-eval/report.md`
- `CALM-v2/episodic-memory/report.md`
- `CALM-v2/episodic-memory/IMPLEMENTATION_NOTES.md`
- `CALM-v2/episodic-memory/README.md`
- `CALM-v1/Documentation/calm_viz_generator.py`
- `CALM_Alternatives_Discussion_Report.md`
- `CALM-SAM3-DeepSeek OCR.md`
- `Paper/calm-neurips/neurips_2026.tex`
- `Paper/final-report/CSE498R-Final-Report-CALM.tex`
- `Paper/NOVELTY.md`
- `CALM-v1/Latex/calm.tex`

### Design-only material

The following materials describe planned work or expected outcomes rather than
completed ablations:

- SAM3 integration;
- optical or image-rendered memory compression;
- MAML, EWC, LLaVA, COCO-Text, and miniImageNet comparisons;
- RLHF reward modeling or policy learning;
- active-learning and conformal-prediction extensions;
- target values for autonomy, feedback rate, BWT, and memory capacity.

They may motivate future experiments, but they should not be cited as results
of the present CALM implementation.

## Bottom line

The repository currently supports these defensible conclusions:

1. CALM-v1 is a frozen CLIP prototype-memory system with strong reported
   ImageNet few-shot performance and non-catastrophic, but still negative, BWT.
2. CALM-v2 is an implemented extension with ensemble confidence and
   hierarchical-memory machinery.
3. In the reported ImageNet comparison, the v2 configuration is slightly worse
   in few-shot accuracy and clearly worse in continual average accuracy than
   v1.
4. Across the five new non-ImageNet runs, v2 is lower in few-shot and
   continual average accuracy on every dataset. BWT is modestly less negative
   on CIFAR-10, SVHN, and FashionMNIST, but more negative on CIFAR-100 and
   STL-10. These comparisons do not isolate the ensemble from compression.
5. The current short continual runs do not activate v2 tier migration, so the
   reported results cannot establish compression-driven retention or capacity.
6. The most important missing experiment is a representation-consistent
   single-backbone/per-backbone-prototype ablation, followed by a long,
   trained-compression study with byte-level accounting.
7. Readiness results are dataset-specific and contain a threshold-handling
   inconsistency that must be resolved before making deployment claims.

Until those experiments are completed, the ablation section should present the
current v1--v2 results as an honest systems comparison and identify causal
component ablations as ongoing work rather than retroactively assigning credit
to individual mechanisms.
