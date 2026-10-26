# Hierarchical episodic memory

Canonical v2 memory (`memory.py`) and CLIP ensemble (`ensemble_calibration.py`).

Tiers: recent 512-d, medium 256-d, old 128-d. Age thresholds default to
`tau_recent=100` and `tau_medium=500`. New examples start in recent.
`increment_episode()` moves them when age crosses a threshold.

`run_continual_learning.py` is the older CIFAR trainer. Paper numbers
should come from `../imagenet-eval/` and `../final-eval-other/`.

```powershell
python run_continual_learning.py --dataset cifar10 --train_compression --compression_epochs 50
```

Instrumentation on `HierarchicalEpisodicMemory`:

- `tier_counts()`, `stored_bytes()`, `reconstruction_mse()`, `memory_stats()`
- `flat_memory=True` — no migration
- `max_bytes=N` — drop oldest codes when over budget
- `save_codecs` / `load_codecs`
