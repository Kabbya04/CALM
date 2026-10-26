# CALM-v2 readiness agent

The original evaluator is `evaluate_agent.py`. For the review protocol
(k=5, v1 and v2 on the same split, coverage / deferral / deployed error)
use `CALM/experiments/run_readiness.py` instead.

```powershell
python evaluate_agent.py --dataset cifar10 --k_shot 5 --seed 42 --target_deployment 0.75
```

`--use_disagreement` is not on this script. Passing agreement into
`ReadinessAgent.decide()` replaces the calibrated threshold with fixed
0.55 / 0.85 gates. The experiments runner keeps that path behind a flag.
