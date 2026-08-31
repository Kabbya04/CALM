"""
Local loader for CALM-v2's ensemble implementation.

The actual EnsembleCLIPModels source remains in episodic-memory/.
"""

import importlib.util
import os
from types import ModuleType


def _load_core_ensemble() -> ModuleType:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    calm_v2_root = os.path.dirname(this_dir)
    path = os.path.join(calm_v2_root, "episodic-memory", "ensemble_calibration.py")

    if not os.path.exists(path):
        raise FileNotFoundError(f"CALM-v2 ensemble implementation was not found: {path}")

    spec = importlib.util.spec_from_file_location("calm_v2_final_eval_ensemble_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CALM-v2 ensemble implementation: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core_ensemble = _load_core_ensemble()
EnsembleCLIPModels = _core_ensemble.EnsembleCLIPModels  # type: ignore[attr-defined]

