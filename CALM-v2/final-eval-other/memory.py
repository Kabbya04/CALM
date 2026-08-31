"""
Local CALM-v2 memory import for the final multi-dataset evaluation.

The implementation remains the shared CALM-v2 hierarchical memory; this
wrapper keeps the final-evaluation folder self-contained at import time.
"""

import importlib.util
import os
from types import ModuleType


def _load_core_memory() -> ModuleType:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    calm_v2_root = os.path.dirname(this_dir)
    path = os.path.join(calm_v2_root, "episodic-memory", "memory.py")

    if not os.path.exists(path):
        raise FileNotFoundError(f"CALM-v2 memory implementation was not found: {path}")

    spec = importlib.util.spec_from_file_location("calm_v2_final_eval_memory_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CALM-v2 memory implementation: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core_memory = _load_core_memory()
HierarchicalEpisodicMemory = _core_memory.HierarchicalEpisodicMemory  # type: ignore[attr-defined]

