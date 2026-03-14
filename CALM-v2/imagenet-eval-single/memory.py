"""
Local episodic memory wrapper for the single-backbone ImageNet evaluation.

This re-uses the main CALM-v2 `HierarchicalEpisodicMemory` implementation
so that experiments are comparable, but keeps imports local to this folder.
"""

import importlib.util
import os
from types import ModuleType


def _load_episodic_memory_module() -> ModuleType:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    calm_v2_root = os.path.dirname(this_dir)
    episodic_memory_path = os.path.join(calm_v2_root, "episodic-memory", "memory.py")

    if not os.path.exists(episodic_memory_path):
        raise FileNotFoundError(
            f"Expected episodic memory file at '{episodic_memory_path}', "
            "but it was not found. Ensure CALM-v2/episodic-memory/memory.py exists."
        )

    module_name = "calm_v2_episodic_memory_module_single_backbone"
    spec = importlib.util.spec_from_file_location(module_name, episodic_memory_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {episodic_memory_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_episodic_memory_module = _load_episodic_memory_module()

HierarchicalEpisodicMemory = _episodic_memory_module.HierarchicalEpisodicMemory  # type: ignore[attr-defined]

