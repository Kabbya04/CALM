"""
Local episodic memory module for ImageNet evaluation.

This file is intentionally thin: it loads and re-exports the
`HierarchicalEpisodicMemory` implementation from the main
CALM-v2 episodic-memory package so that:

- `run_imagenet_evaluation.py` can simply do:
      from memory import HierarchicalEpisodicMemory
- The ImageNet eval code does not depend on Python import
  side effects from running inside the episodic-memory folder.
"""

import importlib.util
import os
from types import ModuleType


def _load_episodic_memory_module() -> ModuleType:
    """
    Dynamically load the core episodic-memory `memory.py` from CALM-v2.

    We avoid a normal `import memory` here to prevent importing this
    local file recursively. Instead we load by absolute file path and
    give it a unique module name.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    calm_v2_root = os.path.dirname(this_dir)
    episodic_memory_path = os.path.join(calm_v2_root, "episodic-memory", "memory.py")

    if not os.path.exists(episodic_memory_path):
        raise FileNotFoundError(
            f"Expected episodic memory file at '{episodic_memory_path}', "
            "but it was not found. Ensure CALM-v2/episodic-memory/memory.py exists."
        )

    module_name = "calm_v2_episodic_memory_module"
    spec = importlib.util.spec_from_file_location(module_name, episodic_memory_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {episodic_memory_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_episodic_memory_module = _load_episodic_memory_module()

# Re-export the hierarchical episodic memory class so callers in this
# folder can import it directly from `memory`.
HierarchicalEpisodicMemory = _episodic_memory_module.HierarchicalEpisodicMemory  # type: ignore[attr-defined]

