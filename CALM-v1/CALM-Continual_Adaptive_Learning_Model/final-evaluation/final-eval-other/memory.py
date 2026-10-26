"""Re-export the canonical CALM-v1 episodic memory."""

import importlib.util
import os


def _load():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory.py")
    spec = importlib.util.spec_from_file_location("calm_v1_memory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load()
EpisodicMemory = _module.EpisodicMemory
__all__ = ["EpisodicMemory"]
