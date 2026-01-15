# File: memory.py
# Import HierarchicalEpisodicMemory from parent episodic-memory module
# This allows readiness-agent to use the enhanced hierarchical memory

import sys
import os

# Add parent directory to path to import from episodic-memory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'episodic-memory'))

from memory import HierarchicalEpisodicMemory

# Re-export for convenience
__all__ = ['HierarchicalEpisodicMemory']
