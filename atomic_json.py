"""Crash-safe JSON writes shared by CALM runners (tmp + replace)."""

from __future__ import annotations

import json
import os
from typing import Any


def atomic_write_json(path: str, payload: Any) -> str:
    path = os.path.abspath(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path
