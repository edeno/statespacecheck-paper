"""Deterministic machine-readable artifacts for manuscript-facing results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: object) -> Any:
    """Convert common scientific Python values into strict JSON values."""
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON artifact mapping keys must be strings; got {key!r}")
            converted[key] = _jsonable(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON artifact value: {type(value).__name__}")


def write_json_artifact(path: Path | str, payload: Mapping[str, object]) -> Path:
    """Write ``payload`` as sorted, indented, standards-compliant JSON.

    The file contains no timestamp or other run-specific metadata, so identical
    scientific inputs produce byte-identical output. ``allow_nan=False`` makes
    undefined reported values fail loudly instead of emitting non-standard JSON.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _jsonable(payload),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    output_path.write_text(f"{text}\n", encoding="utf-8")
    return output_path
