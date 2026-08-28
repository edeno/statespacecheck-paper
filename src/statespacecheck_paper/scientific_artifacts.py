"""Deterministic machine-readable artifacts for manuscript-facing results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np

FlagDirection = Literal["below", "above"]


class ScientificSourceProvenance(TypedDict):
    """Stable identities for the analysis source and locked environment."""

    statespacecheck_paper_version: str
    source_tree_sha256: str
    uv_lock_sha256: str


_INCLUSIVE_FLAG_OPERATORS: dict[FlagDirection, str] = {
    "below": "less_than_or_equal",
    "above": "greater_than_or_equal",
}


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_source_provenance(repo_root: Path | None = None) -> ScientificSourceProvenance:
    """Identify the source and locked dependency graph that produced an artifact.

    The source digest covers every Python module under ``src/statespacecheck_paper``
    in relative-path order. It deliberately excludes generated artifacts, tests,
    timestamps, and absolute paths, so it is stable across clean checkouts while
    changing whenever the package implementation changes.
    """
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    package_root = root / "src" / "statespacecheck_paper"
    source_files = sorted(package_root.rglob("*.py"))
    if not source_files:
        raise FileNotFoundError(f"No scientific source files found under {package_root}")

    source_digest = hashlib.sha256()
    for source_path in source_files:
        relative_path = source_path.relative_to(root).as_posix()
        source_digest.update(relative_path.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(source_path.read_bytes())
        source_digest.update(b"\0")

    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Locked dependency graph not found: {lock_path}")

    return {
        "statespacecheck_paper_version": version("statespacecheck-paper"),
        "source_tree_sha256": source_digest.hexdigest(),
        "uv_lock_sha256": _sha256_file(lock_path),
    }


def inclusive_flag_rules(
    thresholds: Mapping[str, float],
    directions: Mapping[str, FlagDirection],
) -> dict[str, dict[str, float | str]]:
    """Bind each diagnostic threshold to its exact inclusive comparison rule."""
    missing_thresholds = set(directions) - set(thresholds)
    missing_directions = set(thresholds) - set(directions)
    if missing_thresholds or missing_directions:
        raise ValueError(
            "Flag thresholds and directions must name the same metrics; "
            f"missing thresholds for {sorted(missing_thresholds)}, "
            f"missing directions for {sorted(missing_directions)}."
        )
    return {
        metric: {
            "comparison": _INCLUSIVE_FLAG_OPERATORS[direction],
            "threshold": float(thresholds[metric]),
        }
        for metric, direction in directions.items()
    }


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
