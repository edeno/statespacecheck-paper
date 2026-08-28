"""Tests for deterministic machine-readable manuscript result artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from statespacecheck_paper.scientific_artifacts import write_json_artifact


def test_write_json_artifact_is_deterministic_and_converts_numpy(tmp_path) -> None:
    path = tmp_path / "summary.json"
    payload = {
        "z": np.array([1.5, 2.5]),
        "a": {"count": np.int64(3)},
    }

    write_json_artifact(path, payload)
    first = path.read_bytes()
    write_json_artifact(path, payload)

    assert path.read_bytes() == first
    assert first.endswith(b"\n")
    assert json.loads(first) == {"a": {"count": 3}, "z": [1.5, 2.5]}


def test_write_json_artifact_rejects_nonfinite_reported_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_artifact(tmp_path / "bad.json", {"reported_mean": np.nan})
