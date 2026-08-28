"""Tests for deterministic machine-readable manuscript result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from statespacecheck_paper.scientific_artifacts import (
    inclusive_flag_rules,
    scientific_source_provenance,
    write_json_artifact,
)


def test_write_json_artifact_is_deterministic_and_converts_numpy(tmp_path: Path) -> None:
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


def test_write_json_artifact_rejects_nonfinite_reported_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_artifact(tmp_path / "bad.json", {"reported_mean": np.nan})


def test_inclusive_flag_rules_store_exact_comparison_semantics() -> None:
    assert inclusive_flag_rules(
        {"low": 0.05, "high": 2.0},
        {"low": "below", "high": "above"},
    ) == {
        "low": {"comparison": "less_than_or_equal", "threshold": 0.05},
        "high": {"comparison": "greater_than_or_equal", "threshold": 2.0},
    }


def test_inclusive_flag_rules_reject_mismatched_metrics() -> None:
    with pytest.raises(ValueError, match="same metrics"):
        inclusive_flag_rules({"hpd": 0.05}, {"kl": "above"})


def test_scientific_source_provenance_tracks_source_and_lockfile(tmp_path: Path) -> None:
    package = tmp_path / "src" / "statespacecheck_paper"
    package.mkdir(parents=True)
    source = package / "analysis.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")

    original = scientific_source_provenance(tmp_path)
    assert original == scientific_source_provenance(tmp_path)
    assert len(original["source_tree_sha256"]) == 64
    assert len(original["uv_lock_sha256"]) == 64

    source.write_text("VALUE = 2\n", encoding="utf-8")
    source_changed = scientific_source_provenance(tmp_path)
    assert source_changed["source_tree_sha256"] != original["source_tree_sha256"]
    assert source_changed["uv_lock_sha256"] == original["uv_lock_sha256"]

    lockfile.write_text("version = 2\n", encoding="utf-8")
    lock_changed = scientific_source_provenance(tmp_path)
    assert lock_changed["uv_lock_sha256"] != original["uv_lock_sha256"]


def test_scientific_source_provenance_is_line_ending_independent(tmp_path: Path) -> None:
    """CRLF and LF checkouts of identical content must produce identical digests.

    Guards against the Windows-only regression where ``text=auto`` checks files
    out as CRLF, so hashing raw bytes yielded a different fingerprint per
    platform and broke the summary round-trip tests on Windows only.
    """
    package = tmp_path / "src" / "statespacecheck_paper"
    package.mkdir(parents=True)
    # Write raw bytes so the line endings are exactly as specified, mimicking a
    # checkout rather than relying on the platform's text-mode translation.
    (package / "analysis.py").write_bytes(b"VALUE = 1\nOTHER = 2\n")
    (tmp_path / "uv.lock").write_bytes(b"version = 1\npackage = 'x'\n")
    lf = scientific_source_provenance(tmp_path)

    (package / "analysis.py").write_bytes(b"VALUE = 1\r\nOTHER = 2\r\n")
    (tmp_path / "uv.lock").write_bytes(b"version = 1\r\npackage = 'x'\r\n")
    crlf = scientific_source_provenance(tmp_path)

    assert crlf["source_tree_sha256"] == lf["source_tree_sha256"]
    assert crlf["uv_lock_sha256"] == lf["uv_lock_sha256"]
