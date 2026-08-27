"""Tests for the Figure-4 decoder-output cache boundary."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from statespacecheck_paper import figure04_cache
from statespacecheck_paper.figure04_cache import (
    FIGURE04_CACHE_SCHEMA_VERSION,
    Figure4Paths,
    compute_figure04_cache_fingerprint,
    load_figure04_cache,
    save_figure04_cache,
)
from statespacecheck_paper.real_data_analysis import Figure4Config


def _payload() -> dict[str, Any]:
    """A joblib-serializable decode payload matching the cache keys."""
    return {
        "continuous_results": np.zeros(3),
        "contfrag_results": np.ones(3),
        "continuous_diagnostics": {"tag": "cont"},
        "contfrag_diagnostics": {"tag": "cf"},
        "spike_counts": np.zeros((8, 2), dtype=np.int64),
        "place_field_peaks": np.zeros(2),
        "diagnostic_place_fields": np.zeros((2, 4)),
        "diagnostic_position_bins": np.arange(4.0),
    }


def test_cache_path_uses_injected_identifiers(tmp_path: Path) -> None:
    paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="epoch_x")
    assert paths.cache_path == tmp_path / "intermediates" / "epoch_x_fig4_cache.joblib"


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "intermediates" / "c.joblib"
    payload = _payload()
    save_figure04_cache(path, "fp1", payload)
    loaded = load_figure04_cache(path, "fp1")
    assert loaded is not None
    assert set(loaded.keys()) == set(payload.keys())
    np.testing.assert_array_equal(loaded["contfrag_results"], payload["contfrag_results"])


def test_miss_when_absent(tmp_path: Path) -> None:
    assert load_figure04_cache(tmp_path / "nope.joblib", "fp") is None


def test_miss_on_fingerprint_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    save_figure04_cache(path, "fp1", _payload())
    assert load_figure04_cache(path, "fp2") is None


def test_miss_on_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    joblib.dump(
        {"schema_version": FIGURE04_CACHE_SCHEMA_VERSION + 1, "fingerprint": "fp", **_payload()},
        path,
    )
    assert load_figure04_cache(path, "fp") is None


def test_miss_on_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    joblib.dump([1, 2, 3], path)
    assert load_figure04_cache(path, "fp") is None


def test_miss_on_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    path.write_bytes(b"not a joblib file")
    assert load_figure04_cache(path, "fp") is None


def test_miss_on_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    wrapper = {"schema_version": FIGURE04_CACHE_SCHEMA_VERSION, "fingerprint": "fp", **_payload()}
    del wrapper["spike_counts"]
    joblib.dump(wrapper, path)
    assert load_figure04_cache(path, "fp") is None


def test_miss_on_extra_key(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    wrapper = {
        "schema_version": FIGURE04_CACHE_SCHEMA_VERSION,
        "fingerprint": "fp",
        "unexpected": 1,
        **_payload(),
    }
    joblib.dump(wrapper, path)
    assert load_figure04_cache(path, "fp") is None


def test_save_rejects_missing_payload_key(tmp_path: Path) -> None:
    payload = _payload()
    del payload["spike_counts"]
    with pytest.raises(ValueError, match="payload keys"):
        save_figure04_cache(tmp_path / "c.joblib", "fp", payload)


def test_save_rejects_extra_payload_key(tmp_path: Path) -> None:
    payload = _payload()
    payload["unexpected"] = 1
    with pytest.raises(ValueError, match="payload keys"):
        save_figure04_cache(tmp_path / "c.joblib", "fp", payload)


def test_fingerprint_changes_with_config_and_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="epoch_x")
    config = Figure4Config()
    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "1.0.0")
    fp1 = compute_figure04_cache_fingerprint(config, paths)
    assert compute_figure04_cache_fingerprint(config, paths) == fp1  # deterministic

    changed = dataclasses.replace(config, movement_var=config.movement_var + 1.0)
    assert compute_figure04_cache_fingerprint(changed, paths) != fp1

    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "2.0.0")
    assert compute_figure04_cache_fingerprint(config, paths) != fp1
