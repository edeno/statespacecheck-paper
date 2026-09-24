"""Tests for the Figure-4 decoder-output cache boundary."""

from __future__ import annotations

import dataclasses
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from statespacecheck_paper import figure04_cache
from statespacecheck_paper.figure04_cache import (
    FIGURE04_CACHE_SCHEMA_VERSION,
    FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
    Figure4CacheProvenance,
    Figure4Paths,
    compute_figure04_cache_fingerprint,
    compute_figure04_cache_provenance,
    compute_figure04_diagnostics_fingerprint,
    executable_source_digest,
    load_figure04_cache,
    load_figure04_diagnostics_cache,
    save_figure04_cache,
    save_figure04_diagnostics_cache,
)
from statespacecheck_paper.figure04_decoder import Figure4Config, Figure4DiagnosticsConfig
from statespacecheck_paper.load_local_data import EXPORT_FILE_SUFFIXES


def _payload() -> dict[str, Any]:
    """A joblib-serializable *decode* payload matching the decode-cache keys."""
    return {
        "continuous_results": np.zeros(3),
        "contfrag_results": np.ones(3),
        "spike_counts": np.zeros((8, 2), dtype=np.int64),
        "place_field_peaks": np.zeros(2),
        "diagnostic_place_fields": np.zeros((2, 4)),
        "diagnostic_position_bins": np.arange(4.0),
    }


def _diagnostics_payload() -> dict[str, Any]:
    """A joblib-serializable *diagnostics* payload matching the diagnostics-cache keys."""
    return {"continuous_diagnostics": {"tag": "cont"}, "contfrag_diagnostics": {"tag": "cf"}}


def test_cache_path_uses_injected_identifiers(tmp_path: Path) -> None:
    paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="epoch_x")
    assert paths.cache_path == tmp_path / "intermediates" / "epoch_x_fig4_cache.joblib"
    assert (
        paths.diagnostics_cache_path
        == tmp_path / "intermediates" / "epoch_x_fig4_diagnostics.joblib"
    )


def test_missing_decoder_version_rejects_unknown_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(figure04_cache, "version", missing_version)
    with pytest.raises(RuntimeError, match="Cannot fingerprint Figure 4"):
        figure04_cache._installed_non_local_detector_version()


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "intermediates" / "c.joblib"
    payload = _payload()
    save_figure04_cache(path, "fp1", payload)
    loaded = load_figure04_cache(path, "fp1")
    assert loaded is not None
    assert set(loaded.keys()) == set(payload.keys())
    np.testing.assert_array_equal(loaded["contfrag_results"], payload["contfrag_results"])
    assert not path.with_name(path.name + ".tmp").exists()


def test_legacy_bundle_with_embedded_diagnostics_serves_decode_payload(tmp_path: Path) -> None:
    """A pre-split bundle (decode + diagnostics keys) is accepted as a decode cache.

    Its embedded diagnostics are dropped from the returned payload: the
    separate diagnostics bundle is the only diagnostics source.
    """
    path = tmp_path / "c.joblib"
    joblib.dump(
        {
            "schema_version": FIGURE04_CACHE_SCHEMA_VERSION,
            "fingerprint": "fp",
            **_payload(),
            **_diagnostics_payload(),
        },
        path,
    )
    loaded = load_figure04_cache(path, "fp")
    assert loaded is not None
    assert set(loaded.keys()) == set(_payload().keys())


def test_diagnostics_cache_round_trip_and_misses(tmp_path: Path) -> None:
    path = tmp_path / "intermediates" / "d.joblib"
    save_figure04_diagnostics_cache(path, "fp1", "dfp1", _diagnostics_payload())
    loaded = load_figure04_diagnostics_cache(path, "fp1", "dfp1")
    assert loaded is not None
    assert set(loaded.keys()) == set(_diagnostics_payload().keys())
    # Both fingerprints gate the bundle: a decode change or a diagnostics change misses.
    assert load_figure04_diagnostics_cache(path, "fp2", "dfp1") is None
    assert load_figure04_diagnostics_cache(path, "fp1", "dfp2") is None
    assert load_figure04_diagnostics_cache(tmp_path / "nope.joblib", "fp1", "dfp1") is None
    joblib.dump(
        {
            "diagnostics_schema_version": FIGURE04_DIAGNOSTICS_SCHEMA_VERSION + 1,
            "fingerprint": "fp1",
            "diagnostics_fingerprint": "dfp1",
            **_diagnostics_payload(),
        },
        path,
    )
    assert load_figure04_diagnostics_cache(path, "fp1", "dfp1") is None


def test_diagnostics_save_rejects_wrong_payload_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="payload keys"):
        save_figure04_diagnostics_cache(tmp_path / "d.joblib", "fp", "dfp", _payload())


def test_diagnostics_fingerprint_tracks_config_and_executable_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = compute_figure04_diagnostics_fingerprint(Figure4DiagnosticsConfig())
    assert base == compute_figure04_diagnostics_fingerprint(Figure4DiagnosticsConfig())
    changed = compute_figure04_diagnostics_fingerprint(Figure4DiagnosticsConfig(hpd_coverage=0.8))
    assert changed != base
    monkeypatch.setattr(figure04_cache, "_installed_statespacecheck_version", lambda: "9.9.9")
    assert compute_figure04_diagnostics_fingerprint(Figure4DiagnosticsConfig()) != base

    # The executable-source digest ignores docstrings and comments but not code.
    a = tmp_path / "a.py"
    a.write_text(
        '"""Doc."""\n\n\ndef f(x):\n    """Inner doc."""\n    # comment\n    return x + 1\n'
    )
    digest_a = executable_source_digest((a,))
    a.write_text('"""Other doc."""\n\n\ndef f(x):\n    """Changed."""\n    return x + 1\n')
    assert executable_source_digest((a,)) == digest_a
    a.write_text('"""Doc."""\n\n\ndef f(x):\n    return x + 2\n')
    assert executable_source_digest((a,)) != digest_a


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

    changed = dataclasses.replace(
        config,
        provenance=dataclasses.replace(
            config.provenance, movement_var=config.provenance.movement_var + 1.0
        ),
    )
    assert compute_figure04_cache_fingerprint(changed, paths) != fp1

    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "2.0.0")
    assert compute_figure04_cache_fingerprint(config, paths) != fp1


def test_fingerprint_unchanged_when_block_size_changes(tmp_path: Path) -> None:
    # block_size (Figure4ExecutionConfig) is a performance-only knob that leaves
    # the decode result identical, so it must NOT be hashed into the fingerprint:
    # changing it must not invalidate a cached decode.
    paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="epoch_x")
    config = Figure4Config()
    fp = compute_figure04_cache_fingerprint(config, paths)

    changed = dataclasses.replace(
        config,
        execution=dataclasses.replace(config.execution, block_size=config.execution.block_size * 2),
    )
    assert compute_figure04_cache_fingerprint(changed, paths) == fp


def test_fingerprint_changes_when_export_file_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replacing an export under the same epoch must invalidate the cache: the
    # fingerprint hashes the file contents, not just ``animal_date_epoch``.
    paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="epoch_x")
    config = Figure4Config()
    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "1.0.0")

    (suffix,) = EXPORT_FILE_SUFFIXES
    export = tmp_path / f"epoch_x{suffix}"
    export.write_bytes(b"original")
    fp_original = compute_figure04_cache_fingerprint(config, paths)
    assert compute_figure04_cache_fingerprint(config, paths) == fp_original  # deterministic

    export.write_bytes(b"REPLACED with different data")
    assert compute_figure04_cache_fingerprint(config, paths) != fp_original


def test_cache_provenance_serializes_complete_path_independent_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="epoch_x")
    for suffix in EXPORT_FILE_SUFFIXES:
        (tmp_path / f"epoch_x{suffix}").write_bytes(suffix.encode())
    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "1.2.3")

    provenance = compute_figure04_cache_provenance(Figure4Config(), paths)
    payload = provenance.artifact_payload()

    assert provenance.fingerprint_sha256 == compute_figure04_cache_fingerprint(
        Figure4Config(), paths
    )
    assert payload["schema_version"] == FIGURE04_CACHE_SCHEMA_VERSION
    assert payload["non_local_detector_version"] == "1.2.3"
    assert set(payload["export_file_sha256"]) == {
        f"epoch_x{suffix}" for suffix in EXPORT_FILE_SUFFIXES
    }
    assert payload["diagnostics_schema_version"] == FIGURE04_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["diagnostics_fingerprint_sha256"] == compute_figure04_diagnostics_fingerprint(
        Figure4DiagnosticsConfig()
    )
    assert payload["diagnostics_config"] == {
        "hpd_coverage": 0.95,
        "event_selection": "all_spikes_in_recording",
    }
    assert str(tmp_path) not in repr(payload)


def test_cache_provenance_rejects_missing_canonical_input_checksum() -> None:
    provenance = Figure4CacheProvenance(
        fingerprint_sha256="f" * 64,
        schema_version=FIGURE04_CACHE_SCHEMA_VERSION,
        animal_date_epoch="epoch_x",
        export_checksums=tuple((suffix, None) for suffix in EXPORT_FILE_SUFFIXES),
        non_local_detector_version="1.2.3",
        diagnostics_fingerprint_sha256="d" * 64,
        diagnostics_schema_version=FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
        statespacecheck_version="0.1.0",
        diagnostics_config=Figure4DiagnosticsConfig(),
    )
    with pytest.raises(ValueError, match="requires every exported input"):
        provenance.artifact_payload()
