"""Tests for the Figure-4 decoder-output cache boundary."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from statespacecheck_paper import (
    figure04_cache,
    figure04_diagnostics,
    figure04_fit,
    figure04_workflow,
)
from statespacecheck_paper.figure04_cache import (
    FIGURE04_DECODE_SCHEMA_VERSION,
    FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
    Figure4Paths,
    compute_figure04_cache_provenance,
    compute_figure04_decode_identity,
    compute_figure04_diagnostics_fingerprint,
    compute_figure04_diagnostics_identity,
    executable_source_digest,
    load_figure04_decode_cache,
    load_figure04_diagnostics_cache,
    save_figure04_decode_cache,
    save_figure04_diagnostics_cache,
)
from statespacecheck_paper.figure04_decoder import (
    Figure4Config,
    Figure4DecoderConfig,
    Figure4DiagnosticsConfig,
)
from statespacecheck_paper.figure04_input import INPUT_FILE_SUFFIX, input_file_path
from statespacecheck_paper.paths import FIGURE04_SUMMARY_PATH
from statespacecheck_paper.reported_values import load_summary


def _payload() -> dict[str, Any]:
    """A joblib-serializable *decode* payload matching the decode-cache keys."""
    return {
        "continuous_results": np.zeros(3),
        "continuous_fragmented_results": np.ones(3),
        "spike_counts": np.zeros((8, 2), dtype=np.int64),
        "place_field_peaks": np.zeros(2),
        "diagnostic_place_fields": np.zeros((2, 4)),
        "diagnostic_position_bins": np.arange(4.0),
    }


def _diagnostics_payload() -> dict[str, Any]:
    """A joblib-serializable *diagnostics* payload matching the diagnostics-cache keys."""
    return {
        "continuous_diagnostics": {"tag": "cont"},
        "continuous_fragmented_diagnostics": {"tag": "cf"},
    }


def test_cache_path_uses_injected_identifiers(tmp_path: Path) -> None:
    paths = Figure4Paths(data_path=tmp_path, animal_date_epoch="epoch_x")
    assert paths.decode_cache_path == tmp_path / "intermediates" / "epoch_x_figure04_decode.joblib"
    assert (
        paths.diagnostics_cache_path
        == tmp_path / "intermediates" / "epoch_x_figure04_diagnostics.joblib"
    )


def _paths_with_input(data_path: Path, epoch: str = "epoch_x") -> Figure4Paths:
    """Paths of an epoch whose input file exists (the fingerprints hash its bytes)."""
    (data_path / f"{epoch}{INPUT_FILE_SUFFIX}").write_bytes(b"input")
    return Figure4Paths(data_path=data_path, animal_date_epoch=epoch)


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
    save_figure04_decode_cache(path, "fp1", payload)
    loaded = load_figure04_decode_cache(path, "fp1")
    assert loaded is not None
    assert set(loaded.keys()) == set(payload.keys())
    np.testing.assert_array_equal(
        loaded["continuous_fragmented_results"], payload["continuous_fragmented_results"]
    )
    assert not path.with_name(path.name + ".tmp").exists()


def test_diagnostics_cache_round_trip_and_misses(tmp_path: Path) -> None:
    path = tmp_path / "intermediates" / "d.joblib"
    save_figure04_diagnostics_cache(path, "fp1", "dfp1", _diagnostics_payload())
    loaded = load_figure04_diagnostics_cache(path, "fp1", "dfp1")
    assert loaded is not None
    assert set(loaded.keys()) == set(_diagnostics_payload().keys())
    # Both fingerprints gate the diagnostics cache: a decode change or a diagnostics change misses.
    assert load_figure04_diagnostics_cache(path, "fp2", "dfp1") is None
    assert load_figure04_diagnostics_cache(path, "fp1", "dfp2") is None
    assert load_figure04_diagnostics_cache(tmp_path / "nope.joblib", "fp1", "dfp1") is None


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
    assert load_figure04_decode_cache(tmp_path / "nope.joblib", "fp") is None


def test_miss_on_fingerprint_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    save_figure04_decode_cache(path, "fp1", _payload())
    assert load_figure04_decode_cache(path, "fp2") is None


def test_schema_bumps_change_the_fingerprints_that_gate_the_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths_with_input(tmp_path)
    original = compute_figure04_cache_provenance(Figure4Config(), paths)
    monkeypatch.setattr(
        figure04_cache, "FIGURE04_DECODE_SCHEMA_VERSION", FIGURE04_DECODE_SCHEMA_VERSION + 1
    )
    monkeypatch.setattr(
        figure04_cache,
        "FIGURE04_DIAGNOSTICS_SCHEMA_VERSION",
        FIGURE04_DIAGNOSTICS_SCHEMA_VERSION + 1,
    )
    changed = compute_figure04_cache_provenance(Figure4Config(), paths)
    assert changed.fingerprint_sha256 != original.fingerprint_sha256
    assert changed.diagnostics_fingerprint_sha256 != original.diagnostics_fingerprint_sha256


def test_miss_on_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    joblib.dump([1, 2, 3], path)
    assert load_figure04_decode_cache(path, "fp") is None


def test_miss_on_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    path.write_bytes(b"not a joblib file")
    with pytest.warns(RuntimeWarning, match="could not be read"):
        assert load_figure04_decode_cache(path, "fp") is None


def test_miss_on_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    wrapper = {"schema_version": FIGURE04_DECODE_SCHEMA_VERSION, "fingerprint": "fp", **_payload()}
    del wrapper["spike_counts"]
    joblib.dump(wrapper, path)
    assert load_figure04_decode_cache(path, "fp") is None


def test_miss_on_extra_key(tmp_path: Path) -> None:
    path = tmp_path / "c.joblib"
    wrapper = {
        "schema_version": FIGURE04_DECODE_SCHEMA_VERSION,
        "fingerprint": "fp",
        "unexpected": 1,
        **_payload(),
    }
    joblib.dump(wrapper, path)
    assert load_figure04_decode_cache(path, "fp") is None


def test_save_rejects_missing_payload_key(tmp_path: Path) -> None:
    payload = _payload()
    del payload["spike_counts"]
    with pytest.raises(ValueError, match="payload keys"):
        save_figure04_decode_cache(tmp_path / "c.joblib", "fp", payload)


def test_save_rejects_extra_payload_key(tmp_path: Path) -> None:
    payload = _payload()
    payload["unexpected"] = 1
    with pytest.raises(ValueError, match="payload keys"):
        save_figure04_decode_cache(tmp_path / "c.joblib", "fp", payload)


def test_fingerprint_changes_with_config_and_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths_with_input(tmp_path)
    config = Figure4Config()
    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "1.0.0")
    fp1 = compute_figure04_cache_provenance(config, paths).fingerprint_sha256
    assert (
        compute_figure04_cache_provenance(config, paths).fingerprint_sha256 == fp1
    )  # deterministic

    changed = dataclasses.replace(
        config,
        package_defaults=dataclasses.replace(
            config.package_defaults, movement_var=config.package_defaults.movement_var + 1.0
        ),
    )
    assert compute_figure04_cache_provenance(changed, paths).fingerprint_sha256 != fp1

    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "2.0.0")
    assert compute_figure04_cache_provenance(config, paths).fingerprint_sha256 != fp1


@pytest.mark.parametrize(
    "field", [field.name for field in dataclasses.fields(Figure4DecoderConfig)]
)
def test_every_decoder_field_changes_the_fingerprint(tmp_path: Path, field: str) -> None:
    paths = _paths_with_input(tmp_path)
    config = Figure4Config()
    changed = dataclasses.replace(
        config,
        decoder=dataclasses.replace(
            config.decoder, **{field: getattr(config.decoder, field) * 2.0}
        ),
    )
    assert (
        compute_figure04_cache_provenance(changed, paths).fingerprint_sha256
        != compute_figure04_cache_provenance(config, paths).fingerprint_sha256
    )


def test_identity_is_the_same_on_another_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python, the architecture, and the dependency closure move the cache keys and
    leave the analysis identities alone: a reader's computer is not a new analysis."""
    paths = _paths_with_input(tmp_path)
    config = Figure4Config()
    here = compute_figure04_cache_provenance(config, paths)

    monkeypatch.setattr(figure04_cache, "_python_version", lambda: "3.99.0")
    monkeypatch.setattr(figure04_cache, "_machine", lambda: "riscv64")
    monkeypatch.setattr(
        figure04_cache, "installed_dependency_versions", lambda _: {"numpy": "99.0"}
    )
    there = compute_figure04_cache_provenance(config, paths)

    assert there.fingerprint_sha256 != here.fingerprint_sha256
    assert there.diagnostics_fingerprint_sha256 != here.diagnostics_fingerprint_sha256
    assert there.decode_identity_sha256 == here.decode_identity_sha256
    assert there.diagnostics_identity_sha256 == here.diagnostics_identity_sha256


def test_identity_changes_with_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuration, input bytes, and hashed source each change the identity."""
    paths = _paths_with_input(tmp_path)
    config = Figure4Config()
    base = compute_figure04_cache_provenance(config, paths)

    changed = dataclasses.replace(
        config,
        decoder=dataclasses.replace(config.decoder, position_bin_size_cm=3.0),
        diagnostics=dataclasses.replace(config.diagnostics, hpd_coverage=0.8),
    )
    other_config = compute_figure04_cache_provenance(changed, paths)
    assert other_config.decode_identity_sha256 != base.decode_identity_sha256
    assert other_config.diagnostics_identity_sha256 != base.diagnostics_identity_sha256

    monkeypatch.setattr(figure04_cache, "portable_source_digest", lambda _: "edited")
    other_source = compute_figure04_cache_provenance(config, paths)
    assert other_source.decode_identity_sha256 != base.decode_identity_sha256
    assert other_source.diagnostics_identity_sha256 != base.diagnostics_identity_sha256
    monkeypatch.undo()

    input_file_path(paths.data_path, paths.animal_date_epoch).write_bytes(b"other bytes")
    other_input = compute_figure04_cache_provenance(config, paths)
    assert other_input.decode_identity_sha256 != base.decode_identity_sha256


def test_portable_source_digest_sees_code_and_ignores_comments(tmp_path: Path) -> None:
    path = tmp_path / "module.py"

    def digest(source: str) -> str:
        path.write_text(source, encoding="utf-8")
        return figure04_cache.portable_source_digest((path,))

    base = digest("def f(a, b=None):\n    return a + 1\n")

    assert digest('# note\ndef f(a, b=None):\n    """Doc."""\n    return a + 1\n') == base
    assert digest("def f(a, b=None):\n    return a + 2\n") != base
    assert digest("def f(a, b=0):\n    return a + 1\n") != base


def test_canonical_syntax_omits_only_what_newer_pythons_add() -> None:
    """Python 3.15 sets ``is_lazy=0`` on every ordinary import. It must read as the
    same code as on older versions, and a genuinely lazy import, or a constant 0
    against None, must not."""

    class Import(ast.Import):
        _fields = ("names", "is_lazy")

    names = [ast.alias(name="numpy")]
    plain = ast.Import(names=names)

    assert figure04_cache._canonical_syntax(Import(names=names, is_lazy=0)) == (
        figure04_cache._canonical_syntax(plain)
    )
    assert figure04_cache._canonical_syntax(Import(names=names, is_lazy=1)) != (
        figure04_cache._canonical_syntax(plain)
    )
    assert figure04_cache._canonical_syntax(ast.Constant(value=0)) != (
        figure04_cache._canonical_syntax(ast.Constant(value=None))
    )


def test_committed_summary_identifies_the_current_analysis() -> None:
    """The summary's identities are recomputed from the live source here, on any
    machine and Python version, with no data. An executable edit to a hashed module therefore
    fails until Figure 4 is regenerated; refreshing the source digest alone does
    not satisfy this."""
    payload = load_summary(FIGURE04_SUMMARY_PATH)
    epoch = payload["dataset"]["animal_date_epoch"]
    caches = payload["provenance"]["figure04_caches"]
    config = Figure4Config()

    assert caches["decode_identity_sha256"] == compute_figure04_decode_identity(
        config, epoch, caches["input_file_sha256"][f"{epoch}{INPUT_FILE_SUFFIX}"]
    )
    assert caches["diagnostics_identity_sha256"] == compute_figure04_diagnostics_identity(
        config.diagnostics
    )


def test_runtime_dependency_closure_walks_a_metadata_graph() -> None:
    graph: dict[str, list[str]] = {
        "root-pkg": [
            "Dep_A>=1",  # followed; spelled un-normalized
            "cyclic",
            "with-extra[Fast]",  # requests an extra of with-extra
            'dev-only; extra == "dev"',  # extras-only: excluded for a no-extras walk
            'old-python; python_version < "3"',  # marker false for this interpreter
            "not-installed",  # optional package absent from the environment
        ],
        "dep-a": ["shared"],
        "cyclic": ["root_pkg", "shared"],  # cycle back to the root
        "with-extra": ['accelerator; extra == "fast"', 'docs-tool; extra == "docs"'],
        "accelerator": [],
        "shared": [],
        "dev-only": ["shared"],
        "old-python": [],
        "docs-tool": [],
    }
    calls: list[str] = []

    def requirements_of(name: str) -> list[str] | None:
        calls.append(name)
        return graph.get(name)

    closure = figure04_cache.runtime_dependency_closure("Root_Pkg", requirements_of)
    assert closure == {"root-pkg", "dep-a", "cyclic", "with-extra", "accelerator", "shared"}
    # Every lookup used the normalized name; the cycle visited each node once per extra.
    assert all(name == name.lower() and "_" not in name for name in calls)
    assert calls.count("root-pkg") == 1


def test_installed_dependency_versions_records_the_real_closure() -> None:
    versions = figure04_cache.installed_dependency_versions("statespacecheck")
    assert list(versions) == sorted(versions)
    assert {"statespacecheck", "numpy", "scipy"} <= set(versions)
    assert "pytest" not in versions  # a dev extra of statespacecheck
    decode = figure04_cache.installed_dependency_versions("non_local_detector")
    assert {"non-local-detector", "jax", "numpy", "scipy"} <= set(decode)


def _patched_versions(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]) -> None:
    """Report ``overrides`` in place of the installed versions of those packages."""
    installed = figure04_cache.version

    def patched(name: str) -> str:
        normalized = name.lower().replace("_", "-")
        return overrides.get(normalized, installed(name))

    monkeypatch.setattr(figure04_cache, "version", patched)


@pytest.mark.parametrize(
    ("package", "changes_decode", "changes_diagnostics"),
    [
        ("jax", True, False),  # decode closure only
        ("numpy", True, True),  # both closures
        ("statespacecheck", False, True),  # the diagnostics root
    ],
)
def test_fingerprints_track_dependency_closure_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package: str,
    changes_decode: bool,
    changes_diagnostics: bool,
) -> None:
    paths = _paths_with_input(tmp_path)
    config = Figure4Config()
    original = compute_figure04_cache_provenance(config, paths)
    _patched_versions(monkeypatch, {package: "0.0.0.test"})
    changed = compute_figure04_cache_provenance(config, paths)
    assert (changed.fingerprint_sha256 != original.fingerprint_sha256) is changes_decode
    assert (
        changed.diagnostics_fingerprint_sha256 != original.diagnostics_fingerprint_sha256
    ) is changes_diagnostics


@pytest.mark.parametrize(
    ("helper", "value"), [("_python_version", "3.99.0"), ("_machine", "riscv64")]
)
def test_fingerprints_track_python_version_and_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helper: str, value: str
) -> None:
    paths = _paths_with_input(tmp_path)
    config = Figure4Config()
    original = compute_figure04_cache_provenance(config, paths)
    monkeypatch.setattr(figure04_cache, helper, lambda: value)
    changed = compute_figure04_cache_provenance(config, paths)
    assert changed.fingerprint_sha256 != original.fingerprint_sha256
    assert changed.diagnostics_fingerprint_sha256 != original.diagnostics_fingerprint_sha256
    environment = changed.environment
    recorded = environment.python_version if helper == "_python_version" else environment.machine
    assert recorded == value


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A source tree whose files can change without editing the checkout."""
    package = tmp_path / "package"
    package.mkdir()
    filenames = set(figure04_cache._DECODE_SOURCE_FILES) | set(
        figure04_cache._DIAGNOSTIC_SOURCE_FILES
    )
    for filename in filenames | {"figure04_layout.py"}:
        (package / filename).write_text('"""Documentation."""\nVALUE = 1\n', encoding="utf-8")
    monkeypatch.setattr(figure04_cache, "__file__", str(package / "figure04_cache.py"))
    return package


@pytest.mark.parametrize(
    "filename",
    [
        "figure04_decoder.py",
        "figure04_fit.py",
        "figure04_place_fields.py",
        "figure04_input.py",
    ],
)
def test_decode_source_change_rejects_existing_cache(
    source_tree: Path, tmp_path: Path, filename: str
) -> None:
    paths = _paths_with_input(tmp_path, "epoch")
    config = Figure4Config()
    original = compute_figure04_cache_provenance(config, paths).fingerprint_sha256
    save_figure04_decode_cache(paths.decode_cache_path, original, _payload())
    assert load_figure04_decode_cache(paths.decode_cache_path, original) is not None

    (source_tree / filename).write_text('"""Documentation."""\nVALUE = 2\n', encoding="utf-8")
    changed = compute_figure04_cache_provenance(config, paths).fingerprint_sha256
    assert changed != original
    assert load_figure04_decode_cache(paths.decode_cache_path, changed) is None


def test_documentation_and_plotting_edits_preserve_both_caches(
    source_tree: Path, tmp_path: Path
) -> None:
    paths = _paths_with_input(tmp_path, "epoch")
    config = Figure4Config()
    original = compute_figure04_cache_provenance(config, paths)
    for filename in set(figure04_cache._DECODE_SOURCE_FILES) | set(
        figure04_cache._DIAGNOSTIC_SOURCE_FILES
    ):
        (source_tree / filename).write_text(
            '"""Updated documentation."""\n# New comment\nVALUE = 1\n', encoding="utf-8"
        )
    (source_tree / "figure04_layout.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert compute_figure04_cache_provenance(config, paths) == original


@pytest.mark.parametrize(
    "filename", ["diagnostics.py", "figure04_diagnostics.py", "figure04_workflow.py"]
)
def test_diagnostic_only_source_edit_preserves_decode_cache(
    source_tree: Path, tmp_path: Path, filename: str
) -> None:
    paths = _paths_with_input(tmp_path, "epoch")
    config = Figure4Config()
    original = compute_figure04_cache_provenance(config, paths)
    (source_tree / filename).write_text("VALUE = 2\n", encoding="utf-8")
    changed = compute_figure04_cache_provenance(config, paths)
    assert changed.fingerprint_sha256 == original.fingerprint_sha256
    assert changed.diagnostics_fingerprint_sha256 != original.diagnostics_fingerprint_sha256


def test_diagnostics_and_decode_are_computed_in_their_hashed_modules() -> None:
    """The functions that compute each cached payload live in modules hashed
    into that cache's fingerprint, so editing them cannot reuse a stale cache."""
    assert (
        Path(inspect.getfile(figure04_fit.fit_and_decode)).name
        in figure04_cache._DECODE_SOURCE_FILES
    )
    for function in (
        figure04_workflow._compute_diagnostics_payload,
        figure04_diagnostics.compute_results_diagnostics,
    ):
        assert Path(inspect.getfile(function)).name in figure04_cache._DIAGNOSTIC_SOURCE_FILES


def test_fingerprint_unchanged_when_block_size_changes(tmp_path: Path) -> None:
    # block_size (Figure4ExecutionConfig) is a performance-only knob that leaves
    # the decode result identical, so it must NOT be hashed into the fingerprint:
    # changing it must not invalidate a cached decode.
    paths = _paths_with_input(tmp_path)
    config = Figure4Config()
    fp = compute_figure04_cache_provenance(config, paths).fingerprint_sha256

    changed = dataclasses.replace(
        config,
        execution=dataclasses.replace(config.execution, block_size=config.execution.block_size * 2),
    )
    assert compute_figure04_cache_provenance(changed, paths).fingerprint_sha256 == fp


def test_fingerprint_changes_when_input_file_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replacing the input file under the same epoch must invalidate the cache:
    # the fingerprint hashes the file contents, not just ``animal_date_epoch``.
    paths = _paths_with_input(tmp_path)
    config = Figure4Config()
    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "1.0.0")

    input_file = tmp_path / f"epoch_x{INPUT_FILE_SUFFIX}"
    input_file.write_bytes(b"original")
    fp_original = compute_figure04_cache_provenance(config, paths).fingerprint_sha256
    assert (
        compute_figure04_cache_provenance(config, paths).fingerprint_sha256 == fp_original
    )  # deterministic

    input_file.write_bytes(b"REPLACED with different data")
    assert compute_figure04_cache_provenance(config, paths).fingerprint_sha256 != fp_original


def test_cache_provenance_serializes_complete_path_independent_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths_with_input(tmp_path)
    (tmp_path / f"epoch_x{INPUT_FILE_SUFFIX}").write_bytes(b"input")
    monkeypatch.setattr(figure04_cache, "_installed_non_local_detector_version", lambda: "1.2.3")

    provenance = compute_figure04_cache_provenance(Figure4Config(), paths)
    payload = provenance.artifact_payload()

    assert payload["schema_version"] == FIGURE04_DECODE_SCHEMA_VERSION
    assert payload["non_local_detector_version"] == "1.2.3"
    assert set(payload["input_file_sha256"]) == {f"epoch_x{INPUT_FILE_SUFFIX}"}
    assert payload["diagnostics_schema_version"] == FIGURE04_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["diagnostics_fingerprint_sha256"] == compute_figure04_diagnostics_fingerprint(
        Figure4DiagnosticsConfig()
    )
    assert payload["diagnostics_config"] == {
        "hpd_coverage": 0.95,
        "event_selection": "all_spikes_in_recording",
    }
    assert payload["environment"] == {
        "python_version": figure04_cache._python_version(),
        "machine": figure04_cache._machine(),
        "decode_dependency_versions": figure04_cache.installed_dependency_versions(
            "non_local_detector"
        ),
        "diagnostics_dependency_versions": figure04_cache.installed_dependency_versions(
            "statespacecheck"
        ),
    }
    assert str(tmp_path) not in repr(payload)
