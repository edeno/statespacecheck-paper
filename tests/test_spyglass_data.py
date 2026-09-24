"""Tests for the Spyglass rebuild of the Figure-4 exports (no database access).

The Spyglass fetches themselves need the lab database; these tests cover the
offline pieces (import laziness, clipping, linearization, file round trip, and
comparison) on simulated data.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from track_linearization import make_track_graph

from statespacecheck_paper import spyglass_data
from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.spyglass_data import (
    Figure4Inputs,
    check_output_paths,
    compare_figure04_exports,
    declared_attribute_names,
    epoch_identifier,
    export_file_path,
    filter_spike_times,
    get_interpolated_position_info,
    get_patch_id,
    log_figure04_export,
    write_figure04_inputs,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EPOCH = "rat20200101_02_r1"
_NO_DATABASE_IMPORTS = (
    "import sys\n"
    "loaded = sorted({m.split('.')[0] for m in sys.modules} & {'spyglass', 'datajoint'})\n"
    "assert not loaded, loaded\n"
)


def test_importing_module_does_not_import_spyglass_or_datajoint() -> None:
    code = "import statespacecheck_paper.spyglass_data\n" + _NO_DATABASE_IMPORTS
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("script", ["fetch_figure04_inputs.py", "spyglass_export_figure04.py"])
def test_script_help_does_not_import_spyglass_or_datajoint(script: str) -> None:
    path = str(_REPO_ROOT / "scripts" / script)
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{path!r}, '--help']\n"
        "try:\n"
        f"    runpy.run_path({path!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
    ) + _NO_DATABASE_IMPORTS
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_filter_spike_times_is_inclusive_and_keeps_empty_units() -> None:
    position_time = np.array([1.0, 1.5, 2.0])
    spike_times = [
        np.array([0.5, 1.0, 1.7, 2.0, 2.5]),
        np.array([0.1, 3.0]),
    ]

    filtered = filter_spike_times(spike_times, position_time)

    assert len(filtered) == 2
    np.testing.assert_array_equal(filtered[0], [1.0, 1.7, 2.0])
    assert filtered[1].shape == (0,)


def test_get_interpolated_position_info_interpolates_and_linearizes() -> None:
    # L-shaped track: 0 -(10 cm)- 1 -(10 cm)- 2, no gap between edges.
    track_graph = make_track_graph([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], [(0, 1), (1, 2)])
    position_info = pd.DataFrame(
        {"head_position_x": [0.0, 10.0, 10.0], "head_position_y": [0.0, 0.0, 10.0]},
        index=pd.Index([0.0, 1.0, 2.0], name="time"),
    )
    time = np.array([0.5, 1.5])

    result = get_interpolated_position_info(position_info, time, track_graph, [(0, 1), (1, 2)], 0)

    np.testing.assert_array_equal(result.index.to_numpy(), time)
    np.testing.assert_allclose(result["head_position_x"], [5.0, 10.0])
    np.testing.assert_allclose(result["head_position_y"], [0.0, 5.0])
    np.testing.assert_allclose(result["linear_position"], [5.0, 15.0])
    np.testing.assert_array_equal(result["track_segment_id"], [0, 1])


def test_get_interpolated_position_info_adds_edge_spacing() -> None:
    track_graph = make_track_graph([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], [(0, 1), (1, 2)])
    position_info = pd.DataFrame(
        {"head_position_x": [5.0, 10.0], "head_position_y": [0.0, 5.0]},
        index=pd.Index([0.0, 1.0], name="time"),
    )

    result = get_interpolated_position_info(
        position_info, position_info.index.to_numpy(), track_graph, [(0, 1), (1, 2)], 15
    )

    np.testing.assert_allclose(result["linear_position"], [5.0, 30.0])
    # Onto its own timestamps (as the Figure-4 export does) the input is unchanged.
    pd.testing.assert_frame_equal(result[list(position_info.columns)], position_info)


def _inputs(spike_shift: float = 0.0) -> Figure4Inputs:
    track_graph = make_track_graph([(0.0, 0.0), (10.0, 0.0)], [(0, 1)])
    position_info = pd.DataFrame(
        {
            "head_position_x": np.linspace(0.0, 10.0, 5),
            "head_position_y": np.zeros(5),
            "linear_position": np.linspace(0.0, 10.0, 5),
            "patch_id": np.ones(5, dtype=np.int64),
        },
        index=pd.Index(np.linspace(0.0, 0.008, 5), name="time"),
    )
    return Figure4Inputs(
        position_info=position_info,
        spike_times=[np.array([0.001, 0.004 + spike_shift]), np.array([], dtype=np.float64)],
        track_graph=track_graph,
        linear_edge_order=[(0, 1)],
        linear_edge_spacing=15,
    )


def test_written_exports_load_through_the_figure_loader(tmp_path: Path) -> None:
    inputs = _inputs()
    write_figure04_inputs(inputs, tmp_path, _EPOCH)

    recording = load_neural_recording_from_files(tmp_path, _EPOCH)

    np.testing.assert_array_equal(recording.spike_times[0], [0.001, 0.004])
    assert recording.spike_times[1].shape == (0,)
    assert recording.linear_edge_order == ((0, 1),)
    assert recording.linear_edge_spacing == 15.0
    pd.testing.assert_frame_equal(recording.position_info, inputs.position_info)


def test_write_refuses_an_existing_file_and_leaves_it(tmp_path: Path) -> None:
    existing = export_file_path(tmp_path, _EPOCH)
    existing.write_bytes(b"earlier")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_figure04_inputs(_inputs(), tmp_path, _EPOCH)
    assert existing.read_bytes() == b"earlier"


def test_write_overwrite_replaces_the_contents(tmp_path: Path) -> None:
    write_figure04_inputs(_inputs(), tmp_path, _EPOCH)
    write_figure04_inputs(_inputs(spike_shift=1e-3), tmp_path, _EPOCH, overwrite=True)

    recording = load_neural_recording_from_files(tmp_path, _EPOCH)
    np.testing.assert_array_equal(recording.spike_times[0], [0.001, 0.004 + 1e-3])


def test_write_is_deterministic(tmp_path: Path) -> None:
    first = write_figure04_inputs(_inputs(), tmp_path / "a", _EPOCH)
    second = write_figure04_inputs(_inputs(), tmp_path / "b", _EPOCH)

    assert first.read_bytes() == second.read_bytes()


def test_compare_finds_no_difference_between_identical_exports(tmp_path: Path) -> None:
    write_figure04_inputs(_inputs(), tmp_path / "reference", _EPOCH)
    write_figure04_inputs(_inputs(), tmp_path / "same", _EPOCH)

    differences = compare_figure04_exports(tmp_path / "reference", tmp_path / "same", _EPOCH)

    assert "spike_times" in differences and "position/head_position_x" in differences
    assert set(differences.values()) == {None}


def _longer_edge(inputs: Figure4Inputs) -> Figure4Inputs:
    track_graph = inputs.track_graph.copy()
    track_graph.edges[0, 1]["distance"] = 10.5
    return dataclasses.replace(inputs, track_graph=track_graph)


_PERTURBATIONS = {
    "position/patch_id": lambda inputs: dataclasses.replace(
        inputs, position_info=inputs.position_info.astype({"patch_id": np.int32})
    ),
    "spike_times": lambda inputs: _inputs(spike_shift=1e-9),
    "track_edge_distance": _longer_edge,
    "linear_edge_order": lambda inputs: dataclasses.replace(inputs, linear_edge_order=[(1, 0)]),
    "linear_edge_spacing": lambda inputs: dataclasses.replace(inputs, linear_edge_spacing=16),
}


@pytest.mark.parametrize("array_name", list(_PERTURBATIONS))
def test_compare_flags_only_the_array_that_differs(tmp_path: Path, array_name: str) -> None:
    write_figure04_inputs(_inputs(), tmp_path / "reference", _EPOCH)
    write_figure04_inputs(_PERTURBATIONS[array_name](_inputs()), tmp_path / "changed", _EPOCH)

    differences = compare_figure04_exports(tmp_path / "reference", tmp_path / "changed", _EPOCH)

    assert {name for name, difference in differences.items() if difference} == {array_name}


def test_write_refuses_a_directed_graph(tmp_path: Path) -> None:
    directed = dataclasses.replace(_inputs(), track_graph=nx.DiGraph(_inputs().track_graph))

    with pytest.raises(ValueError, match="undirected"):
        write_figure04_inputs(directed, tmp_path, _EPOCH)
    assert list(tmp_path.iterdir()) == []


# --- Data checks -------------------------------------------------------------


def test_epoch_identifier_rejects_names_without_the_spyglass_suffix() -> None:
    with pytest.raises(ValueError, match="_.nwb"):
        epoch_identifier("j1620210710.nwb", "02_r1")


def test_get_patch_id_maps_segments_and_refuses_unmapped_ones() -> None:
    patch_id = get_patch_id(pd.Series([0, 6, 3, 8]))

    np.testing.assert_array_equal(patch_id, [1, 1, 2, 3])
    with pytest.raises(ValueError, match="without a patch"):
        get_patch_id(pd.Series([0, 99]))
    with pytest.raises(ValueError, match="without a patch"):
        get_patch_id(pd.Series([0.0, np.nan]))


def test_write_refuses_inputs_the_loader_would_reject(tmp_path: Path) -> None:
    unsorted = dataclasses.replace(_inputs(), spike_times=[np.array([0.004, 0.001])])

    with pytest.raises(ValueError, match="nondecreasing"):
        write_figure04_inputs(unsorted, tmp_path, _EPOCH)
    assert list(tmp_path.iterdir()) == []


# --- Path checks -------------------------------------------------------------


def test_check_output_paths_rejects_reference_as_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same"):
        check_output_paths(tmp_path, _EPOCH, reference_dir=tmp_path / "sub" / "..", overwrite=True)


def test_check_output_paths_rejects_existing_output_and_missing_reference(tmp_path: Path) -> None:
    write_figure04_inputs(_inputs(), tmp_path / "out", _EPOCH)

    with pytest.raises(FileExistsError):
        check_output_paths(tmp_path / "out", _EPOCH)
    with pytest.raises(FileNotFoundError, match="Missing reference"):
        check_output_paths(tmp_path / "new", _EPOCH, reference_dir=tmp_path / "missing")
    check_output_paths(tmp_path / "new", _EPOCH, reference_dir=tmp_path / "out")


# --- Export guard and ordering (fake Spyglass export tables) -----------------


def test_declared_attribute_names_ignores_references_indexes_and_comments() -> None:
    definition = """
    # table of logged restrictions
    -> master
    table_id: int
    ---
    table_name: varchar(128)
    restriction = null: mediumblob  # the table's restriction
    unique index (export_id, table_name)
    """
    assert declared_attribute_names(definition) == {"table_id", "table_name", "restriction"}


def _fake_common_usage(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    n_existing: int = 0,
    undeclared: tuple[str, ...] = (),
) -> None:
    def table(
        name: str, definition: str, secondary: tuple[str, ...] = (), **parts: object
    ) -> SimpleNamespace:
        return SimpleNamespace(
            full_table_name=name,
            definition=definition,
            heading=SimpleNamespace(secondary_attributes=list(secondary)),
            **parts,
        )

    class ExportSelection:
        full_table_name = "export_selection"
        definition = "export_id: int\n---\npaper_id: varchar(32)\n"
        heading = SimpleNamespace(secondary_attributes=["paper_id", *undeclared])
        Table = table("export_selection__table", "-> master\n")
        File = table("export_selection__file", "-> master\n")

        def __and__(self, restriction: object) -> list[None]:
            calls.append("query")
            return [None] * n_existing

        def start_export(self, paper_id: str, analysis_id: str) -> None:
            calls.append("start_export")

        def stop_export(self) -> None:
            calls.append("stop_export")

    export = table(
        "export",
        "-> ExportSelection\n---\npaper_id: varchar(32)\n",
        ("paper_id",),
        Table=table("export__table", "-> master\n"),
        File=table("export__file", "-> master\n"),
    )
    module = ModuleType("spyglass.common.common_usage")
    module.__dict__.update(ExportSelection=ExportSelection, Export=export)
    monkeypatch.setitem(sys.modules, "spyglass", ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", ModuleType("spyglass.common"))
    monkeypatch.setitem(sys.modules, "spyglass.common.common_usage", module)


def test_log_export_refuses_undeclared_columns_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _fake_common_usage(monkeypatch, calls, undeclared=("included_nwb_file_names",))

    with pytest.raises(RuntimeError, match="included_nwb_file_names"):
        log_figure04_export("paper", "analysis")
    assert calls == []


def test_log_export_refuses_an_existing_paper_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _fake_common_usage(monkeypatch, calls, n_existing=1)

    with pytest.raises(ValueError, match="already has export selections"):
        log_figure04_export("paper", "analysis")
    assert calls == ["query"]


def test_log_export_stops_the_session_when_the_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _fake_common_usage(monkeypatch, calls)

    def failing_fetch(*args: object) -> Figure4Inputs:
        raise OSError("analysis file unavailable")

    monkeypatch.setattr(spyglass_data, "fetch_figure04_inputs", failing_fetch)

    with pytest.raises(OSError):
        log_figure04_export("paper", "analysis")
    assert calls == ["query", "start_export", "stop_export"]


# --- Export script (database-bound helpers replaced) --------------------------


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_script(monkeypatch: pytest.MonkeyPatch, calls: list[str], answer: str) -> ModuleType:
    script = _load_script("spyglass_export_figure04")

    def log(paper_id: str, analysis_id: str) -> Figure4Inputs:
        calls.append("log")
        return _inputs()

    def package(paper_id: str) -> None:
        calls.append("package")

    def describe(paper_id: str) -> list[str]:
        calls.append("describe")
        return []

    def ask(prompt: str) -> str:
        calls.append("prompt")
        return answer

    monkeypatch.setattr(script, "version", lambda package_name: "0.0.0")
    monkeypatch.setattr(script, "log_figure04_export", log)
    monkeypatch.setattr(script, "package_figure04_export", package)
    monkeypatch.setattr(script, "describe_figure04_export", describe)
    monkeypatch.setattr("builtins.input", ask)
    return script


def _script_epoch() -> str:
    return spyglass_data.epoch_identifier(
        spyglass_data.FIGURE04_NWB_FILE_NAME, spyglass_data.FIGURE04_EPOCH_NAME
    )


def test_export_script_requires_compare_to_for_populate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    script = _export_script(monkeypatch, calls, "y")

    with pytest.raises(SystemExit) as exc:
        script.main(["--paper-id", "p", "--output-dir", "out", "--populate"])
    assert exc.value.code == 2
    assert calls == []


def test_export_script_checks_paths_before_asking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    script = _export_script(monkeypatch, calls, "y")
    write_figure04_inputs(_inputs(), tmp_path / "used", _script_epoch())

    with pytest.raises(SystemExit) as exc:
        script.main(["--paper-id", "p", "--output-dir", str(tmp_path / "used")])
    assert exc.value.code == 2
    assert calls == []


def test_export_script_writes_nothing_unless_confirmed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    script = _export_script(monkeypatch, calls, "n")

    assert script.main(["--paper-id", "p", "--output-dir", str(tmp_path / "out")]) == 1
    assert calls == ["prompt"]


def test_export_script_does_not_package_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    script = _export_script(monkeypatch, calls, "y")
    write_figure04_inputs(_inputs(spike_shift=1e-9), tmp_path / "ref", _script_epoch())
    args = ["--paper-id", "p", "--output-dir", str(tmp_path / "out")]

    assert script.main([*args, "--compare-to", str(tmp_path / "ref"), "--populate"]) == 1
    assert calls == ["prompt", "log"]


def test_export_script_packages_a_verified_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    script = _export_script(monkeypatch, calls, "y")
    write_figure04_inputs(_inputs(), tmp_path / "ref", _script_epoch())
    args = ["--paper-id", "p", "--output-dir", str(tmp_path / "out")]

    assert script.main([*args, "--compare-to", str(tmp_path / "ref"), "--populate"]) == 0
    assert calls == ["prompt", "log", "describe", "package"]


def test_fetch_script_refuses_to_compare_the_output_with_itself(tmp_path: Path) -> None:
    script = _load_script("fetch_figure04_inputs")

    with pytest.raises(SystemExit) as exc:
        script.main(["--output-dir", str(tmp_path), "--compare-to", str(tmp_path), "--overwrite"])
    assert exc.value.code == 2
