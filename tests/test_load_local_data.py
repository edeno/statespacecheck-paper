"""Tests for the NeuralRecordingData contract at the loader boundary."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from statespacecheck_paper.load_local_data import (
    LEGACY_PICKLE_SUFFIXES,
    NeuralRecordingData,
    convert_legacy_pickle_exports,
    load_neural_recording_from_files,
    read_legacy_pickle_exports,
    recording_arrays,
    recording_difference,
    write_npz,
)


def _position_info(n_time: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "head_position_x": np.linspace(0.0, 1.0, n_time),
            "head_position_y": np.linspace(1.0, 0.0, n_time),
            "linear_position": np.linspace(0.0, 2.0, n_time),
        },
        index=np.linspace(0.0, 0.014, n_time),
    )


def _track_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edge(0, 1)
    return g


def _recording(**overrides: object) -> NeuralRecordingData:
    kwargs: dict[str, Any] = dict(
        position_info=_position_info(),
        spike_times=(np.array([0.001, 0.005]), np.array([0.010])),
        track_graph=_track_graph(),
        linear_edge_order=((0, 1),),
        linear_edge_spacing=0.0,
    )
    kwargs.update(overrides)
    return NeuralRecordingData(**kwargs)


def test_valid_construction() -> None:
    recording = _recording()
    assert recording.linear_edge_spacing == 0.0
    assert len(recording.spike_times) == 2


def test_rejects_empty_position_info() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _recording(position_info=_position_info(0))


def test_rejects_missing_column() -> None:
    df = _position_info().drop(columns=["linear_position"])
    with pytest.raises(ValueError, match="missing required columns"):
        _recording(position_info=df)


def test_rejects_nonincreasing_index() -> None:
    df = _position_info()
    df.index = np.r_[df.index[:-1], df.index[-2]]  # duplicate/non-increasing tail
    with pytest.raises(ValueError, match="strictly increasing"):
        _recording(position_info=df)


def test_rejects_nonfinite_spike_times() -> None:
    with pytest.raises(ValueError, match="finite"):
        _recording(spike_times=(np.array([0.0, np.inf]),))


def test_rejects_nonsorted_spike_times() -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        _recording(spike_times=(np.array([1.0, 0.0]),))


def test_rejects_edge_not_in_graph() -> None:
    with pytest.raises(ValueError, match="not in track_graph"):
        _recording(linear_edge_order=((0, 2),))


def test_rejects_negative_spacing() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _recording(linear_edge_spacing=-1.0)


def test_spike_arrays_are_read_only() -> None:
    recording = _recording()
    with pytest.raises(ValueError, match="read-only|write"):
        recording.spike_times[0][0] = 99.0


def test_spike_arrays_are_copied_and_leave_caller_writable() -> None:
    # A float64 input array must not be frozen in place, and the stored copy
    # must be isolated from later mutation of the caller's array.
    caller_array = np.array([0.001, 0.005], dtype=np.float64)
    recording = _recording(spike_times=(caller_array,))
    assert caller_array.flags.writeable  # caller's array untouched by freezing
    caller_array[0] = 99.0
    assert recording.spike_times[0][0] == 0.001  # stored copy isolated
    assert not recording.spike_times[0].flags.writeable


def test_missing_directory_raises_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        load_neural_recording_from_files(tmp_path / "does_not_exist", "j1620210710_02_r1")


def test_missing_export_files_lists_what_is_absent(tmp_path: Path) -> None:
    # Directory exists but the input file does not: the loader should name the
    # missing file rather than surfacing a bare np.load traceback.
    with pytest.raises(
        FileNotFoundError, match="Missing 1 expected export file.*_figure04_inputs.npz"
    ):
        load_neural_recording_from_files(tmp_path, "j1620210710_02_r1")


# --- .npz input file ----------------------------------------------------------

_EPOCH = "rat20200101_02_r1"


def _graph_with_attributes() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, pos=(np.float64(0.0), np.float64(0.0)))
    graph.add_node(1, pos=(np.float64(10.0), np.float64(0.0)))
    graph.add_edge(0, 1, distance=np.float64(10.0), edge_id=0)
    return graph


def _write(recording_kwargs: dict[str, Any], directory: Path) -> Path:
    path = directory / f"{_EPOCH}_figure04_inputs.npz"
    return write_npz(path, recording_arrays(**recording_kwargs))


def _kwargs(**overrides: object) -> dict[str, Any]:
    position_info = _position_info().assign(patch_id=np.arange(8, dtype=np.int64))
    position_info.index.name = "time"
    kwargs: dict[str, Any] = dict(
        position_info=position_info,
        spike_times=[np.array([0.001, 0.005]), np.array([], dtype=np.float64)],
        track_graph=_graph_with_attributes(),
        linear_edge_order=[(0, 1)],
        linear_edge_spacing=15,
    )
    kwargs.update(overrides)
    return kwargs


def test_npz_round_trip_is_exact(tmp_path: Path) -> None:
    kwargs = _kwargs()
    _write(kwargs, tmp_path)

    loaded = load_neural_recording_from_files(tmp_path, _EPOCH)
    expected = NeuralRecordingData(
        position_info=kwargs["position_info"],
        spike_times=tuple(kwargs["spike_times"]),
        track_graph=kwargs["track_graph"],
        linear_edge_order=((0, 1),),
        linear_edge_spacing=15.0,
    )
    assert recording_difference(loaded, expected) is None
    assert loaded.spike_times[1].shape == (0,)
    assert loaded.position_info["patch_id"].dtype == np.int64


def test_npz_loads_without_pickle(tmp_path: Path) -> None:
    path = _write(_kwargs(), tmp_path)
    with np.load(path, allow_pickle=False) as arrays:
        assert all(arrays[name].dtype != object for name in arrays.files)


def test_npz_refuses_an_unknown_format_version(tmp_path: Path) -> None:
    arrays = recording_arrays(**_kwargs())
    arrays["format_version"] = np.asarray(99, dtype=np.int64)
    write_npz(tmp_path / f"{_EPOCH}_figure04_inputs.npz", arrays)

    with pytest.raises(ValueError, match="format_version 99"):
        load_neural_recording_from_files(tmp_path, _EPOCH)


def test_recording_arrays_refuses_what_it_cannot_store() -> None:
    with pytest.raises(ValueError, match="not numeric"):
        recording_arrays(**_kwargs(position_info=_position_info().assign(label="x")))
    graph = _graph_with_attributes()
    graph.edges[0, 1]["color"] = "red"
    with pytest.raises(ValueError, match="attributes"):
        recording_arrays(**_kwargs(track_graph=graph))


def _write_legacy_pickles(directory: Path) -> None:
    kwargs = _kwargs()
    names = dict(
        zip(
            ("position", "spikes", "graph", "order", "spacing"), LEGACY_PICKLE_SUFFIXES, strict=True
        )
    )
    kwargs["position_info"].to_pickle(directory / f"{_EPOCH}{names['position']}")
    for key, value in (
        ("spikes", kwargs["spike_times"]),
        ("graph", kwargs["track_graph"]),
        ("order", kwargs["linear_edge_order"]),
        ("spacing", 15),
    ):
        with (directory / f"{_EPOCH}{names[key]}").open("wb") as f:
            pickle.dump(value, f, protocol=4)


def test_convert_legacy_pickles_writes_an_identical_npz(tmp_path: Path) -> None:
    _write_legacy_pickles(tmp_path)

    path = convert_legacy_pickle_exports(tmp_path, tmp_path / "npz", _EPOCH)

    converted = load_neural_recording_from_files(path.parent, _EPOCH)
    assert recording_difference(converted, read_legacy_pickle_exports(tmp_path, _EPOCH)) is None


def test_convert_refuses_to_overwrite(tmp_path: Path) -> None:
    _write_legacy_pickles(tmp_path)
    convert_legacy_pickle_exports(tmp_path, tmp_path, _EPOCH)

    with pytest.raises(FileExistsError):
        convert_legacy_pickle_exports(tmp_path, tmp_path, _EPOCH)
