"""Tests for the NeuralRecordingData contract at the loader boundary."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from statespacecheck_paper.load_local_data import NeuralRecordingData


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
    kwargs: dict[str, object] = dict(
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
