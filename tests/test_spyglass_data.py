"""Tests for the Spyglass rebuild of the Figure-4 exports (no database access).

The Spyglass fetches themselves need the lab database; these tests cover the
offline pieces (import laziness, clipping, linearization, file round trip, and
comparison) on simulated data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from track_linearization import make_track_graph

from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.spyglass_data import (
    Figure4Inputs,
    compare_figure04_exports,
    filter_spike_times,
    get_interpolated_position_info,
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


def _inputs(spike_shift: float = 0.0) -> Figure4Inputs:
    track_graph = nx.Graph()
    track_graph.add_node(0, pos=(0.0, 0.0))
    track_graph.add_node(1, pos=(10.0, 0.0))
    track_graph.add_edge(0, 1, distance=10.0, edge_id=0)
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
    write_figure04_inputs(_inputs(), tmp_path, _EPOCH)

    recording = load_neural_recording_from_files(tmp_path, _EPOCH)

    np.testing.assert_array_equal(recording.spike_times[0], [0.001, 0.004])
    assert recording.spike_times[1].shape == (0,)
    assert recording.linear_edge_order == ((0, 1),)
    assert recording.linear_edge_spacing == 15.0
    pd.testing.assert_frame_equal(recording.position_info, _inputs().position_info)


def test_write_refuses_to_overwrite_unless_asked(tmp_path: Path) -> None:
    write_figure04_inputs(_inputs(), tmp_path, _EPOCH)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_figure04_inputs(_inputs(), tmp_path, _EPOCH)
    write_figure04_inputs(_inputs(), tmp_path, _EPOCH, overwrite=True)


def test_compare_flags_only_the_file_that_differs(tmp_path: Path) -> None:
    write_figure04_inputs(_inputs(), tmp_path / "reference", _EPOCH)
    write_figure04_inputs(_inputs(), tmp_path / "same", _EPOCH)
    write_figure04_inputs(_inputs(spike_shift=1e-9), tmp_path / "shifted", _EPOCH)

    same = compare_figure04_exports(tmp_path / "reference", tmp_path / "same", _EPOCH)
    shifted = compare_figure04_exports(tmp_path / "reference", tmp_path / "shifted", _EPOCH)

    assert all(same.values())
    assert shifted == {name: not name.endswith("_HPC_spike_times.pkl") for name in same}
