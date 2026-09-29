"""Convert the five pickles the Figure-4 recording was first exported as.

Before the Figure-4 input file existed, the recording was exported as five
pickles (see ``docs/data-lineage.md``). :func:`convert_pickles_to_input_file`
writes them as the ``.npz`` input file with
:func:`~statespacecheck_paper.spyglass_pipeline.figure04_input.recording_arrays`
and :func:`~statespacecheck_paper.spyglass_pipeline.figure04_input.write_npz`, and
checks with :func:`recording_difference` that the file loads back identical. It
was needed once; only ``scripts/convert_figure04_pickles.py`` uses it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from statespacecheck_paper.figure04_input import (
    NeuralRecordingData,
    input_file_path,
    load_figure04_input,
)
from statespacecheck_paper.spyglass_pipeline.figure04_input import recording_arrays, write_npz

# Suffixes of the five pickles the recording was first exported as.
PICKLE_SUFFIXES = (
    "_position_info.pkl",
    "_HPC_spike_times.pkl",
    "_track_graph.pkl",
    "_linear_edge_order.pkl",
    "_linear_edge_spacing.pkl",
)


def read_recording_pickles(
    data_path: str | Path,
    animal_date_epoch: str,
) -> NeuralRecordingData:
    """Read the five pickles the recording was first exported as (for conversion only).

    Unpickling runs code from the files, so use this only on the lab's own
    exports, to convert them with :func:`recording_arrays` and :func:`write_npz`.

    Parameters
    ----------
    data_path : str or Path
        Directory containing the pickles (see :data:`PICKLE_SUFFIXES`).
    animal_date_epoch : str
        File-name prefix.

    Returns
    -------
    NeuralRecordingData
        Validated recording.
    """
    position, spikes, graph, order, spacing = (
        Path(data_path) / f"{animal_date_epoch}{suffix}" for suffix in PICKLE_SUFFIXES
    )
    return NeuralRecordingData(
        position_info=pd.read_pickle(position),
        spike_times=tuple(np.asarray(st, dtype=np.float64) for st in joblib.load(spikes)),
        track_graph=joblib.load(graph),
        linear_edge_order=tuple(tuple(edge) for edge in joblib.load(order)),
        linear_edge_spacing=joblib.load(spacing),
    )


def recording_difference(a: NeuralRecordingData, b: NeuralRecordingData) -> str | None:
    """Describe the first difference between two recordings, or return ``None``.

    Position must match exactly (values, dtypes, index and its name, column
    order); spike times unit by unit; the track graph node-, edge-, and
    attribute-wise, in the same node and edge order; edge order and spacing by
    value.

    Parameters
    ----------
    a, b : NeuralRecordingData
        Recordings to compare.

    Returns
    -------
    str or None
        ``None`` if identical, otherwise what differs.
    """
    try:
        pd.testing.assert_frame_equal(a.position_info, b.position_info, check_exact=True)
    except AssertionError as exc:
        return f"position_info: {' '.join(str(exc).split())}"
    if len(a.spike_times) != len(b.spike_times):
        return f"{len(a.spike_times)} vs {len(b.spike_times)} units"
    for unit, (x, y) in enumerate(zip(a.spike_times, b.spike_times, strict=True)):
        if not np.array_equal(x, y):
            return f"spike times of unit {unit} differ"
    if (
        not nx.utils.graphs_equal(a.track_graph, b.track_graph)
        or list(a.track_graph.nodes) != list(b.track_graph.nodes)
        or list(a.track_graph.edges) != list(b.track_graph.edges)
    ):
        return "track graph differs"
    if a.linear_edge_order != b.linear_edge_order:
        return "linear edge order differs"
    if a.linear_edge_spacing != b.linear_edge_spacing:
        return f"linear edge spacing {a.linear_edge_spacing} vs {b.linear_edge_spacing}"
    return None


def convert_pickles_to_input_file(
    data_path: str | Path,
    output_dir: str | Path,
    animal_date_epoch: str,
) -> Path:
    """Convert the five pickles to the ``.npz`` input file, and verify it.

    Parameters
    ----------
    data_path : str or Path
        Directory with the pickles (see :data:`PICKLE_SUFFIXES`).
    output_dir : str or Path
        Directory to write ``{animal_date_epoch}_figure04_inputs.npz`` to.
    animal_date_epoch : str
        File-name prefix.

    Returns
    -------
    Path
        The written file.

    Raises
    ------
    FileExistsError
        If the output file exists.
    ValueError
        If the written file does not load back identical to the pickles.
    """
    output = input_file_path(output_dir, animal_date_epoch)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing input file: {output}")
    pickled = read_recording_pickles(data_path, animal_date_epoch)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_npz(
        output,
        recording_arrays(
            pickled.position_info,
            pickled.spike_times,
            pickled.track_graph,
            pickled.linear_edge_order,
            pickled.linear_edge_spacing,
        ),
    )
    difference = recording_difference(
        pickled, load_figure04_input(output.parent, animal_date_epoch)
    )
    if difference is not None:
        raise ValueError(f"Converted file differs from the pickles ({difference}): {output}")
    return output
