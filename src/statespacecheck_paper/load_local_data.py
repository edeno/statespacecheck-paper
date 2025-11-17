"""Load neural recording data from local files without database dependencies.

This module provides file-based data loading without requiring Spyglass database
connections. Useful for working with pre-exported datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_neural_recording_from_files(
    data_path: str | Path,
    animal_date_epoch: str,
) -> dict[str, Any]:
    """Load neural recording data from local pickle/joblib files.

    Alternative data loading function for pre-exported datasets stored as
    local files. Useful for sharing datasets or working without database access.

    Parameters
    ----------
    data_path : str or Path
        Directory containing the data files.
    animal_date_epoch : str
        Identifier for the recording session (e.g., "j1620210710_02_r1").

    Returns
    -------
    data : dict
        Dataset dictionary with keys:
        - "position_info" : pd.DataFrame
            Time-indexed position data.
        - "spike_times" : list[np.ndarray]
            Spike times for each unit/tetrode.
        - "track_graph" : networkx.Graph
            Track environment structure.
        - "linear_edge_order" : list[tuple]
            Edge ordering for linearization.
        - "linear_edge_spacing" : float or list[float]
            Spacing between nodes.

    Examples
    --------
    >>> data = load_neural_recording_from_files(
    ...     "data/", "j1620210710_02_r1"
    ... )
    >>> position = data["position_info"]
    >>> spikes = data["spike_times"]

    Notes
    -----
    Expected files in data_path:
    - {animal_date_epoch}_position_info.pkl
    - {animal_date_epoch}_HPC_spike_times.pkl
    - {animal_date_epoch}_track_graph.pkl
    - {animal_date_epoch}_linear_edge_order.pkl
    - {animal_date_epoch}_linear_edge_spacing.pkl
    """
    data_path = Path(data_path)

    position_info = pd.read_pickle(data_path / f"{animal_date_epoch}_position_info.pkl")
    spike_times = joblib.load(data_path / f"{animal_date_epoch}_HPC_spike_times.pkl")
    track_graph = joblib.load(data_path / f"{animal_date_epoch}_track_graph.pkl")
    linear_edge_order = joblib.load(data_path / f"{animal_date_epoch}_linear_edge_order.pkl")
    linear_edge_spacing = joblib.load(data_path / f"{animal_date_epoch}_linear_edge_spacing.pkl")

    return {
        "position_info": position_info,
        "spike_times": spike_times,
        "track_graph": track_graph,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
    }
