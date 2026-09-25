"""Rebuild the Figure-4 recording input from the Frank-lab Spyglass database.

:func:`statespacecheck_paper.load_local_data.load_neural_recording_from_files`
reads one pre-exported ``.npz`` file. This module is the upstream side of that
boundary: it fetches the Spyglass entries the recording is derived from and
writes that file, so it can be regenerated, compared with the one the figure
used, and captured by a Spyglass export (``scripts/spyglass_export_figure04.py``).
See ``docs/data-lineage.md`` for the entries, processing steps, and verification
record.

Position follows ``continuum-swr-replay``'s ``get_position_info``, and spike
times follow the pattern of its sorted-unit loader. Where this module
deliberately differs, the function docstring says so.

Spyglass is an optional dependency (``uv sync --extra spyglass``). Every
Spyglass import is inside a function, so importing this module never connects
to the database; importing Spyglass does, the first time a function here needs
it. Fetching position and spike times reads analysis NWB files, which requires a
machine with the lab's analysis store mounted.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from statespacecheck_paper.load_local_data import (
    EXPORT_FILE_SUFFIXES,
    NeuralRecordingData,
    recording_arrays,
    write_npz,
)

if TYPE_CHECKING:
    from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
    from statespacecheck_paper.figure04_workflow import Figure4Summary

FIGURE04_NWB_FILE_NAME = "j1620210710_.nwb"
FIGURE04_EPOCH_NAME = "02_r1"

POSITION_INFO_PARAM_NAME = "default_decoding"
LINEARIZATION_PARAM_NAME = "default"

HPC_SORTING_RESTRICTION: Mapping[str, str | int] = MappingProxyType(
    {
        "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
        "preproc_params_name": "franklab_tetrode_hippocampus",
        "team_name": "ac_em_xs",
        "sorter": "mountainsort4",
        "sorter_params_name": "franklab_tetrode_hippocampus_30KHz",
        "curation_id": 1,
    }
)
"""v0 ``CuratedSpikeSorting`` restriction (minus ``nwb_file_name``) for the HPC units.

It matches one entry per sort group; ``artifact_removed_interval_list_name``, the
remaining key field, differs by sort group and is left free (a second match per
sort group is refused).
"""

HPC_REGION_NAME = "hippocampus"

TRACK_SEGMENT_TO_PATCH: Mapping[int, int] = MappingProxyType(
    {0: 1, 1: 1, 6: 1, 2: 2, 3: 2, 7: 2, 4: 3, 5: 3, 8: 3}
)
"""Track segment (edge) ID → patch ID on the spatial-bandit track."""


class PositionInfoDict(TypedDict):
    """Position data and the track graph used to linearize it."""

    position_info: pd.DataFrame
    track_graph: nx.Graph
    linear_edge_order: list[tuple[int, int]]
    linear_edge_spacing: float


@dataclasses.dataclass(frozen=True)
class Figure4Inputs:
    """The five parts of the Figure-4 input as fetched from Spyglass, before writing.

    Values keep the types Spyglass returns (e.g. ``linear_edge_spacing`` is the
    stored blob, an ``int`` for this track); the writer encodes them, and
    :class:`~statespacecheck_paper.load_local_data.NeuralRecordingData`
    normalizes them on load.

    Parameters
    ----------
    position_info : pd.DataFrame
        Time-indexed (seconds) position with linearization and ``patch_id``.
    spike_times : list of np.ndarray, shape (n_spikes,)
        Per-unit spike times (seconds), in ``(sort_group_id, unit_id)`` order.
    track_graph : networkx.Graph
        Track graph used for linearization.
    linear_edge_order : list of (int, int)
        Edge order used for linearization.
    linear_edge_spacing : float
        Spacing between linearized edges (centimeters).
    """

    position_info: pd.DataFrame
    spike_times: list[NDArray[np.float64]]
    track_graph: nx.Graph
    linear_edge_order: list[tuple[int, int]]
    linear_edge_spacing: float


def epoch_identifier(nwb_file_name: str, epoch_name: str) -> str:
    """Return the ``{animal}{date}_{epoch}`` identifier used in export file names.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name, e.g. ``"j1620210710_.nwb"``.
    epoch_name : str
        Epoch interval name, e.g. ``"02_r1"``.

    Returns
    -------
    str
        E.g. ``"j1620210710_02_r1"``.

    Raises
    ------
    ValueError
        If ``nwb_file_name`` does not end in ``"_.nwb"`` (Spyglass's name for its
        copy of a raw NWB file).

    Examples
    --------
    >>> epoch_identifier("j1620210710_.nwb", "02_r1")
    'j1620210710_02_r1'
    """
    if not nwb_file_name.endswith("_.nwb"):
        raise ValueError(f"Expected a Spyglass NWB file name ending in '_.nwb': {nwb_file_name!r}")
    return f"{nwb_file_name.removesuffix('_.nwb')}_{epoch_name}"


def get_position_interval_name(nwb_file_name: str, epoch_name: str) -> str:
    """Map an epoch interval to its position interval via ``PositionIntervalMap``.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name.
    epoch_name : str
        Epoch interval name, e.g. ``"02_r1"``.

    Returns
    -------
    str
        Position interval name, e.g. ``"pos 1 valid times"``.
    """
    from spyglass.common import PositionIntervalMap

    return str(
        (
            PositionIntervalMap & {"nwb_file_name": nwb_file_name, "interval_list_name": epoch_name}
        ).fetch1("position_interval_name")
    )


def get_interpolated_position_info(
    position_info: pd.DataFrame,
    time: NDArray[np.float64],
    track_graph: nx.Graph,
    edge_order: list[tuple[int, int]],
    edge_spacing: float | list[float],
    position_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Interpolate position onto new time points and add linearization.

    Same algorithm as ``continuum-swr-replay``'s
    ``data_loaders.position.get_interpolated_position_info``.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with a time index (seconds) and x/y position columns.
    time : np.ndarray, shape (n_time,)
        Time points (seconds) of the output.
    track_graph : networkx.Graph
        Track graph.
    edge_order : list of (int, int)
        Edge order for linearization.
    edge_spacing : float or list of float
        Spacing between linearized edges (centimeters).
    position_columns : list of str, optional
        x and y position columns. Default ``["head_position_x", "head_position_y"]``.

    Returns
    -------
    pd.DataFrame, shape (n_time, n_columns)
        Interpolated ``position_info`` columns plus ``linear_position``,
        ``track_segment_id``, ``projected_x_position``, and
        ``projected_y_position``.
    """
    from track_linearization import get_linearized_position

    if position_columns is None:
        position_columns = ["head_position_x", "head_position_y"]

    new_index = pd.Index(np.unique(np.concatenate((position_info.index, time))), name="time")
    interpolated_position_info = (
        position_info.reindex(index=new_index).interpolate(method="linear").reindex(index=time)
    )
    linear_position_info = get_linearized_position(
        position=interpolated_position_info[position_columns].to_numpy(),
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).set_index(interpolated_position_info.index)

    return pd.concat((interpolated_position_info, linear_position_info), axis=1)


def get_patch_id(track_segment_id: pd.Series) -> pd.Series:
    """Map track segment IDs to patch IDs with :data:`TRACK_SEGMENT_TO_PATCH`.

    Parameters
    ----------
    track_segment_id : pd.Series, shape (n_time,)
        Track segment (edge) ID per sample.

    Returns
    -------
    pd.Series, shape (n_time,)
        Patch ID per sample.

    Raises
    ------
    ValueError
        If any segment (including a missing one) has no patch, instead of
        silently producing NaN.
    """
    patch_id = track_segment_id.map(TRACK_SEGMENT_TO_PATCH)
    if patch_id.isna().any():
        unmapped = track_segment_id[patch_id.isna()].unique().tolist()
        raise ValueError(f"Track segments without a patch: {unmapped}")
    return patch_id


def get_track_graph(
    nwb_file_name: str, pos_name: str
) -> tuple[nx.Graph, list[tuple[int, int]], float]:
    """Fetch the track graph ``IntervalLinearizedPosition`` recorded for a position.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name.
    pos_name : str
        Position interval name (see :func:`get_position_interval_name`).

    Returns
    -------
    track_graph : networkx.Graph
        Track graph from ``TrackGraph``.
    linear_edge_order : list of (int, int)
        Edge order for linearization.
    linear_edge_spacing : float
        Spacing between linearized edges (centimeters), as stored.
    """
    from spyglass.linearization.v0.main import IntervalLinearizedPosition, TrackGraph

    linearization_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": POSITION_INFO_PARAM_NAME,
        "linearization_param_name": LINEARIZATION_PARAM_NAME,
    }
    track_graph_name = (IntervalLinearizedPosition & linearization_key).fetch1("track_graph_name")
    track_graph_entry = TrackGraph & {"track_graph_name": track_graph_name}
    params = track_graph_entry.fetch1()
    return (
        track_graph_entry.get_networkx_track_graph(),
        params["linear_edge_order"],
        params["linear_edge_spacing"],
    )


def get_position_info(nwb_file_name: str, epoch_name: str, pos_name: str) -> PositionInfoDict:
    """Fetch LED position for one epoch and linearize it onto the track graph.

    Reads ``IntervalPositionInfo`` (``position_info_param_name="default_decoding"``:
    smoothed LED position, upsampled to 500 Hz), drops NaN rows, trims to the
    epoch's ``noPrePostTrialTimes`` interval, linearizes with the ``TrackGraph``
    recorded by ``IntervalLinearizedPosition``, and adds ``patch_id``.

    Unlike ``continuum-swr-replay``'s ``get_position_info``, this does not try
    ``DLCPosV1`` first and does not fall back when an entry is missing. The
    Figure-4 epoch has no DLC position, so that loader resolves to the same
    ``IntervalPositionInfo`` entry; pinning the source keeps a later DLC run
    from silently changing the export.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name.
    epoch_name : str
        Epoch interval name, e.g. ``"02_r1"``.
    pos_name : str
        Position interval name (see :func:`get_position_interval_name`).

    Returns
    -------
    PositionInfoDict
        ``position_info`` (time-indexed, seconds; positions in centimeters),
        ``track_graph``, ``linear_edge_order``, ``linear_edge_spacing``.
    """
    from spyglass.common import IntervalList
    from spyglass.common.common_position import IntervalPositionInfo

    position_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": POSITION_INFO_PARAM_NAME,
    }
    track_graph, linear_edge_order, linear_edge_spacing = get_track_graph(nwb_file_name, pos_name)

    # What ``fetch1_dataframe`` does, without its ``ensure_single_entry()``: that
    # restricts by ``True``, which a Spyglass export session logs as the whole
    # table (see ``unrestricted_log_entries``).
    position_entry = IntervalPositionInfo & position_key
    if len(position_entry) != 1:
        raise ValueError(f"Expected one IntervalPositionInfo entry for {position_key}")
    position_info = IntervalPositionInfo._data_to_df(position_entry.fetch_nwb()[0]).dropna()
    valid_times = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": f"{epoch_name} noPrePostTrialTimes",
        }
    ).fetch1("valid_times")
    position_info = position_info.loc[valid_times[0][0] : valid_times[-1][1]]

    position_info = get_interpolated_position_info(
        position_info,
        position_info.index.to_numpy(),
        track_graph,
        linear_edge_order,
        linear_edge_spacing,
    )
    position_info["patch_id"] = get_patch_id(position_info["track_segment_id"])

    return {
        "position_info": position_info,
        "track_graph": track_graph,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
    }


def get_hpc_sorted_spike_times(
    nwb_file_name: str,
    sorting_restriction: Mapping[str, str | int] = HPC_SORTING_RESTRICTION,
) -> list[NDArray[np.float64]]:
    """Fetch curated hippocampal unit spike times from v0 ``CuratedSpikeSorting``.

    Units come in ``(sort_group_id, unit_id)`` order: each analysis file's units
    are checked to be exactly the group's ``CuratedSpikeSorting.Unit`` entries,
    in ascending ``unit_id``. Sort groups whose curation left no units contribute
    nothing.

    Follows the pattern of ``continuum-swr-replay``'s sorted-unit loader
    (``get_pfc_spike_times``), except that the curation is pinned (not the latest
    ``curation_id``) and the brain region is checked against ``BrainRegion`` in
    the database instead of being selected from the raw NWB file's
    electrode-group descriptions.

    Parameters
    ----------
    nwb_file_name : str
        Spyglass NWB file name.
    sorting_restriction : Mapping
        ``CuratedSpikeSorting`` restriction without ``nwb_file_name``.

    Returns
    -------
    list of np.ndarray, shape (n_spikes,)
        Spike times (seconds) for every unit of the sort, over the whole sort
        interval.

    Raises
    ------
    ValueError
        If no sort matches, a sort group matches more than one sort, any sort
        group lies outside the hippocampus, or an analysis file's units differ
        from ``CuratedSpikeSorting.Unit``.
    """
    from spyglass.common import BrainRegion, ElectrodeGroup
    from spyglass.spikesorting.v0 import CuratedSpikeSorting, SortGroup

    restriction = {"nwb_file_name": nwb_file_name, **sorting_restriction}
    sort_group_keys = (CuratedSpikeSorting & restriction).fetch("KEY", order_by="sort_group_id")
    if len(sort_group_keys) == 0:
        raise ValueError(f"No CuratedSpikeSorting entries match {restriction}")
    sort_group_ids = [key["sort_group_id"] for key in sort_group_keys]
    if len(set(sort_group_ids)) != len(sort_group_ids):
        # e.g. a second artifact_removed_interval_list_name sorted with the same parameters
        raise ValueError(f"{restriction} matches more than one sort for some sort groups")

    # Restrict once, with an OR-list: an active export logs every ``&`` separately
    # and exports their union, so a broad first ``&`` (e.g. the whole session)
    # would pull in all of the session's sort groups.
    sort_group_electrodes = SortGroup.SortGroupElectrode & [
        {"nwb_file_name": nwb_file_name, "sort_group_id": key["sort_group_id"]}
        for key in sort_group_keys
    ]
    region_names = set(
        (sort_group_electrodes * ElectrodeGroup * BrainRegion.proj("region_name")).fetch(
            "region_name"
        )
    )
    if region_names != {HPC_REGION_NAME}:
        raise ValueError(f"Sort groups span regions {sorted(region_names)}, not only hippocampus")

    spike_times: list[NDArray[np.float64]] = []
    for key in sort_group_keys:
        nwb = (CuratedSpikeSorting & key).fetch_nwb()[0]
        # No "units" entry when curation left the sort group empty.
        units = nwb.get("units")
        file_unit_ids = [] if units is None else [int(unit_id) for unit_id in units.index]
        unit_ids = sorted(int(u) for u in (CuratedSpikeSorting.Unit & key).fetch("unit_id"))
        if file_unit_ids != unit_ids:
            raise ValueError(
                f"Sort group {key['sort_group_id']}: analysis file units {file_unit_ids} "
                f"differ from CuratedSpikeSorting.Unit {unit_ids} (or are out of order)"
            )
        if units is not None:
            spike_times.extend(np.asarray(st, dtype=np.float64) for st in units["spike_times"])
    return spike_times


def filter_spike_times(
    spike_times: Sequence[NDArray[np.float64]],
    position_time: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Restrict each unit's spikes to the position time range (inclusive).

    Same bounds as ``continuum-swr-replay``'s ``filter_spike_times``, but units
    left with no spikes are kept, so the unit list stays aligned with the sort.

    Parameters
    ----------
    spike_times : sequence of np.ndarray, shape (n_spikes,)
        Per-unit spike times (seconds).
    position_time : np.ndarray, shape (n_time,)
        Position timestamps (seconds); only the first and last are used.

    Returns
    -------
    list of np.ndarray, shape (n_spikes_in_range,)
        Spikes with ``position_time[0] <= t <= position_time[-1]``, one array
        per input unit.
    """
    start, end = position_time[0], position_time[-1]
    return [st[(st >= start) & (st <= end)] for st in spike_times]


def fetch_figure04_inputs(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME,
    epoch_name: str = FIGURE04_EPOCH_NAME,
) -> Figure4Inputs:
    """Fetch the Figure-4 inputs from Spyglass.

    Parameters
    ----------
    nwb_file_name : str, optional
        Spyglass NWB file name. Default is the Figure-4 session.
    epoch_name : str, optional
        Epoch interval name. Default is the Figure-4 epoch.

    Returns
    -------
    Figure4Inputs
        Position, spike times clipped to the position time range, and the track
        graph with its linearization parameters.
    """
    pos_name = get_position_interval_name(nwb_file_name, epoch_name)
    position = get_position_info(nwb_file_name, epoch_name, pos_name)
    position_time = position["position_info"].index.to_numpy()
    spike_times = filter_spike_times(get_hpc_sorted_spike_times(nwb_file_name), position_time)
    return Figure4Inputs(spike_times=spike_times, **position)


def export_file_path(output_dir: str | Path, animal_date_epoch: str) -> Path:
    """Return the path of the ``.npz`` input file for an epoch.

    Parameters
    ----------
    output_dir : str or Path
        Directory holding the file.
    animal_date_epoch : str
        Epoch identifier (see :func:`epoch_identifier`).

    Returns
    -------
    Path
        ``{output_dir}/{animal_date_epoch}_figure04_inputs.npz``.
    """
    (suffix,) = EXPORT_FILE_SUFFIXES
    return Path(output_dir) / f"{animal_date_epoch}{suffix}"


def check_output_paths(
    output_dir: str | Path,
    animal_date_epoch: str,
    *,
    reference_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    """Check the export destination (and a comparison reference) before any work.

    Parameters
    ----------
    output_dir : str or Path
        Directory the input file will be written to.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.
    reference_dir : str or Path, optional
        Directory of the reference input file the new one will be compared with.
    overwrite : bool, optional
        Whether an existing file in ``output_dir`` may be replaced. Default False.

    Raises
    ------
    ValueError
        If ``reference_dir`` is ``output_dir``: the reference would be overwritten
        and then compared with itself.
    FileExistsError
        If the input file already exists in ``output_dir`` and ``overwrite`` is False.
    FileNotFoundError
        If the reference input file is missing.
    """
    if reference_dir is not None and Path(reference_dir).resolve() == Path(output_dir).resolve():
        raise ValueError(f"Output and reference directory are the same: {output_dir}")
    output = export_file_path(output_dir, animal_date_epoch)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    if reference_dir is not None:
        reference = export_file_path(reference_dir, animal_date_epoch)
        if not reference.is_file():
            raise FileNotFoundError(f"Missing reference export: {reference}")


def write_figure04_inputs(
    inputs: Figure4Inputs,
    output_dir: str | Path,
    animal_date_epoch: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the Figure-4 inputs as the ``.npz`` file the figure reads.

    The inputs are validated with the loader's checks, encoded with
    :func:`~statespacecheck_paper.load_local_data.recording_arrays`, and written
    deterministically, so the same inputs always give the same SHA-256.

    Parameters
    ----------
    inputs : Figure4Inputs
        Fetched inputs.
    output_dir : str or Path
        Destination directory (created if missing).
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.
    overwrite : bool, optional
        Replace an existing file. Default False.

    Returns
    -------
    Path
        The written file.

    Raises
    ------
    FileExistsError
        If the file exists and ``overwrite`` is False.
    ValueError
        If the inputs fail the loader's checks
        (:class:`~statespacecheck_paper.load_local_data.NeuralRecordingData`) or
        cannot be encoded; nothing is written.
    """
    check_output_paths(output_dir, animal_date_epoch, overwrite=overwrite)
    # Validate with the loader's contract before writing anything.
    NeuralRecordingData(
        position_info=inputs.position_info,
        spike_times=tuple(inputs.spike_times),
        track_graph=inputs.track_graph,
        linear_edge_order=tuple(inputs.linear_edge_order),
        linear_edge_spacing=inputs.linear_edge_spacing,
    )
    arrays = recording_arrays(
        inputs.position_info,
        inputs.spike_times,
        inputs.track_graph,
        inputs.linear_edge_order,
        inputs.linear_edge_spacing,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return write_npz(export_file_path(output_dir, animal_date_epoch), arrays)


def _array_difference(reference: NDArray[np.generic], candidate: NDArray[np.generic]) -> str | None:
    if reference.dtype != candidate.dtype:
        return f"dtype {reference.dtype} vs {candidate.dtype}"
    if reference.shape != candidate.shape:
        return f"shape {reference.shape} vs {candidate.shape}"
    same = np.array_equal(
        reference, candidate, equal_nan=np.issubdtype(reference.dtype, np.inexact)
    )
    return None if same else "values differ"


def compare_figure04_exports(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    animal_date_epoch: str,
) -> dict[str, str | None]:
    """Compare two ``.npz`` input files array by array.

    Every array must match exactly in dtype, shape, and values (NaN equal to NaN).

    Parameters
    ----------
    reference_dir : str or Path
        Directory with the reference input file (e.g. the one the figure used).
    candidate_dir : str or Path
        Directory with the input file to check.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.

    Returns
    -------
    dict of str to str or None
        Array name → ``None`` if identical, otherwise what differs (including an
        array present in only one file).
    """
    reference_path = export_file_path(reference_dir, animal_date_epoch)
    candidate_path = export_file_path(candidate_dir, animal_date_epoch)
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        differences: dict[str, str | None] = {}
        for name in sorted(set(reference.files) | set(candidate.files)):
            if name not in candidate.files:
                differences[name] = "only in the reference"
            elif name not in reference.files:
                differences[name] = "only in the candidate"
            else:
                differences[name] = _array_difference(reference[name], candidate[name])
    return differences


def print_export_comparison(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    animal_date_epoch: str,
) -> bool:
    """Print the arrays :func:`compare_figure04_exports` finds different.

    Parameters
    ----------
    reference_dir : str or Path
        Directory with the reference input file.
    candidate_dir : str or Path
        Directory with the input file to check.
    animal_date_epoch : str
        Epoch identifier used as the file-name prefix.

    Returns
    -------
    bool
        Whether every array is identical.
    """
    differences = compare_figure04_exports(reference_dir, candidate_dir, animal_date_epoch)
    different = {name: why for name, why in differences.items() if why is not None}
    for name, why in different.items():
        print(f"DIFFERENT  {name}: {why}")
    if not different:
        print(f"identical  all {len(differences)} arrays")
    return not different


def figure04_diagnostics_from_decodes(
    continuous_model: Any,
    contfrag_model: Any,
    continuous_results: xr.Dataset,
    contfrag_results: xr.Dataset,
    spike_times: Sequence[NDArray[np.float64]],
    *,
    coverage: float,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[SpikeEventDiagnostics, SpikeEventDiagnostics, Figure4Summary]:
    """Compute Figure 4's per-spike diagnostics and summary from two stored decodes.

    The same computation as the figure pipeline, applied to decodes made
    elsewhere (e.g. by Spyglass ``SortedSpikesDecodingV1``): spikes are clipped to
    the decoded time range, the per-spike likelihood uses the Continuous
    decoder's place fields (which must equal the Continuous-Fragmented ones), and
    the summary uses the Figure-4 flag thresholds.

    Parameters
    ----------
    continuous_model, contfrag_model : non_local_detector model
        Fitted Continuous and Continuous-Fragmented decoders.
    continuous_results, contfrag_results : xr.Dataset
        Their decodes over the same time bins; each must hold
        ``predictive_posterior`` (request it with ``return_outputs``).
    spike_times : sequence of np.ndarray, shape (n_spikes,)
        Per-unit spike times (seconds), in the order the models were fitted with.
        Checked against each unit's fitted mean rate, which assumes the models
        were trained on every decoded time bin (as in Figure 4).
    coverage : float
        HPD coverage for the HPD-overlap diagnostic.
    thresholds : Mapping of str to float, optional
        Flag threshold per metric. Default: the Figure-4 thresholds.

    Notes
    -----
    Needs the figure pipeline's dependencies (e.g. ``statespacecheck``); they are
    imported here so the fetch functions above do not need them.

    Returns
    -------
    continuous, contfrag : SpikeEventDiagnostics
        Per-spike diagnostics of each decoder, on the same spikes.
    summary : Figure4Summary
        Whole-session event means and two-decoder flag agreement.

    Raises
    ------
    ValueError
        If a decode lacks ``predictive_posterior``, the decodes cover different
        time bins, the spike trains do not match the fitted units (count or
        order), or the two decoders' place fields differ.
    """
    from statespacecheck_paper.figure04_decoder import get_spike_counts
    from statespacecheck_paper.figure04_diagnostics import compute_results_diagnostics
    from statespacecheck_paper.figure04_generation import (
        FIGURE4_DIAGNOSTIC_THRESHOLDS,
        FIGURE4_METRIC_DIRECTIONS,
    )
    from statespacecheck_paper.figure04_place_fields import extract_shared_position_place_fields
    from statespacecheck_paper.figure04_workflow import summarize_figure04_diagnostics

    time = continuous_results["time"].to_numpy()
    if not np.array_equal(time, contfrag_results["time"].to_numpy()):
        raise ValueError("The two decodes cover different time bins")
    for name, results in (("Continuous", continuous_results), ("ContFrag", contfrag_results)):
        if "predictive_posterior" not in results:
            raise ValueError(
                f"The {name} decode has no predictive_posterior; decode with "
                "return_outputs including 'predictive_posterior'"
            )
    spikes = filter_spike_times(spike_times, time)
    expected_rates = np.array([len(st) for st in spikes]) / len(time)
    for model in (continuous_model, contfrag_model):
        (encoding_model,) = model.encoding_model_.values()
        fitted_rates = np.asarray(encoding_model["mean_rates"])
        if fitted_rates.shape != expected_rates.shape or not np.allclose(
            fitted_rates, expected_rates, rtol=1e-6, atol=0.0
        ):
            raise ValueError("The spike trains do not match the fitted units (count or order)")
    place_fields, position_bins = extract_shared_position_place_fields(continuous_model)
    contfrag_fields, contfrag_bins = extract_shared_position_place_fields(contfrag_model)
    if not np.allclose(place_fields, contfrag_fields, equal_nan=True) or not np.allclose(
        position_bins, contfrag_bins, equal_nan=True
    ):
        raise ValueError("The Continuous and ContFrag place fields or position grids differ")
    spike_counts = get_spike_counts(spikes, time)
    continuous, contfrag = (
        compute_results_diagnostics(
            results, place_fields, spike_counts, time, spikes, coverage=coverage
        )
        for results in (continuous_results, contfrag_results)
    )
    summary = summarize_figure04_diagnostics(
        continuous,
        contfrag,
        n_units=int(spike_counts.shape[1]),
        thresholds=FIGURE4_DIAGNOSTIC_THRESHOLDS if thresholds is None else thresholds,
        metric_directions=FIGURE4_METRIC_DIRECTIONS,
    )
    return continuous, contfrag, summary


# ``name [= default] : type`` lines of a DataJoint definition (not ``->`` or index lines).
_ATTRIBUTE_LINE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(?:=[^:#\n]*)?:", re.MULTILINE)


def declared_attribute_names(definition: str) -> set[str]:
    """Return the attribute names a DataJoint table definition declares directly.

    Attributes inherited through ``->`` references are not included.

    Parameters
    ----------
    definition : str
        DataJoint table definition.

    Returns
    -------
    set of str
        Declared attribute names.

    Examples
    --------
    >>> sorted(declared_attribute_names('''
    ...     -> master
    ...     table_id: int
    ...     ---
    ...     time=CURRENT_TIMESTAMP: timestamp  # when
    ...     unique index (export_id, table_id)
    ... '''))
    ['table_id', 'time']
    """
    return set(_ATTRIBUTE_LINE.findall(definition))


def check_export_tables_match_spyglass() -> None:
    """Refuse to export with a Spyglass that does not declare the database's export columns.

    The export tables of the lab database can be newer than the installed
    Spyglass (the version in ``uv.lock`` is), and an older Spyglass would write
    rows its successors do not expect. ``Export.make`` also requires packaging
    with the same Spyglass ``x.y.z`` that logged the selection, so a selection
    logged by a Spyglass that cannot package would be stranded. Run this before
    logging as well as before packaging. Read-only.

    Raises
    ------
    RuntimeError
        If a secondary column of the database's ``ExportSelection`` or ``Export``
        tables is not declared by the installed Spyglass.
    """
    from spyglass.common.common_usage import Export, ExportSelection

    tables = (
        ExportSelection,
        ExportSelection.Table,
        ExportSelection.File,
        Export,
        Export.Table,
        Export.File,
    )
    unknown = [
        f"{table.full_table_name}.{name}"
        for table in tables
        for name in table.heading.secondary_attributes
        if name not in declared_attribute_names(table.definition)
    ]
    if unknown:
        raise RuntimeError(
            f"Installed Spyglass predates the database's export tables (undeclared columns: "
            f"{unknown}); run the export with the lab's current Spyglass."
        )


def dry_run_figure04_export_log(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME,
    epoch_name: str = FIGURE04_EPOCH_NAME,
) -> list[dict[str, str]]:
    """Return what an export session would log for the Figure-4 fetch, writing nothing.

    Runs :func:`fetch_figure04_inputs` with Spyglass export logging switched on
    (a placeholder export ID) and the ``ExportSelection.Table`` / ``.File`` inserts
    replaced by a recorder, then restores both. Reads the database and the analysis
    files like a real fetch. This relies on how Spyglass logs exports; if nothing is
    recorded, treat that as a changed mechanism, not as an empty export.

    Parameters
    ----------
    nwb_file_name : str, optional
        Spyglass NWB file name. Default is the Figure-4 session.
    epoch_name : str, optional
        Epoch interval name. Default is the Figure-4 epoch.

    Returns
    -------
    list of dict of str to str
        One entry per row that would be inserted: ``part`` (``"Table"`` or
        ``"File"``) plus the row's fields (e.g. ``table_name`` and
        ``restriction``, or ``analysis_file_name``).
    """
    import os

    from spyglass.common.common_usage import ExportSelection
    from spyglass.utils.mixins.export import EXPORT_ENV_VAR

    recorded: list[dict[str, str]] = []

    def recorder(part: str) -> Any:
        def record(self: Any, rows: Any, *args: Any, **kwargs: Any) -> None:
            rows = [rows] if isinstance(rows, Mapping) else list(rows)
            recorded.extend({"part": part, **{k: str(v) for k, v in r.items()}} for r in rows)

        return record

    parts = {name: getattr(ExportSelection, name) for name in ("Table", "File")}
    originals = {
        (name, method): getattr(part, method)
        for name, part in parts.items()
        for method in ("insert", "insert1")
    }
    previous_export_id = os.environ.get(EXPORT_ENV_VAR)
    try:
        for (name, method), _ in originals.items():
            setattr(parts[name], method, recorder(name))
        os.environ[EXPORT_ENV_VAR] = "999999999"
        fetch_figure04_inputs(nwb_file_name, epoch_name)
    finally:
        if previous_export_id is None:
            os.environ.pop(EXPORT_ENV_VAR, None)
        else:
            os.environ[EXPORT_ENV_VAR] = previous_export_id
        for (name, method), original in originals.items():
            setattr(parts[name], method, original)
    return recorded


def unrestricted_log_entries(entries: Sequence[Mapping[str, str]]) -> list[str]:
    """Return the tables an export log would include whole.

    Spyglass's export logging records ``table & True`` (as in
    ``ensure_single_entry()``, used by ``fetch1_dataframe``) as an unrestricted
    fetch, and packaging then exports the entire table.

    Parameters
    ----------
    entries : sequence of Mapping of str to str
        Log rows, as returned by :func:`dry_run_figure04_export_log`.

    Returns
    -------
    list of str
        Names of tables with an unrestricted log entry.

    Examples
    --------
    >>> unrestricted_log_entries([
    ...     {"part": "Table", "table_name": "a", "restriction": "(True)"},
    ...     {"part": "Table", "table_name": "b", "restriction": "(x=1)"},
    ...     {"part": "File", "analysis_file_name": "f.nwb"},
    ... ])
    ['a']
    """
    return [
        entry["table_name"]
        for entry in entries
        if entry.get("part") == "Table"
        and entry.get("restriction", "").strip("() ") in ("", "True", "1")
    ]


def log_figure04_export(
    paper_id: str,
    analysis_id: str,
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME,
    epoch_name: str = FIGURE04_EPOCH_NAME,
) -> Figure4Inputs:
    """Fetch the Figure-4 inputs inside a new Spyglass export session.

    **Writes to the lab database**: ``ExportSelection.start_export`` inserts a
    selection entry, and every Spyglass fetch until ``stop_export`` is logged
    against it. Before that, :func:`check_export_tables_match_spyglass` runs, the
    ``paper_id`` is checked to be new, and the fetch is rehearsed with
    :func:`dry_run_figure04_export_log` (refusing an export that would include a
    whole table). Packaging is :func:`package_figure04_export`, which must use the
    same Spyglass version.

    Parameters
    ----------
    paper_id : str
        New export paper ID (at most 32 characters).
    analysis_id : str
        Analysis label within the paper (at most 32 characters).
    nwb_file_name : str, optional
        Spyglass NWB file name. Default is the Figure-4 session.
    epoch_name : str, optional
        Epoch interval name. Default is the Figure-4 epoch.

    Returns
    -------
    Figure4Inputs
        The inputs fetched while logging.

    Raises
    ------
    RuntimeError
        From :func:`check_export_tables_match_spyglass`, or if the rehearsal
        records nothing or an unrestricted table.
    ValueError
        If ``paper_id`` already has export selections. Re-starting an existing
        ``(paper_id, analysis_id)`` deletes its packaged ``Export`` entry, and
        packaging rebuilds the paper's one package from all of its selections, so
        this refuses rather than change a previous export.
    """
    from spyglass.common.common_usage import ExportSelection

    check_export_tables_match_spyglass()
    selection = ExportSelection()
    if len(selection & {"paper_id": paper_id}) > 0:
        raise ValueError(f"paper_id {paper_id!r} already has export selections; choose a new one")
    # Rehearse the fetch with logging recorded, not written: this also imports every
    # module the fetch needs, so nothing can fail for that reason mid-export.
    rehearsal = dry_run_figure04_export_log(nwb_file_name, epoch_name)
    if not rehearsal:
        raise RuntimeError("The export rehearsal recorded nothing; Spyglass's logging changed")
    if unrestricted := unrestricted_log_entries(rehearsal):
        raise RuntimeError(f"The export would include whole tables: {sorted(set(unrestricted))}")

    selection.start_export(paper_id=paper_id, analysis_id=analysis_id)
    try:
        return fetch_figure04_inputs(nwb_file_name, epoch_name)
    finally:
        selection.stop_export()


def describe_figure04_export(paper_id: str) -> list[str]:
    """List the tables and files logged for an export (read-only).

    Parameters
    ----------
    paper_id : str
        Export paper ID.

    Returns
    -------
    list of str
        Printable lines: each logged table with its row count, then each file.
    """
    from spyglass.common.common_usage import ExportSelection

    selection = ExportSelection()
    tables = selection.preview_tables(paper_id=paper_id)
    files = selection.list_file_paths({"paper_id": paper_id}, as_dict=False)
    return [
        "Logged tables:",
        *(f"  {table.full_table_name}: {len(table)} rows" for table in tables),
        "Logged files:",
        *(f"  {path}" for path in sorted(files)),
    ]


def package_figure04_export(paper_id: str) -> None:
    """Package a logged export with ``Export().populate_paper``.

    **Writes to the lab database** (``Export`` entries listing the tables and
    files) and to the Spyglass export directory (a ``mysqldump`` script
    ``_ExportSQL_<paper_id>.sh``, the Spyglass version, and ``environment.yml``;
    it may also create ``~/.my.cnf``). Running that script produces the SQL dump.

    Parameters
    ----------
    paper_id : str
        Export paper ID, logged with the installed Spyglass version.

    Raises
    ------
    RuntimeError
        From :func:`check_export_tables_match_spyglass`, or from Spyglass when
        the selection was logged with a different Spyglass version.
    """
    from spyglass.common.common_usage import Export

    check_export_tables_match_spyglass()
    Export().populate_paper(paper_id=paper_id)
