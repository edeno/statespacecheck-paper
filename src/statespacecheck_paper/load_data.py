"""Data loading utilities for NWB files via Spyglass database.

This module provides functions to load neural and behavioral data from
Neurodata Without Borders (NWB) files stored in the Spyglass DataJoint
pipeline. Handles position tracking, spike sorting results, electrode
metadata, and temporal alignment.

The main entry point is `load_data()`, which orchestrates the complete
data loading pipeline including artifact removal and temporal filtering.

Key Features
------------
- Position data loading and linearization for structured environments
- Hippocampal clusterless multi-unit marks for decoding
- Prefrontal cortex sorted single-unit spike times
- Automatic artifact detection and removal
- Temporal alignment between neural and behavioral data
- Electrode group metadata extraction

Data Sources
------------
This module queries several Spyglass database tables:
- IntervalPositionInfo: Position tracking data
- PositionOutput: DLC-tracked position (preferred)
- TrackGraph: Environment structure and linearization
- UnitMarks: Hippocampal clusterless marks
- CuratedSpikeSorting: Prefrontal sorted units
- Nwbfile: NWB file metadata

Examples
--------
Load complete dataset for analysis:

>>> from statespacecheck_paper.load_data import load_data
>>> data = load_data("chimi20200212_.nwb", "01_r1")
>>> position = data["position_info"]["linear_position"].to_numpy()
>>> spikes_hpc = data["spike_times"]["HPC"]

Notes
-----
Requires active connection to Spyglass database and properly configured
NWB file paths. See Spyglass documentation for database setup:
https://github.com/LorenFrankLab/spyglass
"""

import itertools
import warnings
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from datajoint import DataJointError
from scipy.ndimage import label
from spyglass.common import (
    IntervalList,
    Nwbfile,
    PositionIntervalMap,
)
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.decoding.v0.clusterless import UnitMarks
from spyglass.linearization.v0.main import IntervalLinearizedPosition, TrackGraph
from spyglass.position import PositionOutput
from spyglass.spikesorting.v0 import CuratedSpikeSorting, SortGroup
from spyglass.utils.nwb_helper_fn import get_nwb_file
from track_linearization import get_linearized_position

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pynwb")
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint")

NWB_FILES = [
    "chimi20200212_.nwb",
    "chimi20200213_.nwb",
    "chimi20200214_.nwb",
    "chimi20200215_.nwb",
    "chimi20200216_.nwb",
    "chimi20200217_.nwb",
    "chimi20200218_.nwb",
    "chimi20200219_.nwb",
    "chimi20200220_.nwb",
    "chimi20200221_.nwb",
    "chimi20200222_.nwb",
    "chimi20200223_.nwb",
    "chimi20200224_.nwb",
    "chimi20200225_.nwb",
    "chimi20200226_.nwb",
    "chimi20200227_.nwb",
    "chimi20200228_.nwb",
    "chimi20200301_.nwb",
    "chimi20200302_.nwb",
    "chimi20200303_.nwb",
    "chimi20200304_.nwb",
    "chimi20200305_.nwb",
    "chimi20200306_.nwb",
    "chimi20200307_.nwb",
    "chimi20200308_.nwb",
    "chimi20200310_.nwb",
    "chimi20200311_.nwb",
    "chimi20200312_.nwb",
    "chimi20200313_.nwb",
    "j1620210706_.nwb",
    "j1620210707_.nwb",
    "j1620210708_.nwb",
    "j1620210709_.nwb",
    "j1620210710_.nwb",
    "j1620210711_.nwb",
    "j1620210712_.nwb",
    "j1620210713_.nwb",
    "j1620210714_.nwb",
    "j1620210715_.nwb",
    "j1620210716_.nwb",
    "j1620210717_.nwb",
    "j1620210718_.nwb",
    "j1620210719_.nwb",
    "j1620210720_.nwb",
    "j1620210721_.nwb",
    "peanut20201124_.nwb",
    "peanut20201125_.nwb",
    "peanut20201126_.nwb",
    "peanut20201127_.nwb",
    "peanut20201128_.nwb",
    "peanut20201129_.nwb",
    "peanut20201130_.nwb",
    "peanut20201201_.nwb",
    "peanut20201202_.nwb",
    "peanut20201203_.nwb",
    "peanut20201204_.nwb",
    "peanut20201205_.nwb",
    "peanut20201206_.nwb",
    "peanut20201207_.nwb",
    "peanut20201208_.nwb",
    "peanut20201209_.nwb",
    "senor20201027_.nwb",
    "senor20201028_.nwb",
    "senor20201029_.nwb",
    "senor20201030_.nwb",
    "senor20201031_.nwb",
    "senor20201101_.nwb",
    "senor20201102_.nwb",
    "senor20201103_.nwb",
    "senor20201104_.nwb",
    "senor20201105_.nwb",
    "senor20201106_.nwb",
    "senor20201107_.nwb",
    "senor20201108_.nwb",
    "senor20201109_.nwb",
    "senor20201110_.nwb",
    "senor20201111_.nwb",
    "senor20201112_.nwb",
    "senor20201113_.nwb",
    "senor20201114_.nwb",
    "senor20201115_.nwb",
    "senor20201116_.nwb",
    "senor20201117_.nwb",
    "senor20201118_.nwb",
    "senor20201119_.nwb",
    "senor20201120_.nwb",
    "senor20201121_.nwb",
    "wilbur20210326_.nwb",
    "wilbur20210327_.nwb",
    "wilbur20210328_.nwb",
    "wilbur20210329_.nwb",
    "wilbur20210330_.nwb",
    "wilbur20210331_.nwb",
    "wilbur20210401_.nwb",
    "wilbur20210402_.nwb",
    "wilbur20210403_.nwb",
    "wilbur20210404_.nwb",
    "wilbur20210405_.nwb",
    "wilbur20210406_.nwb",
    "wilbur20210407_.nwb",
    "wilbur20210408_.nwb",
    "wilbur20210409_.nwb",
    "wilbur20210410_.nwb",
    "wilbur20210411_.nwb",
    "wilbur20210412_.nwb",
    "wilbur20210413_.nwb",
    "wilbur20210414_.nwb",
    "wilbur20210415_.nwb",
    "wilbur20210416_.nwb",
]


def _get_interpolated_position_info(
    position_info: pd.DataFrame,
    time: np.ndarray,
    track_graph: nx.Graph,
    edge_order: list[tuple[int, int]],
    edge_spacing: list[float],
    position_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Interpolate position data and compute linearized position on track.

    Resamples position data to match neural recording timestamps using linear
    interpolation, then projects 2D position onto a 1D linearized track
    representation based on the track graph structure.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with time index. Contains x/y coordinates in columns.
    time : np.ndarray, shape (n_time,)
        Target timestamps for interpolation (typically neural recording times).
    track_graph : networkx.Graph
        Graph representation of the track environment structure.
    edge_order : list of tuple of int
        Ordered list of graph edges defining the linearization path.
    edge_spacing : list of float
        Distance between nodes for each edge in centimeters.
    position_columns : list of str or None, default None
        Column names for x/y coordinates. If None, uses
        ["head_position_x", "head_position_y"].

    Returns
    -------
    interpolated_position : pd.DataFrame
        Interpolated position data with additional linearized position columns.
        Index is time in seconds. Includes both Cartesian (x, y) and linearized
        (1D) position coordinates.

    Notes
    -----
    Uses linear interpolation to fill gaps between original position samples.
    The linearized position represents distance traveled along the track path,
    useful for 1D decoding on structured environments like linear tracks or
    W-mazes.
    """
    if position_columns is None:
        position_columns = ["head_position_x", "head_position_y"]
    new_index = pd.Index(
        np.unique(np.concatenate((position_info.index, time))),
        name="time",
    )

    interpolated_position_info = (
        position_info.reindex(index=new_index).interpolate(method="linear").reindex(index=time)
    )

    linear_position_info = get_linearized_position(
        position=interpolated_position_info[position_columns].to_numpy(),
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).set_index(interpolated_position_info.index)

    return pd.concat(
        (
            interpolated_position_info,
            linear_position_info,
        ),
        axis=1,
    )


def _get_position_info(nwb_file_name: str, epoch_name: str, pos_name: str) -> dict[str, Any]:
    """Fetch and process position data from NWB file via Spyglass database.

    Retrieves position tracking data for a specific recording epoch from the
    Spyglass DataJoint pipeline. Handles both DLC-tracked position and legacy
    position data, applies temporal filtering, and computes linearized position
    coordinates.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file (e.g., "chimi20200212_.nwb").
    epoch_name : str
        Name of recording epoch (e.g., "01_r1", "02_r2").
    pos_name : str
        Position interval name from PositionIntervalMap table.

    Returns
    -------
    position_data : dict
        Dictionary with keys:
        - "position_info" : pd.DataFrame
            Position data with time index. Columns include head_position_x,
            head_position_y (Cartesian), and linear_position (1D).
        - "linear_edge_order" : list of tuple
            Track graph edge ordering for linearization.
        - "linear_edge_spacing" : list of float
            Spacing along edges in centimeters.
        - "track_graph" : networkx.Graph
            Track environment structure.

    Notes
    -----
    This function queries the Spyglass database tables (IntervalPositionInfo,
    PositionOutput, TrackGraph, IntervalLinearizedPosition) to reconstruct
    the animal's position during the recording epoch.

    Handles multiple data sources:
    - DLCPosV1: DeepLabCut position tracking (preferred)
    - Legacy position data: Fallback for older recordings

    Applies "noPrePostTrialTimes" filtering when available to exclude pre/post
    trial periods from analysis.
    """
    position_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "position_info_param_name": "default_decoding",
    }

    linearization_key = {
        "position_info_param_name": "default_decoding",
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_name,
        "linearization_param_name": "default",
    }

    try:
        track_graph_name = (IntervalLinearizedPosition() & linearization_key).fetch1(
            "track_graph_name"
        )
    except DataJointError:
        track_graph_name = nwb_file_name.split("_")[0]

    track_graph = (TrackGraph() & {"track_graph_name": track_graph_name}).get_networkx_track_graph()
    track_graph_params = (TrackGraph() & {"track_graph_name": track_graph_name}).fetch1()
    linear_edge_order = track_graph_params["linear_edge_order"]
    linear_edge_spacing = track_graph_params["linear_edge_spacing"]

    try:
        epoch = int(epoch_name.split("_")[0])
        pos_merge_id = str(
            (
                PositionOutput().merge_restrict()
                & {
                    "nwb_file_name": nwb_file_name,
                    "source": "DLCPosV1",
                    "epoch": epoch,
                }
            ).fetch1("merge_id")
        )
        position_info = (
            ((PositionOutput() & {"merge_id": pos_merge_id}).fetch1_dataframe().dropna())
            .drop(columns="video_frame_ind")
            .add_prefix("head_")
        )
        time = (IntervalPositionInfo() & position_key).fetch1_dataframe().dropna().index

    except DataJointError:
        position_info = (IntervalPositionInfo() & position_key).fetch1_dataframe().dropna()
        time = position_info.index
    try:
        valid_interval_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": epoch_name + " noPrePostTrialTimes",
            }
        ).fetch1("valid_times")
        position_info = position_info.loc[valid_interval_times[0][0] : valid_interval_times[-1][1]]
    except DataJointError:
        pass

    position_info = _get_interpolated_position_info(
        position_info,
        time,
        track_graph,
        linear_edge_order,
        linear_edge_spacing,
    )

    return {
        "position_info": position_info,
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
        "track_graph": track_graph,
    }


def _get_electrode_group_info(nwb_file_name: str) -> pd.DataFrame:
    """Extract electrode group metadata from NWB file.

    Parses electrode group information including anatomical location and
    description from the NWB file. Standardizes brain region labels to
    canonical names (CA1, mPFC, OFC) for consistent analysis.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file (e.g., "chimi20200212_.nwb").

    Returns
    -------
    electrode_info : pd.DataFrame
        DataFrame with electrode_group_name as index. Columns include:
        - description : str
            Human-readable description of electrode group.
        - location : str
            Anatomical location string.
        - targeted_location : str
            Standardized brain region (CA1, mPFC, OFC, etc.).
        - Additional fields from NWB electrode group metadata.

    Notes
    -----
    Standardizes targeted_location labels:
    - Any location containing "CA1" (excluding CorpusCallosum) → "CA1"
    - Any location containing "mPFC" (excluding CorpusCallosum) → "mPFC"
    - Any location containing "OFC" (excluding CorpusCallosum) → "OFC"

    This standardization enables pooling neurons by brain region regardless
    of minor labeling variations across recording sessions.
    """
    nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
    nwbf = get_nwb_file(nwb_file_abspath)
    electrode_group_df = []
    for electrode_group in nwbf.electrode_groups.values():
        cur = {
            "electrode_group_name": electrode_group.name,
            "description": electrode_group.description,
        }
        cur.update(electrode_group.fields)
        electrode_group_df.append(cur)

    electrode_group_df = (
        pd.DataFrame(electrode_group_df).drop(columns=["device"]).set_index("electrode_group_name")
    )

    is_ca1 = electrode_group_df.targeted_location.str.contains("CA1") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_ca1, "targeted_location"] = "CA1"

    is_mpfc = electrode_group_df.targeted_location.str.contains("mPFC") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_mpfc, "targeted_location"] = "mPFC"

    is_ofc = electrode_group_df.targeted_location.str.contains("OFC") & (
        electrode_group_df.location != "CorpusCallosum"
    )
    electrode_group_df.loc[is_ofc, "targeted_location"] = "OFC"

    return electrode_group_df


def _detect_coincident_spikes(
    spike_times: list[np.ndarray],
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect and remove artifactual coincident spikes across recording groups.

    Identifies and filters spikes that occur simultaneously across multiple
    electrodes/tetrodes, which likely represent electrical artifacts rather
    than genuine neural activity. Uses temporal proximity and cross-electrode
    coincidence rate to flag artifacts.

    Parameters
    ----------
    spike_times : list of np.ndarray
        List of spike time arrays, one per recording group/tetrode.
        Each array contains spike times in seconds.
    spike_closeness_threshold : float, default 0.00004
        Maximum time difference (in seconds) for spikes to be considered
        coincident. Default 40 microseconds (0.04 ms).
    max_coincident_fraction : float, default 0.33
        Maximum fraction of recording groups that can have coincident spikes
        before flagging as artifact. If more than this fraction of groups have
        simultaneous spikes, all spikes in that time window are removed.

    Returns
    -------
    filtered_spike_times : list of np.ndarray
        Filtered spike times with artifacts removed, one array per group.
    filtered_indices : list of np.ndarray
        Boolean indices indicating which original spikes were retained,
        one array per group.

    Notes
    -----
    Artifacts often appear as simultaneous multi-unit activity across many
    electrodes due to electrical noise, movement artifacts, or EMG contamination.
    This function removes such events by detecting spikes that occur within
    the `spike_closeness_threshold` window across more than
    `max_coincident_fraction` of recording groups.

    The default threshold of 40 microseconds is much shorter than typical
    neural propagation times between brain regions, making it specific to
    non-physiological coincidences.
    """
    # Concatenate all spike times
    concat_spike_times = np.concatenate(spike_times)
    # Create group IDs for each spike time
    sort_group_id = np.concatenate(
        [np.ones(len(spike_time), dtype=int) * i for i, spike_time in enumerate(spike_times)]
    )
    time_bin_ind = np.concatenate(
        [np.arange(len(spike_time), dtype=int) for spike_time in spike_times]
    )

    # Sort spike times and group IDs based on the spike times
    sort_ind = np.argsort(concat_spike_times)
    sorted_spike_times = concat_spike_times[sort_ind]
    sort_group_id = sort_group_id[sort_ind]
    time_bin_ind = time_bin_ind[sort_ind]

    # Find differences and label close spikes
    is_close = np.diff(sorted_spike_times) < spike_closeness_threshold
    is_close = np.concatenate([[False], is_close])
    labels, _ = label(is_close)

    # Create a DataFrame for further analysis
    df = pd.DataFrame(
        {
            "labels": labels,
            "sort_group_id": sort_group_id,
            "time_bin_ind": time_bin_ind,
            "spike_times": sorted_spike_times,
        },
    )
    # Calculate the fraction of each group and the median spike times
    n_sort_groups = len(spike_times)
    frac = df.loc[df.labels > 0].groupby("labels").sort_group_id.nunique() / n_sort_groups
    frac = frac[frac > max_coincident_fraction]

    df = df.loc[~df.labels.isin(frac.index)]

    filtered_spike_times = df.groupby("sort_group_id").spike_times.apply(np.array).tolist()
    filtered_time_bin_ind = df.groupby("sort_group_id").time_bin_ind.apply(np.array).tolist()

    return filtered_spike_times, filtered_time_bin_ind


def _get_hpc_marks(nwb_file_name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Fetch hippocampal clusterless spike marks from Spyglass database.

    Retrieves spike times and waveform features for hippocampal recordings
    using the clusterless decoding approach. Fetches multi-unit activity
    detected via amplitude thresholding, then filters artifactual coincident
    spikes.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file (e.g., "chimi20200212_.nwb").

    Returns
    -------
    spike_times : list of np.ndarray
        Spike times in seconds for each tetrode, after artifact removal.
    spike_waveform_features : list of np.ndarray
        Spike waveform features (marks) for each tetrode, corresponding to
        retained spike times. Typically 4D features per spike (amplitude on
        4 channels).

    Notes
    -----
    Attempts two parameter sets for clusterless thresholding:
    1. "clusterless_fixed" (preferred)
    2. "default_clusterless" (fallback)

    Uses team "ac_em_xs" preprocessing parameters specific to Frank lab
    hippocampal tetrode recordings.

    Clusterless decoding uses multi-unit marks rather than sorted single units,
    providing more information about local population activity.

    Raises
    ------
    ValueError
        If no hippocampal marks found in database for this file.
    """
    try:
        restriction = {
            "nwb_file_name": nwb_file_name,
            "curation_id": 0,
            "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
            "preproc_params_name": "franklab_tetrode_hippocampus",
            "team_name": "ac_em_xs",
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "clusterless_fixed",
            "mark_param_name": "default",
        }
        marks = (UnitMarks & restriction).fetch_dataframe()
        marks = [(mark.index.to_numpy(), mark.to_numpy()) for mark in marks]
        spike_times_tuple, spike_waveform_features_tuple = zip(*marks, strict=False)
        spike_times = list(spike_times_tuple)
        spike_waveform_features = list(spike_waveform_features_tuple)
    except ValueError:
        restriction = {
            "nwb_file_name": nwb_file_name,
            "curation_id": 0,
            "sort_interval_name": "runs_noPrePostTrialTimes raw data valid times",
            "preproc_params_name": "franklab_tetrode_hippocampus",
            "team_name": "ac_em_xs",
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default_clusterless",
            "mark_param_name": "default",
        }
        marks = (UnitMarks & restriction).fetch_dataframe()
        marks = [(mark.index.to_numpy(), mark.to_numpy()) for mark in marks]
        spike_times_tuple, spike_waveform_features_tuple = zip(*marks, strict=False)
        spike_times = list(spike_times_tuple)
        spike_waveform_features = list(spike_waveform_features_tuple)

    spike_times, filtered_time_bin_ind = _detect_coincident_spikes(spike_times)
    spike_waveform_features = [
        features[ind]
        for ind, features in zip(filtered_time_bin_ind, spike_waveform_features, strict=False)
    ]

    return spike_times, spike_waveform_features


def _get_pfc_spike_times(nwb_file_name: str, brain_area: str) -> list[np.ndarray]:
    """Fetch prefrontal cortex spike times from curated spike sorting.

    Retrieves sorted single-unit spike times from prefrontal cortex recordings
    (mPFC or OFC) using MountainSort4 spike sorting results. Automatically
    uses the most recent curation and filters by anatomical location.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file (e.g., "chimi20200212_.nwb").
    brain_area : str
        Target brain region, either "mPFC" (medial prefrontal cortex) or
        "OFC" (orbitofrontal cortex).

    Returns
    -------
    spike_times : list of np.ndarray
        Spike times in seconds for all sorted units in the target brain area.
        Each array corresponds to one unit. Empty list if no units found.

    Notes
    -----
    Uses MountainSort4 spike sorting with Frank lab probe parameters:
    - Preprocessing: "default"
    - Sorter: "mountainsort4"
    - Sorter params: "franklab_probe_ctx_30KHz_115rad_new_mountainsort2"
    - Team: "ac_em_xs"

    Automatically selects the latest curation_id for each recording, ensuring
    the most up-to-date manual spike sorting curation is used.

    Filters units by anatomical location using electrode group information,
    only returning units from the specified brain area.

    Raises
    ------
    ValueError
        If no curated spikes found for this brain area in this file.
    """
    restriction = {
        "nwb_file_name": nwb_file_name,
        "preproc_params_name": "default",
        "sort_interval_name": "sleeps_runs_noPrePostTrialTimes raw data valid times",
        "sorter_params_name": "franklab_probe_ctx_30KHz_115rad_new_mountainsort2",
        "team_name": "ac_em_xs",
        "sorter": "mountainsort4",
    }

    # Get the latest curation_id
    max_curation_id = (CuratedSpikeSorting & restriction).fetch("curation_id").max()
    restriction.update({"curation_id": max_curation_id})
    # “ampl_2000_z_30_prop_075_1ms” for artifact_params_name

    # Find brain area
    curated_spikes_info = pd.DataFrame(
        (CuratedSpikeSorting & restriction) * SortGroup.SortGroupElectrode
    )
    electrode_group_df = _get_electrode_group_info(nwb_file_name)
    curated_spikes_info = pd.merge(
        curated_spikes_info, electrode_group_df, on="electrode_group_name"
    )
    spikesorting_keys = pd.merge(
        pd.DataFrame(CuratedSpikeSorting() & restriction),
        curated_spikes_info.groupby("sort_group_id").targeted_location.first(),
        on="sort_group_id",
    )
    spikesorting_keys = spikesorting_keys.loc[
        spikesorting_keys.targeted_location == brain_area
    ].to_dict(orient="records")

    nwb_pfc = (CuratedSpikeSorting() & spikesorting_keys).fetch_nwb()

    return list(
        itertools.chain.from_iterable(
            [file["units"]["spike_times"].to_list() for file in nwb_pfc if "units" in file]
        )
    )


def _get_spike_data(nwb_file_name: str) -> dict[str, dict[str, Any]]:
    """Aggregate spike data from all recorded brain regions.

    Fetches spike times and waveform features from all available brain regions
    in the recording (hippocampus, mPFC, OFC). Combines clusterless marks
    (hippocampus) and sorted spikes (prefrontal regions) into a unified data
    structure.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file (e.g., "chimi20200212_.nwb").

    Returns
    -------
    spike_data : dict
        Dictionary with keys:
        - "spike_times" : dict
            Spike times by brain region. Keys are "HPC", "mPFC", "OFC".
            Values are lists of np.ndarray (one per unit/tetrode).
        - "spike_waveform_features" : dict
            Waveform features by brain region. Only populated for "HPC"
            (clusterless marks). Keys are brain region names, values are
            lists of np.ndarray.

    Notes
    -----
    Not all recordings contain all brain regions. The returned dictionaries
    only include keys for regions where data was successfully fetched.

    For hippocampus (HPC):
    - Uses clusterless multi-unit marks
    - Includes waveform features for decoding

    For prefrontal regions (mPFC, OFC):
    - Uses sorted single units
    - No waveform features (just spike times)

    Silently skips brain regions with no available data (returns empty dict
    entries rather than raising errors).
    """
    spike_times = dict()
    spike_waveform_features = dict()

    try:
        spike_times["HPC"], spike_waveform_features["HPC"] = _get_hpc_marks(nwb_file_name)
    except ValueError:
        pass

    for brain_area in ["mPFC", "OFC"]:
        try:
            spike_times[brain_area] = _get_pfc_spike_times(nwb_file_name, brain_area)
            if len(spike_times[brain_area]) < 1:
                del spike_times[brain_area]
        except ValueError:
            pass

    return {
        "spike_times": spike_times,
        "spike_waveform_features": spike_waveform_features,
    }


def _filter_spike_times(
    spike_times: dict[str, Any],
    spike_waveform_features: dict[str, Any],
    position_time: np.ndarray,
) -> None:
    """Filter spike times to match position data temporal bounds.

    Removes spikes occurring outside the position tracking interval, ensuring
    temporal alignment between neural and behavioral data. Modifies dictionaries
    in-place.

    Parameters
    ----------
    spike_times : dict
        Spike times by brain region (modified in-place). Keys are brain region
        names ("HPC", "mPFC", "OFC"), values are lists of np.ndarray.
    spike_waveform_features : dict
        Waveform features by brain region (modified in-place). Keys are brain
        region names, values are lists of np.ndarray.
    position_time : np.ndarray, shape (n_time,)
        Position data timestamps in seconds. Spikes outside
        [position_time[0], position_time[-1]] are removed.

    Returns
    -------
    None
        Modifies spike_times and spike_waveform_features dictionaries in-place.

    Notes
    -----
    This function ensures that all spike data is temporally aligned with
    position tracking. Common scenarios requiring filtering:
    - Neural recording starts before position tracking
    - Position tracking ends before neural recording
    - Gaps in position tracking due to LED occlusion

    Both spike times and corresponding waveform features are filtered using
    the same boolean mask to maintain alignment.
    """
    for brain_area, brain_area_spike_times in spike_times.items():
        filtered_spike_times = []
        filtered_spike_waveform_features = []
        for sort_ind, sort_group_spike_times in enumerate(brain_area_spike_times):
            is_in_bounds = np.logical_and(
                sort_group_spike_times >= position_time[0],
                sort_group_spike_times <= position_time[-1],
            )
            filtered_spike_times.append(sort_group_spike_times[is_in_bounds])

            try:
                filtered_spike_waveform_features.append(
                    spike_waveform_features[brain_area][sort_ind][is_in_bounds]
                )
            except KeyError:
                pass

        spike_times[brain_area] = filtered_spike_times

        if filtered_spike_waveform_features:
            spike_waveform_features[brain_area] = filtered_spike_waveform_features


def load_data(
    nwb_file_name: str,
    epoch_name: str,
) -> dict[str, Any]:
    """Load complete dataset for a recording epoch from Spyglass database.

    Main data loading function that retrieves and processes all data needed for
    analysis: position tracking, spike times, waveform features, track geometry,
    and electrode metadata. Handles temporal alignment and artifact removal
    automatically.

    Parameters
    ----------
    nwb_file_name : str
        Name of NWB file containing the recording session. Must be one of the
        files in NWB_FILES constant (e.g., "chimi20200212_.nwb").
    epoch_name : str
        Name of recording epoch within the session (e.g., "01_r1", "02_r2").
        Typically corresponds to a single behavioral trial or run.

    Returns
    -------
    data : dict
        Comprehensive dataset dictionary with keys:

        Position data:
        - "position_info" : pd.DataFrame
            Time-indexed position data with columns for Cartesian (x, y) and
            linearized (1D) coordinates.
        - "track_graph" : networkx.Graph
            Track environment structure.
        - "linear_edge_order" : list of tuple
            Edge ordering for track linearization.
        - "linear_edge_spacing" : list of float
            Distance between nodes (cm) for each edge.

        Neural data:
        - "spike_times" : dict
            Spike times by brain region. Keys: "HPC", "mPFC", "OFC".
            Values: lists of np.ndarray (one per unit/tetrode).
        - "spike_waveform_features" : dict
            Waveform features for clusterless decoding (HPC only).
            Keys: brain region names. Values: lists of np.ndarray.

        Metadata:
        - "electrode_group_info" : pd.DataFrame
            Electrode group information including anatomical locations.

    Notes
    -----
    This function orchestrates the complete data loading pipeline:

    1. Fetch electrode metadata from NWB file
    2. Query Spyglass database for position and neural data
    3. Interpolate position to match neural timestamps
    4. Compute linearized position coordinates
    5. Filter artifactual coincident spikes (HPC)
    6. Temporally align spikes with position tracking

    The returned data structure is ready for decoding analysis without
    additional preprocessing.

    Examples
    --------
    Load data for a single epoch:

    >>> data = load_data("chimi20200212_.nwb", "01_r1")
    >>> position = data["position_info"]["linear_position"].to_numpy()
    >>> hpc_spikes = data["spike_times"]["HPC"]
    >>> print(f"Position shape: {position.shape}")
    >>> print(f"Number of HPC units: {len(hpc_spikes)}")

    See Also
    --------
    _get_position_info : Position data loading
    _get_spike_data : Neural data loading
    _filter_spike_times : Temporal alignment
    """
    electrode_group_info = _get_electrode_group_info(nwb_file_name)

    pos_name = (
        PositionIntervalMap & {"nwb_file_name": nwb_file_name, "interval_list_name": epoch_name}
    ).fetch1("position_interval_name")
    position_data = _get_position_info(nwb_file_name, epoch_name, pos_name)

    spike_data = _get_spike_data(nwb_file_name)
    # Filter spike times by the start and end time of the position data
    _filter_spike_times(
        spike_data["spike_times"],
        spike_data["spike_waveform_features"],
        position_data["position_info"].index,
    )

    return {
        **position_data,
        **spike_data,
        "electrode_group_info": electrode_group_info,
    }
