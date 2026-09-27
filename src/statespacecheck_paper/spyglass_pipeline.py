"""Spyglass pipeline for the Figure-4 decoding and per-spike diagnostics.

Figure 4 decodes one epoch with two ``non_local_detector`` models and scores every
spike with three diagnostics. This module puts that analysis in the lab's Spyglass
pipeline so it can be reproduced and exported with the rest of the lab's data:

1. Existing Spyglass tables hold the decode: a ``PositionGroup`` (the v0
   ``IntervalPositionInfo`` entry, not upsampled), a ``SortedSpikesGroup`` (the v0
   HPC sort, first registered in ``SpikeSortingOutput``), two
   ``DecodingParameters`` (the Figure-4 models, also returning the one-step
   predictive distribution the diagnostics need), and two
   ``SortedSpikesDecodingSelection`` entries (encoding = decoding = the epoch
   interval, parameter estimation off), populated by ``SortedSpikesDecodingV1``.
2. The custom schema :data:`SCHEMA_NAME` holds the diagnostics:
   :class:`Figure4DiagnosticsParameters`, :class:`Figure4DiagnosticsSelection`
   (the two decodes), and :class:`Figure4Diagnostics` (per-spike diagnostics in an
   analysis NWB file, and the summary in part tables).

With these settings the decode reproduces the figure pipeline exactly (checked
offline by emulating ``SortedSpikesDecodingV1``; see ``docs/spyglass-pipeline.md``).
Leaving parameter estimation on (the Spyglass default) re-fits the ContFrag
transition matrix and changes the Figure-4 results.

**Importing this module imports Spyglass, which connects to the lab database**, so
nothing in figure generation imports it. It creates nothing on import: the custom
schema exists only after :func:`activate_schema` with ``create=True``. The
``*_entries`` functions only read; each ``create_*``, ``insert_*``, and
``populate_*`` function writes to the database. ``scripts/spyglass_pipeline_figure04.py``
runs them one step at a time, as a dry run unless asked to write.

``SortedSpikesDecodingV1`` in current Spyglass fails on single-state decoders such
as the Continuous model (its result assembly needs a one-dimensional ``states``
coordinate); that decode needs a Spyglass with the fix.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import datajoint as dj
import numpy as np
import pandas as pd
from spyglass.common import AnalysisNwbfile
from spyglass.decoding.decoding_merge import DecodingOutput
from spyglass.utils import SpyglassMixin, SpyglassMixinPart

from statespacecheck_paper.figure04_decoder import (
    Figure4Config,
    build_decoder_models,
    create_decoder_environment,
)
from statespacecheck_paper.figure04_generation import FIGURE4_DIAGNOSTIC_THRESHOLDS
from statespacecheck_paper.spyglass_data import (
    FIGURE04_EPOCH_NAME,
    FIGURE04_NWB_FILE_NAME,
    HPC_SORTING_RESTRICTION,
    POSITION_INFO_PARAM_NAME,
    figure04_diagnostics_from_decodes,
    get_position_interval_name,
    get_track_graph,
)

SCHEMA_NAME = "edeno_statespacecheck"
GROUP_NAME = "statespacecheck_figure04"
UNIT_FILTER_PARAMS_NAME = "all_units"
DECODING_PARAM_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "continuous": "statespacecheck_figure04_continuous",
        "contfrag": "statespacecheck_figure04_contfrag",
    }
)
# Outputs the diagnostics need; Spyglass passes decoding_kwargs through to predict().
DECODE_OUTPUTS = ("filter", "predictive_posterior", "log_likelihood")

schema = dj.Schema()  # activated only by activate_schema()


def activate_schema(*, create: bool = False) -> None:
    """Connect the custom tables to :data:`SCHEMA_NAME`.

    Parameters
    ----------
    create : bool, optional
        Create the schema and any missing tables (**writes to the database**,
        including the default :class:`Figure4DiagnosticsParameters` entry).
        Default False: the schema must already exist.
    """
    schema.activate(SCHEMA_NAME, create_schema=create, create_tables=create)


@schema
class Figure4DiagnosticsParameters(SpyglassMixin, dj.Lookup):
    """HPD coverage and flag thresholds for the Figure-4 diagnostics."""

    definition = """
    figure4_diagnostics_param_name: varchar(32)
    ---
    hpd_coverage: double                 # coverage of the highest-density regions
    hpd_overlap_threshold: double        # flag when HPD overlap <= this
    predictive_pvalue_threshold: double  # flag when predictive p-value <= this
    """
    contents = [
        (
            "figure04",
            Figure4Config().diagnostics.hpd_coverage,
            FIGURE4_DIAGNOSTIC_THRESHOLDS["hpd_overlap"],
            FIGURE4_DIAGNOSTIC_THRESHOLDS["predictive_pvalue"],
        )
    ]


@schema
class Figure4DiagnosticsSelection(SpyglassMixin, dj.Manual):
    """The Continuous and Continuous-Fragmented decodes to diagnose together."""

    definition = """
    -> DecodingOutput.proj(continuous_merge_id="merge_id")
    -> DecodingOutput.proj(contfrag_merge_id="merge_id")
    -> Figure4DiagnosticsParameters
    """


@schema
class Figure4Diagnostics(SpyglassMixin, dj.Computed):
    """Per-spike diagnostics of both decoders, and the Figure-4 summary."""

    definition = """
    -> Figure4DiagnosticsSelection
    ---
    -> AnalysisNwbfile
    events_object_id: varchar(40)  # per-spike diagnostics table
    n_units: int
    n_events: int
    """

    class Mean(SpyglassMixinPart):
        """Mean of a diagnostic over all spikes, per decoder."""

        definition = """
        -> master
        model: varchar(24)   # continuous | continuous_fragmented
        metric: varchar(24)  # hpd_overlap | kl_divergence | predictive_pvalue
        ---
        value: double
        """

    class FlagConfusion(SpyglassMixinPart):
        """Spikes flagged by each decoder, for one metric."""

        definition = """
        -> master
        metric: varchar(24)
        ---
        threshold: double
        n: int
        both: int
        continuous_only: int
        contfrag_only: int
        neither: int
        """

    def make(self, key: dict[str, Any]) -> None:
        """Diagnose both decodes with the figure pipeline's code and store the results."""
        params = (Figure4DiagnosticsParameters() & key).fetch1()
        continuous_key = {"merge_id": key["continuous_merge_id"]}
        contfrag_key = {"merge_id": key["contfrag_merge_id"]}
        continuous_results = DecodingOutput.fetch_results(continuous_key)
        continuous, contfrag, summary = figure04_diagnostics_from_decodes(
            DecodingOutput.fetch_model(continuous_key),
            DecodingOutput.fetch_model(contfrag_key),
            continuous_results,
            DecodingOutput.fetch_results(contfrag_key),
            DecodingOutput.fetch_spike_data(continuous_key, filter_by_interval=False),
            coverage=params["hpd_coverage"],
            thresholds={
                "hpd_overlap": params["hpd_overlap_threshold"],
                "predictive_pvalue": params["predictive_pvalue_threshold"],
            },
        )
        time = continuous_results["time"].to_numpy()
        events = pd.DataFrame(
            {
                "time": time[continuous.event_time_ind],
                "unit_index": continuous.event_cell_ind,
                **{
                    f"{model}_{metric}": getattr(diagnostics, f"event_{metric}")
                    for model, diagnostics in (("continuous", continuous), ("contfrag", contfrag))
                    for metric in ("hpd_overlap", "kl_divergence", "predictive_pvalue")
                },
            }
        )
        nwb_file_name = (DecodingOutput.SortedSpikesDecodingV1 & continuous_key).fetch1(
            "nwb_file_name"
        )
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            events_object_id = builder.add_nwb_object(events, "figure04_spike_diagnostics")
            analysis_file_name = builder.analysis_file_name

        self.insert1(
            {
                **key,
                "analysis_file_name": analysis_file_name,
                "events_object_id": events_object_id,
                "n_units": summary.n_units,
                "n_events": len(events),
            }
        )
        self.Mean.insert(
            {**key, "model": model, "metric": metric, "value": value}
            for model, means in (
                ("continuous", summary.continuous),
                ("continuous_fragmented", summary.continuous_fragmented),
            )
            for metric, value in dataclasses.asdict(means).items()
        )
        self.FlagConfusion.insert(
            {
                **key,
                "metric": c.metric,
                "threshold": c.threshold,
                "n": c.n,
                "both": c.both,
                "continuous_only": c.a_only,
                "contfrag_only": c.b_only,
                "neither": c.neither,
            }
            for c in summary.flag_confusions
        )

    def fetch_reported_statistics(self) -> dict[str, Any]:
        """Return one entry's summary in the shape of ``figure04_summary.json``."""
        key = self.fetch1("KEY")
        means: dict[str, dict[str, float]] = {}
        for row in (self.Mean() & key).fetch(as_dict=True):
            means.setdefault(row["model"], {})[row["metric"]] = row["value"]
        confusions = []
        for row in (self.FlagConfusion() & key).fetch(as_dict=True, order_by="metric"):
            flagged_by_continuous = row["continuous_only"] + row["both"]
            confusions.append(
                {
                    "metric": row["metric"],
                    "threshold": row["threshold"],
                    "n": row["n"],
                    "both": row["both"],
                    "a_only": row["continuous_only"],
                    "b_only": row["contfrag_only"],
                    "neither": row["neither"],
                    "rescue_rate": row["continuous_only"] / flagged_by_continuous
                    if flagged_by_continuous
                    else None,
                }
            )
        return {
            "n_units": int(self.fetch1("n_units")),
            "diagnostic_means": means,
            "flag_confusions": confusions,
        }


# --- Decode setup (existing Spyglass tables) -------------------------------------


def position_group_entry(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME, epoch_name: str = FIGURE04_EPOCH_NAME
) -> dict[str, Any]:
    """Return the ``PositionGroup`` to create (read-only).

    The group holds the v0 ``IntervalPositionInfo`` (``default_decoding``) entry of
    the epoch's position interval, with head position as the decoded variables and
    no upsampling (the entry is already at 500 Hz).
    """
    from spyglass.position import PositionOutput

    pos_name = get_position_interval_name(nwb_file_name, epoch_name)
    merge_id = PositionOutput.merge_get_part(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": pos_name,
            "position_info_param_name": POSITION_INFO_PARAM_NAME,
        }
    ).fetch1("merge_id")
    return {
        "nwb_file_name": nwb_file_name,
        "group_name": GROUP_NAME,
        "keys": [{"pos_merge_id": merge_id}],
        "position_variables": ["head_position_x", "head_position_y"],
        "upsample_rate": np.nan,
    }


def create_position_group(entry: Mapping[str, Any]) -> None:
    """Create the ``PositionGroup`` (**writes to the database**)."""
    from spyglass.decoding.v1.core import PositionGroup

    PositionGroup().create_group(**entry)


def spike_sorting_output_entries(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME,
) -> list[dict[str, Any]]:
    """Return the v0 HPC sort's ``CuratedSpikeSorting`` keys (read-only).

    ``SortedSpikesGroup`` reads units through the ``SpikeSortingOutput`` merge
    table, and this sort is not registered there yet.

    Raises
    ------
    ValueError
        If the restriction matches more than one sort for a sort group.
    """
    from spyglass.spikesorting.v0 import CuratedSpikeSorting

    restriction = {"nwb_file_name": nwb_file_name, **HPC_SORTING_RESTRICTION}
    keys: list[dict[str, Any]] = list(
        (CuratedSpikeSorting & restriction).fetch("KEY", order_by="sort_group_id")
    )
    sort_group_ids = [key["sort_group_id"] for key in keys]
    if len(set(sort_group_ids)) != len(sort_group_ids):
        raise ValueError(f"{restriction} matches more than one sort for some sort groups")
    return keys


def register_spike_sorting_output(entries: list[dict[str, Any]]) -> None:
    """Register the sort in ``SpikeSortingOutput`` (**writes to the database**).

    Entries already registered are skipped (the merge ID is a hash of the key).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    SpikeSortingOutput().insert(entries, part_name="CuratedSpikeSorting")


def sorted_spikes_group_entry(nwb_file_name: str = FIGURE04_NWB_FILE_NAME) -> dict[str, Any]:
    """Return the ``SortedSpikesGroup`` to create (read-only): the HPC sort's groups.

    Raises
    ------
    ValueError
        If the sort does not resolve to one ``SpikeSortingOutput`` entry per sort
        group.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v0 import CuratedSpikeSorting

    restriction = {"nwb_file_name": nwb_file_name, **HPC_SORTING_RESTRICTION}
    n_sort_groups = len(CuratedSpikeSorting & restriction)
    merge_ids = (SpikeSortingOutput.CuratedSpikeSorting & restriction).fetch("merge_id")
    if len(merge_ids) != n_sort_groups:
        raise ValueError(
            f"{len(merge_ids)} SpikeSortingOutput entries for {n_sort_groups} sort groups; "
            "register the sort first (register_spike_sorting_output)"
        )
    return {
        "group_name": GROUP_NAME,
        "nwb_file_name": nwb_file_name,
        "unit_filter_params_name": UNIT_FILTER_PARAMS_NAME,
        "keys": [{"spikesorting_merge_id": merge_id} for merge_id in merge_ids],
    }


def create_sorted_spikes_group(entry: Mapping[str, Any]) -> None:
    """Create the ``SortedSpikesGroup`` (**writes to the database**)."""
    from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup

    SortedSpikesGroup().create_group(**entry)


def decoding_parameter_entries(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME, epoch_name: str = FIGURE04_EPOCH_NAME
) -> list[dict[str, Any]]:
    """Return the two ``DecodingParameters`` to insert (read-only).

    The models are built by the figure pipeline's ``build_decoder_models`` on the
    epoch's track graph, so they carry exactly the Figure-4 settings.
    """
    config = Figure4Config()
    track_graph, edge_order, edge_spacing = get_track_graph(
        nwb_file_name, get_position_interval_name(nwb_file_name, epoch_name)
    )
    environment = create_decoder_environment(
        track_graph, list(edge_order), edge_spacing, config.decoder.position_bin_size_cm
    )
    continuous, contfrag = build_decoder_models(environment, config.decoder, config.execution)
    return [
        {
            "decoding_param_name": DECODING_PARAM_NAMES[name],
            "decoding_params": model,
            "decoding_kwargs": {"return_outputs": list(DECODE_OUTPUTS)},
        }
        for name, model in (("continuous", continuous), ("contfrag", contfrag))
    ]


def insert_decoding_parameters(entries: list[dict[str, Any]]) -> None:
    """Insert the ``DecodingParameters`` (**writes to the database**; refuses duplicates)."""
    from spyglass.decoding.v1.core import DecodingParameters

    DecodingParameters().insert(entries)


def decoding_selection_entries(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME, epoch_name: str = FIGURE04_EPOCH_NAME
) -> list[dict[str, Any]]:
    """Return the two ``SortedSpikesDecodingSelection`` entries (read-only).

    Encoding and decoding use the whole epoch interval, and parameter estimation
    is off, as in the figure pipeline.
    """
    interval_name = f"{epoch_name} noPrePostTrialTimes"
    return [
        {
            "nwb_file_name": nwb_file_name,
            "sorted_spikes_group_name": GROUP_NAME,
            "unit_filter_params_name": UNIT_FILTER_PARAMS_NAME,
            "position_group_name": GROUP_NAME,
            "decoding_param_name": param_name,
            "encoding_interval": interval_name,
            "decoding_interval": interval_name,
            "estimate_decoding_params": 0,
        }
        for param_name in DECODING_PARAM_NAMES.values()
    ]


def insert_decoding_selections(entries: list[dict[str, Any]]) -> None:
    """Insert the decoding selections (**writes to the database**)."""
    from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingSelection

    SortedSpikesDecodingSelection().insert(entries)


def populate_decoding(entries: list[dict[str, Any]]) -> None:
    """Run ``SortedSpikesDecodingV1`` for the selections (**writes to the database**).

    Writes the decodes and fitted models to the analysis directory and registers
    them in ``DecodingOutput``. Needs the analysis NWB store (a lab server).
    """
    from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1

    SortedSpikesDecodingV1.populate(entries)


# --- Diagnostics (custom schema) -------------------------------------------------


def diagnostics_selection_entry(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME, epoch_name: str = FIGURE04_EPOCH_NAME
) -> dict[str, Any]:
    """Return the :class:`Figure4DiagnosticsSelection` entry (read-only)."""
    selections = dict(
        zip(
            DECODING_PARAM_NAMES,
            decoding_selection_entries(nwb_file_name, epoch_name),
            strict=True,
        )
    )
    merge_ids = {
        name: DecodingOutput.merge_get_part(
            {k: v for k, v in selection.items() if k != "estimate_decoding_params"}
        ).fetch1("merge_id")
        for name, selection in selections.items()
    }
    return {
        "continuous_merge_id": merge_ids["continuous"],
        "contfrag_merge_id": merge_ids["contfrag"],
        "figure4_diagnostics_param_name": "figure04",
    }


def insert_diagnostics_selection(entry: Mapping[str, Any]) -> None:
    """Insert the diagnostics selection (**writes to the database**)."""
    Figure4DiagnosticsSelection().insert1(entry)


def populate_diagnostics(entry: Mapping[str, Any]) -> None:
    """Run :class:`Figure4Diagnostics` for the selection (**writes to the database**)."""
    Figure4Diagnostics.populate(dict(entry))
