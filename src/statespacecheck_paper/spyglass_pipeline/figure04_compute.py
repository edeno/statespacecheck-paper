"""What Figure 4's Spyglass tables compute, runnable without a database.

:mod:`~statespacecheck_paper.spyglass_pipeline.figure04_schema` stores Figure 4's
decoding and per-spike diagnostics in Spyglass tables; importing it connects to the
lab database. The computations those tables run live here instead, so they can be
tested and reused without a connection: the per-spike diagnostics and summary from
two stored decodes (reusing the paper's Figure-4 code, so both routes compute the
same numbers), the part-table rows that store a summary, and the reported
statistics rebuilt from those rows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from statespacecheck_paper.figure04_models import CONTINUOUS, CONTINUOUS_FRAGMENTED

if TYPE_CHECKING:
    from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
    from statespacecheck_paper.figure04_summary import Figure4Summary

from statespacecheck_paper.spyglass_pipeline.figure04_input import filter_spike_times


def figure04_diagnostics_from_decodes(
    continuous_model: Any,
    continuous_fragmented_model: Any,
    continuous_results: xr.Dataset,
    continuous_fragmented_results: xr.Dataset,
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
    continuous_model, continuous_fragmented_model : non_local_detector model
        Fitted Continuous and Continuous-Fragmented decoders.
    continuous_results, continuous_fragmented_results : xr.Dataset
        Their decodes over the same time bins; each must hold the predictive
        distribution ``predictive_posterior`` (request it with ``return_outputs``).
    spike_times : sequence of np.ndarray, shape (n_spikes,)
        Per-unit spike times (seconds), in the order the models were fitted with.
        Checked against each unit's fitted mean rate, which assumes the models
        were trained on every decoded time bin (as in Figure 4).
    coverage : float
        HPD coverage for the HPD-overlap diagnostic.
    thresholds : Mapping of str to float, optional
        Flag threshold per metric. Default: the Figure-4 thresholds.

    Returns
    -------
    continuous, continuous_fragmented : SpikeEventDiagnostics
        Per-spike diagnostics of each decoder, on the same spikes.
    summary : Figure4Summary
        Whole-session event means and two-decoder flag agreement.

    Raises
    ------
    ValueError
        If a decode lacks ``predictive_posterior``, the decodes cover different
        time bins, the spike trains do not match the fitted units (count or
        order), or the two decoders' place fields differ.

    Notes
    -----
    The Figure-4 modules this reuses, and through them ``statespacecheck`` and
    ``non_local_detector``, are imported inside this function, so this module
    stays importable without those packages.
    """
    from statespacecheck_paper.figure04_decoder import get_spike_counts
    from statespacecheck_paper.figure04_diagnostics import compute_results_diagnostics
    from statespacecheck_paper.figure04_generation import (
        FIGURE4_DIAGNOSTIC_THRESHOLDS,
        FIGURE4_METRIC_DIRECTIONS,
    )
    from statespacecheck_paper.figure04_place_fields import (
        DECODER_PREDICTIVE_VAR,
        extract_agreed_place_fields,
    )
    from statespacecheck_paper.figure04_summary import summarize_figure04_diagnostics

    time = continuous_results["time"].to_numpy()
    if not np.array_equal(time, continuous_fragmented_results["time"].to_numpy()):
        raise ValueError("The two decodes cover different time bins")
    for name, results in (
        (CONTINUOUS.label, continuous_results),
        (CONTINUOUS_FRAGMENTED.label, continuous_fragmented_results),
    ):
        if DECODER_PREDICTIVE_VAR not in results:
            raise ValueError(
                f"The {name} decode has no {DECODER_PREDICTIVE_VAR}; decode with "
                f"return_outputs including {DECODER_PREDICTIVE_VAR!r}"
            )
    spikes = filter_spike_times(spike_times, time)
    expected_rates = np.array([len(st) for st in spikes]) / len(time)
    for model in (continuous_model, continuous_fragmented_model):
        (encoding_model,) = model.encoding_model_.values()
        fitted_rates = np.asarray(encoding_model["mean_rates"])
        if fitted_rates.shape != expected_rates.shape or not np.allclose(
            fitted_rates, expected_rates, rtol=1e-6, atol=0.0
        ):
            raise ValueError("The spike trains do not match the fitted units (count or order)")
    place_fields, _ = extract_agreed_place_fields(continuous_model, continuous_fragmented_model)
    spike_counts = get_spike_counts(spikes, time)
    continuous, continuous_fragmented = (
        compute_results_diagnostics(
            results, place_fields, spike_counts, time, spikes, coverage=coverage
        )
        for results in (continuous_results, continuous_fragmented_results)
    )
    summary = summarize_figure04_diagnostics(
        continuous,
        continuous_fragmented,
        n_units=int(spike_counts.shape[1]),
        thresholds=FIGURE4_DIAGNOSTIC_THRESHOLDS if thresholds is None else thresholds,
        metric_directions=FIGURE4_METRIC_DIRECTIONS,
    )
    return continuous, continuous_fragmented, summary


def figure04_summary_rows(
    summary: Figure4Summary,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the ``Figure4Diagnostics`` part-table rows that store a summary.

    The inverse of :func:`figure04_reported_statistics_from_rows`. Rows carry no
    table key; the caller adds it.

    Parameters
    ----------
    summary : Figure4Summary
        Whole-session event means and two-decoder flag agreement.

    Returns
    -------
    mean_rows : list of dict
        ``Figure4Diagnostics.Mean`` rows: ``model``, ``metric``, ``value``.
    confusion_rows : list of dict
        ``Figure4Diagnostics.FlagConfusion`` rows: ``metric``, ``threshold``,
        ``n``, ``both``, ``rescued``, ``newly_flagged``, ``neither``.
    """
    mean_rows = [
        {"model": model, "metric": metric, "value": value}
        for model, means in (
            (CONTINUOUS.id, summary.continuous),
            (CONTINUOUS_FRAGMENTED.id, summary.continuous_fragmented),
        )
        for metric, value in dataclasses.asdict(means).items()
    ]
    confusion_rows = [
        {
            "metric": confusion.metric,
            "threshold": confusion.threshold,
            "n": confusion.n,
            "both": confusion.both,
            "rescued": confusion.rescued,
            "newly_flagged": confusion.newly_flagged,
            "neither": confusion.neither,
        }
        for confusion in summary.flag_confusions
    ]
    return mean_rows, confusion_rows


def figure04_reported_statistics_from_rows(
    mean_rows: Sequence[Mapping[str, Any]],
    confusion_rows: Sequence[Mapping[str, Any]],
    *,
    n_units: int,
) -> dict[str, Any]:
    """Rebuild the reported statistics of ``figure04_summary.json`` from stored rows.

    The rows are turned back into a :class:`~statespacecheck_paper.figure04_summary.Figure4Summary`
    and reported by the figure pipeline's
    :func:`~statespacecheck_paper.figure04_generation.figure04_reported_statistics`,
    so the stored and committed summaries are compared in the same form. Flag
    confusions follow the Figure-4 metric order whatever order the rows come in.

    Parameters
    ----------
    mean_rows : sequence of mapping
        ``Figure4Diagnostics.Mean`` rows (``model``, ``metric``, ``value``).
    confusion_rows : sequence of mapping
        ``Figure4Diagnostics.FlagConfusion`` rows (see :func:`figure04_summary_rows`).
    n_units : int
        Number of units decoded.

    Returns
    -------
    dict
        ``n_units``, ``diagnostic_means`` (per decoder and metric), and
        ``flag_confusions`` (with ``rescue_rate``), as in ``figure04_summary.json``.

    Raises
    ------
    ValueError
        If the flag-confusion rows do not name each Figure-4 thresholded metric
        exactly once.
    """
    from statespacecheck_paper.figure04_diagnostics import FlagConfusion
    from statespacecheck_paper.figure04_generation import (
        FIGURE4_METRIC_DIRECTIONS,
        figure04_reported_statistics,
    )
    from statespacecheck_paper.figure04_summary import Figure4DiagnosticMeans, Figure4Summary

    means: dict[str, dict[str, float]] = {}
    for row in mean_rows:
        means.setdefault(row["model"], {})[row["metric"]] = float(row["value"])
    metrics = [row["metric"] for row in confusion_rows]
    if sorted(metrics) != sorted(FIGURE4_METRIC_DIRECTIONS):
        raise ValueError(
            f"Expected one flag-confusion row per metric in {list(FIGURE4_METRIC_DIRECTIONS)}; "
            f"got {metrics}"
        )
    confusions = {
        row["metric"]: FlagConfusion(
            metric=row["metric"],
            threshold=float(row["threshold"]),
            n=int(row["n"]),
            both=int(row["both"]),
            rescued=int(row["rescued"]),
            newly_flagged=int(row["newly_flagged"]),
            neither=int(row["neither"]),
        )
        for row in confusion_rows
    }
    summary = Figure4Summary(
        continuous=Figure4DiagnosticMeans(**means[CONTINUOUS.id]),
        continuous_fragmented=Figure4DiagnosticMeans(**means[CONTINUOUS_FRAGMENTED.id]),
        flag_confusions=tuple(confusions[metric] for metric in FIGURE4_METRIC_DIRECTIONS),
        n_units=n_units,
    )
    return {"n_units": summary.n_units, **figure04_reported_statistics(summary)}
