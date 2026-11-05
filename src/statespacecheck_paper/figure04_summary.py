"""Figure-4 manuscript scalars from the two decoders' per-spike diagnostics.

Whole-session event means, the two-decoder flag agreement, the two-decoder
KL-divergence comparison, and the detail-window comparison, plus their
command-line formatting. Kept apart from
:mod:`figure04_workflow` so that editing a summary or its printed text does not
change the decode cache's source fingerprint and refit the models.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from statespacecheck_paper.diagnostics import FlagDirection, SpikeEventDiagnostics
from statespacecheck_paper.figure04_diagnostics import FlagConfusion, compute_flag_confusion
from statespacecheck_paper.figure04_models import CONTINUOUS, CONTINUOUS_FRAGMENTED
from statespacecheck_paper.figure04_workflow import Figure4RenderData


def compute_mean_spike_event_diagnostic(diagnostics: SpikeEventDiagnostics, metric: str) -> float:
    """Return the per-spike mean for a diagnostic metric."""
    event_key = f"event_{metric}"
    if not hasattr(diagnostics, event_key):
        raise KeyError(f"Missing per-spike diagnostic array: {event_key}")
    values = np.asarray(getattr(diagnostics, event_key), dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"Cannot compute {event_key} mean: no spike events are present")
    return float(np.mean(values))


def count_units_with_spikes(diagnostics: SpikeEventDiagnostics) -> int:
    """Return the number of units with at least one spike event."""
    return int(np.unique(diagnostics.event_cell_ind).size)


@dataclasses.dataclass(frozen=True)
class Figure4DiagnosticMeans:
    """Whole-session mean of each per-spike diagnostic for one decoder."""

    hpd_overlap: float
    kl_divergence: float
    predictive_pvalue: float


@dataclasses.dataclass(frozen=True)
class Figure4Summary:
    """Manuscript-facing Figure-4 scalars, independent of output formatting.

    ``n_units`` is the number of sorted units the decoders were fit on, including
    units with no spikes in this epoch (:func:`count_units_with_spikes` counts
    the others). It is reported in the figure caption, so it is carried here
    rather than recounted at render time.
    """

    continuous: Figure4DiagnosticMeans
    continuous_fragmented: Figure4DiagnosticMeans
    flag_confusions: tuple[FlagConfusion, ...]
    n_units: int


@dataclasses.dataclass(frozen=True)
class Figure4KlDivergenceComparison:
    """Per-spike KL divergence under the two decoders, split at the identity line.

    Describes the two sides of the identity line in the KL panel of Figure 4c:
    spikes whose KL divergence is lower under the Continuous-Fragmented decoder
    than under the Continuous decoder, and spikes whose KL divergence is higher.
    Spikes with equal values are on neither side. Fractions are of all spike events.

    Attributes
    ----------
    lower_fraction : float
        Fraction of spike events with lower KL divergence under Continuous-Fragmented.
    lower_continuous_quartiles : tuple of float
        25th and 75th percentiles of those events' Continuous-model KL divergence.
    higher_fraction : float
        Fraction of spike events with higher KL divergence under Continuous-Fragmented.
    higher_continuous_median : float
        Median Continuous-model KL divergence of those events.
    higher_continuous_fragmented_max : float
        Largest Continuous-Fragmented KL divergence among those events.
    """

    lower_fraction: float
    lower_continuous_quartiles: tuple[float, float]
    higher_fraction: float
    higher_continuous_median: float
    higher_continuous_fragmented_max: float


def compute_kl_divergence_comparison(
    continuous_diagnostics: SpikeEventDiagnostics,
    continuous_fragmented_diagnostics: SpikeEventDiagnostics,
) -> Figure4KlDivergenceComparison:
    """Compare per-spike KL divergence between the two decoders.

    Parameters
    ----------
    continuous_diagnostics, continuous_fragmented_diagnostics : SpikeEventDiagnostics
        Per-spike diagnostics of the Continuous and Continuous-Fragmented decoders,
        on the same spikes; ``event_kl_divergence`` has shape ``(n_events,)``.

    Returns
    -------
    Figure4KlDivergenceComparison
        Both sides of the identity line in the Figure-4c KL panel.

    Raises
    ------
    ValueError
        If the two decoders' events differ in number, or either side of the
        identity line is empty.
    """
    continuous = np.asarray(continuous_diagnostics.event_kl_divergence, dtype=np.float64)
    fragmented = np.asarray(continuous_fragmented_diagnostics.event_kl_divergence, dtype=np.float64)
    if continuous.shape != fragmented.shape:
        raise ValueError(
            "The two decoders' KL divergences must cover the same spike events; "
            f"got shapes {continuous.shape} and {fragmented.shape}"
        )
    lower = fragmented < continuous
    higher = fragmented > continuous
    if not lower.any() or not higher.any():
        raise ValueError(
            "Both sides of the KL identity line need spike events; "
            f"got {int(lower.sum())} lower and {int(higher.sum())} higher"
        )
    q25, q75 = np.percentile(continuous[lower], [25.0, 75.0])
    return Figure4KlDivergenceComparison(
        lower_fraction=float(lower.mean()),
        lower_continuous_quartiles=(float(q25), float(q75)),
        higher_fraction=float(higher.mean()),
        higher_continuous_median=float(np.median(continuous[higher])),
        higher_continuous_fragmented_max=float(np.max(fragmented[higher])),
    )


@dataclasses.dataclass(frozen=True)
class Figure4DetailWindowComparison:
    """Two-decoder comparison of the spike events in the Figure-4 detail window.

    Describes the detail panels of Figure 4a,b: how many of the window's spikes
    that HPD overlap flags under the Continuous decoder are no longer flagged
    under the Continuous-Fragmented decoder, and how many spikes the
    Continuous-Fragmented decoder makes less unexpected under the predictive check.

    Attributes
    ----------
    n_events : int
        Spike events in the window.
    hpd_overlap : FlagConfusion
        HPD-overlap flag agreement on those events, with the Continuous decoder
        as the reference.
    predictive_pvalue_higher : int
        Events with a higher predictive p-value (lower ``-log(p)``) under
        Continuous-Fragmented than under Continuous.
    """

    n_events: int
    hpd_overlap: FlagConfusion
    predictive_pvalue_higher: int


def compute_detail_window_comparison(
    continuous_diagnostics: SpikeEventDiagnostics,
    continuous_fragmented_diagnostics: SpikeEventDiagnostics,
    time_slice: slice,
    *,
    thresholds: Mapping[str, float],
    metric_directions: Mapping[str, FlagDirection],
) -> Figure4DetailWindowComparison:
    """Compare the two decoders' per-spike diagnostics within the detail window.

    Parameters
    ----------
    continuous_diagnostics, continuous_fragmented_diagnostics : SpikeEventDiagnostics
        Per-spike diagnostics of the Continuous and Continuous-Fragmented decoders,
        on the same spikes; ``event_time_ind`` indexes the decode time samples.
    time_slice : slice
        Half-open range of decode time samples, from
        :meth:`Figure4DetailWindow.to_slice`.
    thresholds, metric_directions : Mapping
        Flag threshold and direction of worse fit per metric, as for
        :func:`summarize_figure04_diagnostics`; ``"hpd_overlap"`` is used.

    Returns
    -------
    Figure4DetailWindowComparison
        The window's event count, HPD-overlap flag agreement, and count of
        events with a higher predictive p-value under Continuous-Fragmented.

    Raises
    ------
    ValueError
        If the window contains no spike events, or the two decoders' events
        differ (from :func:`compute_flag_confusion`).
    """
    time_ind = np.asarray(continuous_diagnostics.event_time_ind)
    in_window = (time_ind >= time_slice.start) & (time_ind < time_slice.stop)
    if not in_window.any():
        raise ValueError(
            f"The detail window [{time_slice.start}, {time_slice.stop}) contains no spike events"
        )
    continuous = continuous_diagnostics.select_events(in_window)
    fragmented = continuous_fragmented_diagnostics.select_events(in_window)
    hpd_overlap = compute_flag_confusion(
        continuous,
        fragmented,
        "hpd_overlap",
        thresholds["hpd_overlap"],
        worse_when=metric_directions["hpd_overlap"],
    )
    return Figure4DetailWindowComparison(
        n_events=int(in_window.sum()),
        hpd_overlap=hpd_overlap,
        predictive_pvalue_higher=int(
            np.sum(fragmented.event_predictive_pvalue > continuous.event_predictive_pvalue)
        ),
    )


def _compute_diagnostic_means(
    diagnostics: SpikeEventDiagnostics,
) -> Figure4DiagnosticMeans:
    """Compute the three manuscript diagnostic means for one decoder."""
    return Figure4DiagnosticMeans(
        hpd_overlap=compute_mean_spike_event_diagnostic(diagnostics, "hpd_overlap"),
        kl_divergence=compute_mean_spike_event_diagnostic(diagnostics, "kl_divergence"),
        predictive_pvalue=compute_mean_spike_event_diagnostic(diagnostics, "predictive_pvalue"),
    )


@dataclasses.dataclass(frozen=True)
class Figure4ReportedStatistics:
    """Every statistic the Figure-4 summary reports about the two decodes.

    Attributes
    ----------
    summary : Figure4Summary
        The part the Spyglass rows can rebuild.
    n_units_with_spikes : int
        Units with at least one spike event.
    kl_divergence_comparison : Figure4KlDivergenceComparison
        The two sides of the identity line in the Figure-4c KL panel.
    detail_window_comparison : Figure4DetailWindowComparison
        The two decoders' spike events in the Figure-4a,b detail window.
    """

    summary: Figure4Summary
    n_units_with_spikes: int
    kl_divergence_comparison: Figure4KlDivergenceComparison
    detail_window_comparison: Figure4DetailWindowComparison


def compute_figure04_reported_statistics(
    render_data: Figure4RenderData,
    thresholds: Mapping[str, float],
    metric_directions: Mapping[str, FlagDirection],
    detail_slice: slice,
) -> Figure4ReportedStatistics:
    """Compute the Figure-4 statistics from the cached decode.

    These scalars can appear in the manuscript text, so their computation must
    stay identical to the cached decode; the render layer never alters them.

    Parameters
    ----------
    render_data : Figure4RenderData
        The two decoders' per-spike diagnostics and spike counts.
    thresholds, metric_directions : Mapping
        Flag threshold and direction of worse fit per metric, as for
        :func:`summarize_figure04_diagnostics`.
    detail_slice : slice
        Half-open range of decode time samples, from
        :meth:`Figure4DetailWindow.to_slice`.

    Returns
    -------
    Figure4ReportedStatistics
        The whole-session summary and the Figure-4 comparisons.
    """
    analysis = render_data.analysis_results
    continuous = analysis.continuous_diagnostics
    fragmented = analysis.continuous_fragmented_diagnostics
    return Figure4ReportedStatistics(
        summary=summarize_figure04_diagnostics(
            continuous,
            fragmented,
            n_units=int(analysis.spike_counts.shape[1]),
            thresholds=thresholds,
            metric_directions=metric_directions,
        ),
        n_units_with_spikes=count_units_with_spikes(continuous),
        kl_divergence_comparison=compute_kl_divergence_comparison(continuous, fragmented),
        detail_window_comparison=compute_detail_window_comparison(
            continuous,
            fragmented,
            detail_slice,
            thresholds=thresholds,
            metric_directions=metric_directions,
        ),
    )


def summarize_figure04_diagnostics(
    continuous_diagnostics: SpikeEventDiagnostics,
    continuous_fragmented_diagnostics: SpikeEventDiagnostics,
    *,
    n_units: int,
    thresholds: Mapping[str, float],
    metric_directions: Mapping[str, FlagDirection],
) -> Figure4Summary:
    """Compute the Figure-4 summary from the two decoders' per-spike diagnostics.

    The computation behind :func:`compute_figure04_summary`, usable without
    render data (e.g. on decodes stored by the Spyglass pipeline).

    Parameters
    ----------
    continuous_diagnostics, continuous_fragmented_diagnostics : SpikeEventDiagnostics
        Per-spike diagnostics of the Continuous and Continuous-Fragmented decoders,
        on the same spikes.
    n_units : int
        Number of units decoded.
    thresholds : Mapping of str to float
        Flag threshold per metric.
    metric_directions : Mapping of str to {"below", "above"}
        Which side of each threshold is flagged.

    Returns
    -------
    Figure4Summary
        Whole-session event means and two-decoder flag agreement.
    """
    missing_thresholds = set(metric_directions) - set(thresholds)
    if missing_thresholds:
        raise ValueError(
            f"metric_directions contains metrics without thresholds: {sorted(missing_thresholds)}"
        )

    flag_confusions = []
    for metric, worse_when in metric_directions.items():
        flag_confusions.append(
            compute_flag_confusion(
                continuous_diagnostics,
                continuous_fragmented_diagnostics,
                metric,
                thresholds[metric],
                worse_when=worse_when,
            )
        )

    return Figure4Summary(
        continuous=_compute_diagnostic_means(continuous_diagnostics),
        continuous_fragmented=_compute_diagnostic_means(continuous_fragmented_diagnostics),
        flag_confusions=tuple(flag_confusions),
        n_units=n_units,
    )


def format_figure04_summary(summary: Figure4Summary) -> str:
    """Format a computed Figure-4 summary for command-line output."""
    lines = [f"=== Diagnostic Summary ({summary.n_units} units, all time points) ==="]
    for model_name, means in (
        (CONTINUOUS.label, summary.continuous),
        (CONTINUOUS_FRAGMENTED.label, summary.continuous_fragmented),
    ):
        lines.extend(
            [
                "",
                f"{model_name}:",
                f"  hpd_overlap: {means.hpd_overlap:.4f}",
                f"  kl_divergence: {means.kl_divergence:.4f}",
                f"  predictive_pvalue: {means.predictive_pvalue:.4f}",
            ]
        )

    # "rescued": flagged by Continuous (the reference) but not by
    # Continuous-Fragmented (the comparison); "newly_flagged": the reverse. The
    # rescued share is rescued / Continuous flags.
    lines.extend(
        [
            "",
            f"=== Flag agreement: {CONTINUOUS.label} (reference) vs "
            f"{CONTINUOUS_FRAGMENTED.label} (comparison) ===",
        ]
    )
    for confusion in summary.flag_confusions:
        lines.append(
            f"  {confusion.metric}: n={confusion.n:,} both={confusion.both:,} "
            f"rescued={confusion.rescued:,} newly_flagged={confusion.newly_flagged:,} "
            f"neither={confusion.neither:,} rescued_percent={100 * confusion.rescued_fraction:.1f}%"
        )
    return "\n".join(lines)
