"""Figure-4 manuscript scalars from the two decoders' per-spike diagnostics.

Whole-session event means, the two-decoder flag agreement, the two-decoder
KL-divergence comparison, the detail-window comparison, where the Continuous
decoder's flags fall by the animal's speed, and how many HPD-overlap flags have
nonzero overlap, plus the command-line formatting. Kept apart from
:mod:`figure04_workflow` so that editing a summary or its printed text does not
change the decode cache's source fingerprint and refit the models.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import (
    METRIC_FLAG_DIRECTIONS,
    FlagDirection,
    SpikeEventDiagnostics,
    flag_mask,
)
from statespacecheck_paper.figure04_diagnostics import FlagConfusion, compute_flag_confusion
from statespacecheck_paper.figure04_models import CONTINUOUS, CONTINUOUS_FRAGMENTED
from statespacecheck_paper.figure04_place_fields import DECODER_PREDICTIVE_VAR
from statespacecheck_paper.figure04_protocol import (
    FIGURE04_FRAGMENTED_PROBABILITY_THRESHOLD,
    FIGURE04_IMMOBILE_SPEED_CM_PER_S,
)
from statespacecheck_paper.figure04_workflow import Figure4RenderData

# The Continuous-Fragmented decoder's jump mode, as non_local_detector names it.
FRAGMENTED_STATE = "Fragmented"

# Two models' per-spike p-values closer than this are ties. Near-equal p-values
# differ by round-off that varies with the CPU: in the detail window 19 events
# differ by less than this, 9 of them upward, and none lie between 1e-9 and 1e-7,
# so a strict comparison makes the count depend on the machine.
FIGURE04_PVALUE_TIE_TOLERANCE = 1e-9


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
    lower_continuous_median, lower_continuous_fragmented_median : float
        Median KL divergence of those events under each decoder.
    higher_fraction : float
        Fraction of spike events with higher KL divergence under Continuous-Fragmented.
    higher_continuous_median, higher_continuous_fragmented_median : float
        Median KL divergence of those events under each decoder.
    higher_continuous_fragmented_max : float
        Largest Continuous-Fragmented KL divergence among those events.
    """

    lower_fraction: float
    lower_continuous_median: float
    lower_continuous_fragmented_median: float
    higher_fraction: float
    higher_continuous_median: float
    higher_continuous_fragmented_median: float
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
    return Figure4KlDivergenceComparison(
        lower_fraction=float(lower.mean()),
        lower_continuous_median=float(np.median(continuous[lower])),
        lower_continuous_fragmented_median=float(np.median(fragmented[lower])),
        higher_fraction=float(higher.mean()),
        higher_continuous_median=float(np.median(continuous[higher])),
        higher_continuous_fragmented_median=float(np.median(fragmented[higher])),
        higher_continuous_fragmented_max=float(np.max(fragmented[higher])),
    )


@dataclasses.dataclass(frozen=True)
class Figure4DetailWindowComparison:
    """Two-decoder comparison of the spike events in the Figure-4 detail window.

    Describes the detail panels of Figure 4a,b: the animal's speed in the window,
    each thresholded metric's flag agreement between the two decoders, how many
    spikes the Continuous-Fragmented decoder makes less unexpected under the
    predictive check, the window's median KL divergence under each decoder, and
    the share of the Continuous-Fragmented prediction assigned to its Fragmented mode.

    Attributes
    ----------
    n_events : int
        Spike events in the window.
    hpd_overlap, predictive_pvalue : FlagConfusion
        Flag agreement on those events, with the Continuous decoder as the reference.
    predictive_pvalue_higher : int
        Events with a higher predictive p-value (lower ``-log(p)``) under
        Continuous-Fragmented than under Continuous, by more than
        :data:`FIGURE04_PVALUE_TIE_TOLERANCE`.
    continuous_kl_median, continuous_fragmented_kl_median : float
        Median KL divergence of the window's events under each decoder.
    median_speed_cm_per_s : float
        The animal's median head speed over the window's time samples.
    fragmented_share_median, fragmented_share_peak : float
        Median and largest share of the Continuous-Fragmented one-step prediction
        assigned to the Fragmented mode over the window's time samples.
    fragmented_probability_threshold : float
        Threshold on that share.
    fragmented_share_above_threshold_samples : int
        Time samples in the window at which that share exceeds the threshold.
    """

    n_events: int
    hpd_overlap: FlagConfusion
    predictive_pvalue: FlagConfusion
    predictive_pvalue_higher: int
    continuous_kl_median: float
    continuous_fragmented_kl_median: float
    median_speed_cm_per_s: float
    fragmented_share_median: float
    fragmented_share_peak: float
    fragmented_probability_threshold: float
    fragmented_share_above_threshold_samples: int


def compute_detail_window_comparison(
    continuous_diagnostics: SpikeEventDiagnostics,
    continuous_fragmented_diagnostics: SpikeEventDiagnostics,
    time_slice: slice,
    *,
    thresholds: Mapping[str, float],
    metric_directions: Mapping[str, FlagDirection],
    speed_cm_per_s: NDArray[np.floating],
    fragmented_share: NDArray[np.floating],
    fragmented_probability_threshold: float,
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
        :func:`summarize_figure04_diagnostics`; ``"hpd_overlap"`` and
        ``"predictive_pvalue"`` are used.
    speed_cm_per_s : np.ndarray, shape (n_time,)
        The animal's head speed at every decode time sample.
    fragmented_share : np.ndarray, shape (n_window_samples,)
        Share of the Continuous-Fragmented one-step prediction assigned to the
        Fragmented mode at each time sample of the window, from
        :func:`predictive_state_share`.
    fragmented_probability_threshold : float
        Share above which the window's time is counted.

    Returns
    -------
    Figure4DetailWindowComparison
        The window's event count, flag agreement, p-value and KL-divergence
        comparisons, median speed, and Fragmented-mode share.

    Raises
    ------
    ValueError
        If the window contains no spike events, ``fragmented_share`` does not
        cover the window, or the two decoders' events differ (from
        :func:`compute_flag_confusion`).
    """
    time_ind = np.asarray(continuous_diagnostics.event_time_ind)
    in_window = (time_ind >= time_slice.start) & (time_ind < time_slice.stop)
    if not in_window.any():
        raise ValueError(
            f"The detail window [{time_slice.start}, {time_slice.stop}) contains no spike events"
        )
    share = np.asarray(fragmented_share, dtype=np.float64)
    if share.shape != (time_slice.stop - time_slice.start,):
        raise ValueError(
            f"fragmented_share must have one value per window sample "
            f"({time_slice.stop - time_slice.start}); got shape {share.shape}"
        )
    continuous = continuous_diagnostics.select_events(in_window)
    fragmented = continuous_fragmented_diagnostics.select_events(in_window)
    hpd_overlap, predictive_pvalue = (
        compute_flag_confusion(
            continuous,
            fragmented,
            metric,
            thresholds[metric],
            worse_when=metric_directions[metric],
        )
        for metric in ("hpd_overlap", "predictive_pvalue")
    )
    return Figure4DetailWindowComparison(
        n_events=int(in_window.sum()),
        hpd_overlap=hpd_overlap,
        predictive_pvalue=predictive_pvalue,
        predictive_pvalue_higher=int(
            np.sum(
                fragmented.event_predictive_pvalue - continuous.event_predictive_pvalue
                > FIGURE04_PVALUE_TIE_TOLERANCE
            )
        ),
        continuous_kl_median=float(np.median(continuous.event_kl_divergence)),
        continuous_fragmented_kl_median=float(np.median(fragmented.event_kl_divergence)),
        median_speed_cm_per_s=float(np.median(np.asarray(speed_cm_per_s)[time_slice])),
        fragmented_share_median=float(np.median(share)),
        fragmented_share_peak=float(np.max(share)),
        fragmented_probability_threshold=float(fragmented_probability_threshold),
        fragmented_share_above_threshold_samples=int(
            np.sum(share > fragmented_probability_threshold)
        ),
    )


def predictive_state_share(
    results: xr.Dataset, state: str, time_slice: slice
) -> NDArray[np.float64]:
    """Share of a decoder's one-step prediction assigned to one dynamics mode.

    Parameters
    ----------
    results : xr.Dataset
        Decoding results from ``model.predict()``, with ``predictive_posterior``
        over ``(time, state_bins)``.
    state : str
        Dynamics-mode name, e.g. ``"Fragmented"``.
    time_slice : slice
        Time samples to return.

    Returns
    -------
    np.ndarray, shape (n_slice_samples,)
        Predictive probability summed over the mode's track-interior position bins.

    Raises
    ------
    ValueError
        If the decoder has no mode named ``state``.
    """
    predictive = results[DECODER_PREDICTIVE_VAR].isel(time=time_slice).dropna("state_bins")
    in_state = predictive["state"].to_numpy() == state
    if not in_state.any():
        raise ValueError(f"The decoder has no {state!r} mode")
    return np.asarray(predictive.to_numpy()[:, in_state].sum(axis=1), dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class Figure4ImmobilityComparison:
    """Where in the session the Continuous decoder's flags fall, by the animal's speed.

    Attributes
    ----------
    speed_threshold_cm_per_s : float
        The animal counts as immobile below this head speed.
    spike_fraction : float
        Fraction of all spike events that occur while the animal is immobile.
    flagged_fractions : dict of str to float
        Per thresholded metric, the fraction of the spikes it flags under the
        Continuous decoder that occur while the animal is immobile.
    """

    speed_threshold_cm_per_s: float
    spike_fraction: float
    flagged_fractions: dict[str, float]


def compute_immobility_comparison(
    diagnostics: SpikeEventDiagnostics,
    speed_cm_per_s: NDArray[np.floating],
    speed_threshold_cm_per_s: float,
    *,
    thresholds: Mapping[str, float],
    metric_directions: Mapping[str, FlagDirection],
) -> Figure4ImmobilityComparison:
    """Compare where one decoder's flags fall with where its spikes fall, by speed.

    Parameters
    ----------
    diagnostics : SpikeEventDiagnostics
        Per-spike diagnostics of one decoder; ``event_time_ind`` indexes the
        decode time samples.
    speed_cm_per_s : np.ndarray, shape (n_time,)
        The animal's head speed at every decode time sample.
    speed_threshold_cm_per_s : float
        Speeds below this count as immobile.
    thresholds, metric_directions : Mapping
        Flag threshold and direction of worse fit per metric.

    Returns
    -------
    Figure4ImmobilityComparison
        The immobile fraction of all spikes and of each metric's flagged spikes.

    Raises
    ------
    ValueError
        If there are no spike events, the speed contains NaN at a spike, or a
        metric flags no spikes.
    """
    event_speed = np.asarray(speed_cm_per_s, dtype=np.float64)[diagnostics.event_time_ind]
    if event_speed.size == 0:
        raise ValueError("Cannot compare immobility: no spike events are present")
    if np.isnan(event_speed).any():
        raise ValueError("The animal's speed is NaN at some spike events")
    immobile = event_speed < speed_threshold_cm_per_s
    flagged_fractions = {}
    for metric, worse_when in metric_directions.items():
        flagged = flag_mask(getattr(diagnostics, f"event_{metric}"), thresholds[metric], worse_when)
        if not flagged.any():
            raise ValueError(f"{metric} flags no spike events")
        flagged_fractions[metric] = float(immobile[flagged].mean())
    return Figure4ImmobilityComparison(
        speed_threshold_cm_per_s=float(speed_threshold_cm_per_s),
        spike_fraction=float(immobile.mean()),
        flagged_fractions=flagged_fractions,
    )


def count_flagged_with_nonzero_overlap(diagnostics: SpikeEventDiagnostics, threshold: float) -> int:
    """Count the spikes flagged by HPD overlap whose overlap is not exactly zero.

    Parameters
    ----------
    diagnostics : SpikeEventDiagnostics
        Per-spike diagnostics of one decoder.
    threshold : float
        HPD-overlap flag threshold; values at or below it are flagged.

    Returns
    -------
    int
        Flagged spikes whose prediction and likelihood HPD regions share at
        least one position bin.
    """
    overlap = np.asarray(diagnostics.event_hpd_overlap, dtype=np.float64)
    flagged = flag_mask(overlap, threshold, METRIC_FLAG_DIRECTIONS["hpd_overlap"])
    return int(np.sum(flagged & (overlap > 0.0)))


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
    immobility_comparison : Figure4ImmobilityComparison
        Where the Continuous decoder's flags fall by the animal's speed.
    hpd_flagged_nonzero_overlap : int
        Spikes HPD overlap flags under the Continuous decoder whose overlap is
        not exactly zero.
    """

    summary: Figure4Summary
    n_units_with_spikes: int
    kl_divergence_comparison: Figure4KlDivergenceComparison
    detail_window_comparison: Figure4DetailWindowComparison
    immobility_comparison: Figure4ImmobilityComparison
    hpd_flagged_nonzero_overlap: int


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
        The two decoders' per-spike diagnostics, spike counts, and decodes, and
        the recording's head speed (``position_info["head_speed"]``, cm/s).
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
    speed = render_data.recording.position_info["head_speed"].to_numpy(dtype=np.float64)
    if speed.shape != (render_data.time.size,):
        raise ValueError(
            f"head_speed must have one value per decode time sample ({render_data.time.size}); "
            f"got shape {speed.shape}"
        )
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
            speed_cm_per_s=speed,
            fragmented_share=predictive_state_share(
                analysis.continuous_fragmented_results, FRAGMENTED_STATE, detail_slice
            ),
            fragmented_probability_threshold=FIGURE04_FRAGMENTED_PROBABILITY_THRESHOLD,
        ),
        immobility_comparison=compute_immobility_comparison(
            continuous,
            speed,
            FIGURE04_IMMOBILE_SPEED_CM_PER_S,
            thresholds=thresholds,
            metric_directions=metric_directions,
        ),
        hpd_flagged_nonzero_overlap=count_flagged_with_nonzero_overlap(
            continuous, thresholds["hpd_overlap"]
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

    The computation behind :func:`compute_figure04_reported_statistics`, usable without
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
