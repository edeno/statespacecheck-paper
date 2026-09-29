"""Figure-4 manuscript scalars from the two decoders' per-spike diagnostics.

Whole-session event means and the two-decoder flag agreement, plus their
command-line formatting. Kept apart from :mod:`figure04_workflow` so that
editing a summary or its printed text does not change the decode cache's
source fingerprint and refit the models.
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


@dataclasses.dataclass(frozen=True)
class Figure4DiagnosticMeans:
    """Whole-session mean of each per-spike diagnostic for one decoder."""

    hpd_overlap: float
    kl_divergence: float
    predictive_pvalue: float


@dataclasses.dataclass(frozen=True)
class Figure4Summary:
    """Manuscript-facing Figure-4 scalars, independent of output formatting.

    ``n_units`` is the number of simultaneously recorded units the decoders
    were fit on — reported in the figure caption, so it is carried here
    rather than recounted at render time.
    """

    continuous: Figure4DiagnosticMeans
    continuous_fragmented: Figure4DiagnosticMeans
    flag_confusions: tuple[FlagConfusion, ...]
    n_units: int


def _compute_diagnostic_means(
    diagnostics: SpikeEventDiagnostics,
) -> Figure4DiagnosticMeans:
    """Compute the three manuscript diagnostic means for one decoder."""
    return Figure4DiagnosticMeans(
        hpd_overlap=compute_mean_spike_event_diagnostic(diagnostics, "hpd_overlap"),
        kl_divergence=compute_mean_spike_event_diagnostic(diagnostics, "kl_divergence"),
        predictive_pvalue=compute_mean_spike_event_diagnostic(diagnostics, "predictive_pvalue"),
    )


def compute_figure04_summary(
    render_data: Figure4RenderData,
    thresholds: Mapping[str, float],
    metric_directions: Mapping[str, FlagDirection],
) -> Figure4Summary:
    """Compute whole-session event means and two-decoder flag agreement.

    These scalars can appear in the manuscript text, so their computation must
    stay identical to the cached decode; the render layer never alters them.
    """
    decode = render_data.decode_results
    return summarize_figure04_diagnostics(
        decode.continuous_diagnostics,
        decode.continuous_fragmented_diagnostics,
        n_units=int(decode.spike_counts.shape[1]),
        thresholds=thresholds,
        metric_directions=metric_directions,
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

    # "cont-only" is the rescue quadrant: flagged by Continuous but not by
    # Continuous-Fragmented. Rescue rate is its fraction of Continuous flags.
    lines.extend(
        [
            "",
            f"=== Flag agreement: {CONTINUOUS.label} (A) vs {CONTINUOUS_FRAGMENTED.label} (B) ===",
        ]
    )
    for confusion in summary.flag_confusions:
        lines.append(
            f"  {confusion.metric}: n={confusion.n:,} both={confusion.both:,} "
            f"cont-only={confusion.a_only:,} cf-only={confusion.b_only:,} "
            f"neither={confusion.neither:,} rescue={100 * confusion.rescue_rate:.1f}%"
        )
    return "\n".join(lines)
