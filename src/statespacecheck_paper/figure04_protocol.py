"""Figure-4 protocol choices shared by the figure, viewer, website, and lab pipeline.

The fixed flag cutoffs and the manuscript's detail window, kept below the
generation and rendering modules so every consumer imports the same values
without loading matplotlib, the layout, or the workflow. This module imports
only :mod:`diagnostics`.
"""

from __future__ import annotations

import dataclasses
import numbers

from statespacecheck_paper.diagnostics import (
    FIXED_PREDICTIVE_PVALUE_CUTOFF,
    METRIC_FLAG_DIRECTIONS,
    FlagDirection,
)

# Diagnostic thresholds. HPD overlap and the predictive p-value use fixed
# cutoffs of 0.05: the predictive p-value's is the paper-wide cutoff that
# Figure 3 also uses; the HPD-overlap cutoff is chosen for this recording
# (Figure 3 derives its HPD threshold from a baseline instead). The KL
# divergence has no natural fixed cutoff, so it is shown without a threshold
# line or a flagged-region callout.
FIGURE04_HPD_OVERLAP_CUTOFF = 0.05
FIGURE04_DIAGNOSTIC_THRESHOLDS: dict[str, float] = {
    "hpd_overlap": FIGURE04_HPD_OVERLAP_CUTOFF,
    "predictive_pvalue": FIXED_PREDICTIVE_PVALUE_CUTOFF,
}
FIGURE04_METRIC_DIRECTIONS: dict[str, FlagDirection] = {
    metric: METRIC_FLAG_DIRECTIONS[metric] for metric in FIGURE04_DIAGNOSTIC_THRESHOLDS
}


def _is_integer(value: object) -> bool:
    """Return True for Python and NumPy integers, excluding ``bool`` (an ``int`` subclass).

    Using :class:`numbers.Integral` accepts ``np.int64`` etc. (common when an
    index is derived from an array), which a strict ``type(x) is int`` check
    would spuriously reject in this NumPy-heavy codebase.
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


@dataclasses.dataclass(frozen=True)
class Figure4DetailWindow:
    """Index window used for the side-by-side Figure-4 detail panels.

    The canonical window is :data:`FIGURE04_DETAIL_WINDOW`; the layout receives
    a window explicitly so tests and alternate recipes can select a
    scientifically meaningful window without mutating module globals.
    """

    center_index: int
    half_width_samples: int

    def __post_init__(self) -> None:
        if not _is_integer(self.center_index) or self.center_index < 0:
            raise ValueError("center_index must be a non-negative integer")
        if not _is_integer(self.half_width_samples) or self.half_width_samples <= 0:
            raise ValueError("half_width_samples must be a positive integer")

    def to_slice(self, n_time_samples: int) -> slice:
        """Return the validated half-open slice for a recording timeline."""
        if not _is_integer(n_time_samples) or n_time_samples <= 0:
            raise ValueError("n_time_samples must be a positive integer")
        start = self.center_index - self.half_width_samples
        stop = self.center_index + self.half_width_samples
        if start < 0 or stop > n_time_samples:
            raise ValueError(
                "detail window falls outside the recording timeline: "
                f"slice({start}, {stop}) for {n_time_samples} samples"
            )
        return slice(start, stop)


# Manuscript detail view: a KL-divergence spike during immobility at a reward
# well, shown with 500 samples on either side (~2 seconds total at 500 Hz).
FIGURE04_DETAIL_WINDOW = Figure4DetailWindow(
    center_index=193_069,
    half_width_samples=500,
)
