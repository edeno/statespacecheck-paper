"""Figure styling utilities for consistent publication-ready figures.

This module provides styling constants and functions to ensure all figures
in the paper have consistent appearance that meets journal requirements
(Nature, Science, Cell, etc.).

Examples
--------
Basic usage for creating a publication figure:

>>> from statespacecheck_paper.style import (
...     WONG, set_figure_defaults, save_figure
... )
>>> import matplotlib.pyplot as plt
>>> set_figure_defaults(context="paper")
>>> fig, ax = plt.subplots(figsize=(3.5, 2.3))
>>> _ = ax.plot([1, 2, 3], [1, 2, 3], color=WONG[1])
>>> save_figure("manuscript/figures/my_figure", fig=fig)  # doctest: +SKIP

For presentations:

>>> set_figure_defaults(context="presentation")
>>> fig, ax = plt.subplots(figsize=(7.0, 4.7))
>>> save_figure("manuscript/figures/presentation_figure", dpi=300, fig=fig)  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from statespacecheck_paper.diagnostics import METRIC_FLAG_DIRECTIONS, FlagDirection

# Wong colorblind-friendly palette
# Reference: Wong, B. (2011). Points of view: Color blindness.
# Nature Methods 8, 441. https://doi.org/10.1038/nmeth.1618
WONG = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
]

# =============================================================================
# Semantic Color System
# =============================================================================
# Provides consistent colors for concepts across all figures.
# Design principles:
# 1. Colorblind accessible (uses WONG palette)
# 2. Semantic consistency (same concept = same color everywhere)
# 3. Visual hierarchy (primary concepts are saturated, secondary are muted)
# 4. Print compatible (distinct in grayscale)
#
# Usage:
#   from statespacecheck_paper.style import COLORS
#   ax.plot(x, predictive, color=COLORS["predictive"])
#   ax.fill_between(x, likelihood, color=COLORS["likelihood"], alpha=0.3)

COLORS: dict[str, str] = {
    # -------------------------------------------------------------------------
    # Primary Distributions (the core comparison in all diagnostics)
    # -------------------------------------------------------------------------
    # Predictive: "What we expect" - the one-step-ahead prediction from the model
    # Blue chosen for "cool" association with prior/prediction (before evidence)
    "predictive": WONG[5],  # Blue
    #
    # Likelihood: "What we observe" - evidence from current observations
    # Orange chosen for "warm" association with data/evidence (new information)
    "likelihood": WONG[1],  # Orange
    #
    # Posterior: "What we believe" - combined belief after incorporating evidence
    # Black chosen as neutral, authoritative color (the "answer")
    "posterior": WONG[0],  # Black
    #
    # -------------------------------------------------------------------------
    # Ground Truth and Reference
    # -------------------------------------------------------------------------
    # True position/state - must be highly visible but distinct from distributions
    # Magenta chosen for high visibility and distinction from all other colors
    "ground_truth": "#FF00FF",  # Magenta
    #
    # Threshold lines - subtle reference, should not compete with data
    "threshold": "#666666",  # Dark gray
    #
    # Zero/baseline reference lines
    "reference": "#999999",  # Medium gray
    #
    # Secondary annotation - captions, guide lines, and neutral elements such
    # as the transition model in the Figure-1 schematic. Same gray as the
    # threshold, but a different role.
    "annotation": "#666666",  # Dark gray
    #
    # -------------------------------------------------------------------------
    # Diagnostic Metrics
    # -------------------------------------------------------------------------
    # HPD Overlap metric - related to distributions but distinct
    # Sky blue: lighter than predictive blue, suggests "overlap/intersection"
    "hpd_overlap": WONG[2],  # Sky Blue
    #
    # KL Divergence metric - measures information difference
    # Bluish green: distinct from both primary colors, suggests "divergence"
    "kl_divergence": WONG[3],  # Bluish Green
    #
    # Predictive p-value metric
    "predictive_pvalue": WONG[7],  # Reddish Purple
    #
    # -------------------------------------------------------------------------
    # Figure-3 Replay Band
    # -------------------------------------------------------------------------
    # Replay event (in clean-recovery 2) — immobile animal, decoded
    # trajectory sweeps the track; a control. Used to mark
    # the replay band in the Figure-3 time series.
    "replay": WONG[3],  # Bluish green; marks the replay band
    #
    # -------------------------------------------------------------------------
    # Heatmap Colormaps
    # -------------------------------------------------------------------------
    # See ``CMAP_*`` constants below for the matplotlib colormaps.
}

# Resolution of the paper's figures, in dots per inch: the canonical figures are
# composed and saved at it. 450 dpi meets most journal requirements (Nature
# requires 300-600 dpi for final figures).
FIGURE_DPI = 450

# Colormap constants (can't be in dict since they're not colors)
CMAP_PREDICTIVE = "bone_r"  # Reversed bone for the predictive heatmaps
CMAP_LIKELIHOOD = "inferno"  # Warm colormap for likelihood overlay at spike times

# Top of the predictive heatmaps' color scale: this quantile of the plotted
# distribution, for robustness to outliers. Figure 3a and the website share it.
PREDICTIVE_VMAX_QUANTILE = 0.975

# Base-10 symlog scale of a ``MetricSpec.symlog_axis`` row: linear within
# ``SYMLOG_LINTHRESH`` of zero, logarithmic beyond, so values near zero stay
# separated from exact zeros. The website's HPD-overlap axis mirrors it.
SYMLOG_LINTHRESH = 0.01
SYMLOG_LINSCALE = 1.0


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert a ``#RRGGBB`` color string to an ``(R, G, B)`` int tuple.

    Used by the pyqtgraph interactive viewer, which takes RGB tuples
    instead of hex strings for ``pg.mkPen`` / ``pg.mkBrush``. Keeps the
    paper's WONG palette (declared as hex in :data:`COLORS`) as the
    single source of truth.
    """
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"hex_to_rgb expected '#RRGGBB'; got {hex_str!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


MetricName = Literal["hpd_overlap", "kl_divergence", "predictive_pvalue"]


DisplayTransform = Literal["identity", "neg_log_p"]


@dataclass(frozen=True)
class MetricSpec:
    """Display metadata for one diagnostic metric.

    Single source of truth for how each metric is rendered and flagged on a
    plotted axis: its color, display transform, label, and worse-fit
    direction, shared by ``figure03_plotting``, ``figure04_panels``, and the
    interactive viewer.

    ``plotted_worse`` is the worse-fit direction **on the plotted axis** (after
    ``display_transform``), derived from the flag rule's
    :data:`~statespacecheck_paper.diagnostics.METRIC_FLAG_DIRECTIONS` so the
    ``worse_fit_direction`` arrow and the hexbin "rescue" quadrant cannot
    disagree with the flags.

    Each metric carries the label variants its renderers need, so no consumer
    restates a metric's label or relies on the registry's order:

    - ``label``: plain-text full name (the pyqtgraph viewer, which renders no
      mathtext).
    - ``short_label``: plain-text abbreviated axis label (Figure 3's rows; its
      summary heatmap stacks the words one per line).
    - ``wrapped_ylabel``: mathtext abbreviated axis label, broken over two
      lines where it would not fit beside Figure 4's narrow detail rows.
    - ``title``: mathtext full name (Figure 4's hexbin titles).
    """

    name: MetricName
    color: str
    label: str
    short_label: str
    wrapped_ylabel: str
    title: str
    display_transform: DisplayTransform = "identity"
    symlog_axis: bool = False

    @property
    def plotted_worse(self) -> FlagDirection:
        """Worse-fit direction on the plotted axis.

        The raw flag direction, flipped by the monotonically decreasing
        ``-log(p)`` transform.
        """
        raw = METRIC_FLAG_DIRECTIONS[self.name]
        if self.display_transform == "neg_log_p":
            return "above" if raw == "below" else "below"
        return raw

    @property
    def event_attr(self) -> str:
        """Name of the per-spike-event attribute on ``SpikeEventDiagnostics``."""
        return f"event_{self.name}"

    @property
    def worse_fit_direction(self) -> str:
        """Worse-fit arrow annotation for the plotted axis."""
        return "↓ Worse fit" if self.plotted_worse == "below" else "↑ Worse fit"


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="hpd_overlap",
        color=COLORS["hpd_overlap"],
        label="HPD overlap",
        short_label="HPD overlap",
        wrapped_ylabel="HPD\noverlap",
        title="HPD overlap",
        symlog_axis=True,
    ),
    MetricSpec(
        name="predictive_pvalue",
        color=COLORS["predictive_pvalue"],
        label="−log(p)",
        short_label="−log(p)",
        wrapped_ylabel=r"$-\log(p)$",
        title=r"$-\log(p)$",
        display_transform="neg_log_p",
    ),
    MetricSpec(
        name="kl_divergence",
        color=COLORS["kl_divergence"],
        label="KL divergence",
        short_label="KL div.",
        wrapped_ylabel="KL div.",
        title="KL divergence",
    ),
)
METRIC_NAMES: tuple[MetricName, ...] = tuple(s.name for s in METRIC_SPECS)
METRIC_SPEC_BY_NAME: dict[str, MetricSpec] = {s.name: s for s in METRIC_SPECS}


def set_figure_defaults(context: Literal["paper", "presentation", "poster"] = "paper") -> None:
    """Set matplotlib defaults for publication figures.

    Configures matplotlib rcParams for consistent, publication-ready figures
    with appropriate font sizes for different contexts. All settings ensure
    compatibility with journal requirements (Nature, Science, Cell, etc.).

    Parameters
    ----------
    context : {"paper", "presentation", "poster"}, default "paper"
        Context for figure display:
        - "paper": Small fonts (8pt base) for journal publications
        - "presentation": Medium fonts (12pt base) for talks/slides
        - "poster": Large fonts (16pt base) for conference posters

    Returns
    -------
    None

    Notes
    -----
    - Font sizes for "paper" context meet the 8-12 pt in-figure range most journals require
    - TrueType font embedding (fonttype 42) required for journal submission
    - Uses Arial font family (widely available and accepted by journals)
    - Sets thin line widths (0.5pt) for professional appearance

    Examples
    --------
    For a journal manuscript:

    >>> set_figure_defaults(context="paper")
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([1, 2, 3], [1, 2, 3])

    For a presentation:

    >>> set_figure_defaults(context="presentation")
    >>> fig, ax = plt.subplots(figsize=(10, 6))
    """
    # Font sizes for different contexts
    font_sizes = {
        "paper": {
            # Journals typically require 8-12 pt for all in-figure text; 8 pt is the floor.
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        },
        "presentation": {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        },
        "poster": {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        },
    }

    sizes = font_sizes[context]
    plt.rcParams.update(
        {
            **sizes,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,  # TrueType fonts for Nature/Science submission
            "ps.fonttype": 42,  # Required for proper font embedding
        }
    )


def save_figure(
    name: str | Path,
    dpi: int = FIGURE_DPI,
    close: bool = True,
    bbox_inches: object = "tight",
    *,
    fig: Figure,
) -> None:
    """Save figure as both PDF and PNG with journal-quality resolution.

    Creates both vector (PDF) and raster (PNG) versions of the figure.
    Automatically creates parent directories if they don't exist.

    Parameters
    ----------
    name : str or Path
        Output filename without extension. Both .pdf and .png will be added.
        Can be a string path or pathlib.Path object.
    dpi : int, default ``FIGURE_DPI``
        Resolution in dots per inch.
    close : bool, default True
        If True, close the figure after saving to free memory.
    bbox_inches : object, default "tight"
        Bounding box passed through to ``Figure.savefig``: ``"tight"`` crops to
        the drawn artists, and a precomputed ``matplotlib.transforms.Bbox``
        gives a custom crop.
    fig : matplotlib.figure.Figure
        The figure to save (keyword-only).

    Returns
    -------
    None
        Files are saved to disk as side effects. Prints confirmation message
        with saved file paths.

    Examples
    --------
    Basic usage (the ``save_figure`` calls are marked ``+SKIP`` because
    they write PDF/PNG files to disk):

    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([1, 2, 3], [1, 2, 3])
    >>> save_figure("manuscript/figures/my_figure", fig=fig)  # doctest: +SKIP
    Saved manuscript/figures/my_figure.pdf and manuscript/figures/my_figure.png

    With custom DPI and keeping figure open:

    >>> save_figure("manuscript/figures/my_figure", dpi=300, close=False, fig=fig)  # doctest: +SKIP

    Using Path object:

    >>> from pathlib import Path
    >>> output_path = Path("results") / "figure1"
    >>> save_figure(output_path, fig=fig)  # doctest: +SKIP

    Auto-creates nested directories:

    >>> save_figure("manuscript/figures/supplementary/figure_s1", fig=fig)  # doctest: +SKIP
    """
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)

    print(f"Saved {pdf_path} and {png_path}")

    if close:
        plt.close(fig)
