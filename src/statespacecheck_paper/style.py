"""Figure styling utilities for consistent publication-ready figures.

This module provides styling constants and functions to ensure all figures
in the paper have consistent appearance that meets journal requirements
(Nature, Science, Cell, etc.).

Examples
--------
Basic usage for creating a publication figure:

>>> from statespacecheck_paper.style import WONG, set_figure_defaults, save_figure
>>> import matplotlib.pyplot as plt
>>> set_figure_defaults(context="paper")
>>> fig, ax = plt.subplots(figsize=get_figure_size("single"))
>>> ax.plot([1, 2, 3], [1, 2, 3], color=WONG[1])
>>> save_figure("figures/my_figure")

For presentations:

>>> set_figure_defaults(context="presentation")
>>> fig, ax = plt.subplots(figsize=get_figure_size("double"))
>>> save_figure("figures/presentation_figure", dpi=300)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

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
    "predictive": "#0072B2",  # WONG[5] Blue
    #
    # Likelihood: "What we observe" - evidence from current observations
    # Orange chosen for "warm" association with data/evidence (new information)
    "likelihood": "#E69F00",  # WONG[1] Orange
    #
    # Posterior: "What we believe" - combined belief after incorporating evidence
    # Black chosen as neutral, authoritative color (the "answer")
    "posterior": "#000000",  # WONG[0] Black
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
    # -------------------------------------------------------------------------
    # Diagnostic Metrics
    # -------------------------------------------------------------------------
    # HPD Overlap metric - related to distributions but distinct
    # Sky blue: lighter than predictive blue, suggests "overlap/intersection"
    "hpd_overlap": "#56B4E9",  # WONG[2] Sky Blue
    #
    # KL Divergence metric - measures information difference
    # Bluish green: distinct from both primary colors, suggests "divergence"
    "kl_divergence": "#009E73",  # WONG[3] Bluish Green
    #
    # Combined/summary metric (e.g., p-value)
    "metric_combined": "#CC79A7",  # WONG[7] Reddish Purple
    #
    # -------------------------------------------------------------------------
    # KL Decomposition (for mechanics figures)
    # -------------------------------------------------------------------------
    # Positive log ratio: posterior > likelihood (model expects more)
    "kl_positive": "#009E73",  # WONG[3] Bluish Green
    #
    # Negative log ratio: posterior < likelihood (model expects less)
    "kl_negative": "#0072B2",  # WONG[5] Blue
    #
    # Pointwise KL contribution
    "kl_pointwise": "#56B4E9",  # WONG[2] Sky Blue
    #
    # -------------------------------------------------------------------------
    # Experimental Phase Backgrounds (very light, ~15% saturation)
    # -------------------------------------------------------------------------
    # These are derived from WONG colors but lightened significantly
    # to serve as subtle background indicators without competing with data
    #
    # Baseline period - no manipulation
    "phase_baseline": "#FFFFFF",  # White
    #
    # Remapping period - place field remapping (related to likelihood change)
    "phase_remap": "#FFF0D6",  # Light orange (from WONG[1])
    #
    # Flat firing period - constant firing rates
    "phase_flat": "#E8E8E8",  # Light gray (neutral)
    #
    # Fast movement period - rapid position changes
    "phase_fast": "#FFE0D6",  # Light vermillion (from WONG[6])
    #
    # Slow/stationary period - minimal movement
    "phase_slow": "#D6E8FF",  # Light blue (from WONG[5])
    #
    # -------------------------------------------------------------------------
    # Heatmap Colormaps
    # -------------------------------------------------------------------------
    # Use these string values with matplotlib's cmap parameter
    # "heatmap_posterior": "bone_r"  # Defined as constant below
}

# Colormap constants (can't be in dict since they're not colors)
CMAP_POSTERIOR = "bone_r"  # Reversed bone for posterior heatmaps
CMAP_DIAGNOSTIC = "bone_r"  # Same for diagnostic heatmaps (consistency)

# Convenience aliases for common use cases
COLORS["prior"] = COLORS["predictive"]  # Alias: prior = predictive
COLORS["one_step"] = COLORS["predictive"]  # Alias: one-step prediction
COLORS["observation"] = COLORS["likelihood"]  # Alias: observation evidence
COLORS["true_position"] = COLORS["ground_truth"]  # Alias: true position


def set_figure_defaults(context: Literal["paper", "presentation", "poster"] = "paper") -> None:
    """Set matplotlib defaults for publication figures.

    Configures matplotlib rcParams for consistent, publication-ready figures
    with appropriate font sizes for different contexts. All settings ensure
    compatibility with journal requirements (Nature, Science, Cell, etc.).

    Parameters
    ----------
    context : {"paper", "presentation", "poster"}, default "paper"
        Context for figure display:
        - "paper": Small fonts (7pt base) for journal publications
        - "presentation": Medium fonts (12pt base) for talks/slides
        - "poster": Large fonts (16pt base) for conference posters

    Returns
    -------
    None

    Notes
    -----
    - Font sizes for "paper" context meet Nature/Science minimums (5-7pt)
    - TrueType font embedding (fonttype 42) required for journal submission
    - Uses Arial font family (widely available and accepted by journals)
    - Sets thin line widths (0.5pt) for professional appearance

    Examples
    --------
    For a journal manuscript:

    >>> set_figure_defaults(context="paper")
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 2, 3])

    For a presentation:

    >>> set_figure_defaults(context="presentation")
    >>> fig, ax = plt.subplots(figsize=(10, 6))
    """
    # Font sizes for different contexts
    font_sizes = {
        "paper": {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
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

    # Get font sizes for selected context
    sizes = font_sizes[context]

    # Apply all settings
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
    dpi: int = 450,
    close: bool = True,
) -> None:
    """Save figure as both PDF and PNG with journal-quality resolution.

    Creates both vector (PDF) and raster (PNG) versions of the figure.
    Automatically creates parent directories if they don't exist.

    Parameters
    ----------
    name : str or Path
        Output filename without extension. Both .pdf and .png will be added.
        Can be a string path or pathlib.Path object.
    dpi : int, default 450
        Resolution in dots per inch. Default 450 meets most journal requirements
        (Nature requires 300-600 dpi for final figures).
    close : bool, default True
        If True, close the figure after saving to free memory.

    Returns
    -------
    None
        Files are saved to disk as side effects. Prints confirmation message
        with saved file paths.

    Examples
    --------
    Basic usage:

    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> save_figure("figures/my_figure")
    Saved figures/my_figure.pdf and figures/my_figure.png

    With custom DPI and keeping figure open:

    >>> save_figure("figures/my_figure", dpi=300, close=False)

    Using Path object:

    >>> from pathlib import Path
    >>> output_path = Path("results") / "figure1"
    >>> save_figure(output_path)

    Auto-creates nested directories:

    >>> save_figure("figures/supplementary/figure_s1")  # Creates figures/supplementary/
    """
    # Convert to Path object for easier handling
    path = Path(name)

    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save both formats using pathlib
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")

    print(f"Saved {pdf_path} and {png_path}")

    # Close figure if requested
    if close:
        plt.close()


def get_figure_size(
    width_type: Literal["single", "double", "full"] = "single",
    aspect_ratio: float = 1.5,
) -> tuple[float, float]:
    """Get figure size in inches for different column widths.

    Provides standard figure sizes that fit journal column widths.
    Most journals use similar column widths (Nature, Science, Cell, etc.).

    Parameters
    ----------
    width_type : {"single", "double", "full"}, default "single"
        Figure width type:
        - "single": Single column width (~3.5 inches)
        - "double": Double column width (~7.0 inches)
        - "full": Full page width (~7.0 inches, same as double)
    aspect_ratio : float, default 1.5
        Width to height ratio. Default 1.5 gives pleasant proportions.
        Use 1.0 for square figures, 2.0 for wide figures.

    Returns
    -------
    width : float
        Figure width in inches.
    height : float
        Figure height in inches, computed as width / aspect_ratio.

    Notes
    -----
    Standard journal column widths:
    - Nature: Single column 89mm (~3.5"), double column 183mm (~7.2")
    - Science: Single column 90mm (~3.54"), double column 180mm (~7.08")
    - Cell: Single column 85mm (~3.35"), double column 174mm (~6.85")

    This function uses compromise values that work for all major journals.

    Examples
    --------
    Single column figure with default aspect ratio:

    >>> width, height = get_figure_size("single")
    >>> fig, ax = plt.subplots(figsize=(width, height))

    Wide double-column figure:

    >>> width, height = get_figure_size("double", aspect_ratio=2.0)
    >>> fig, axes = plt.subplots(1, 2, figsize=(width, height))

    Square single column figure:

    >>> width, height = get_figure_size("single", aspect_ratio=1.0)
    """
    # Standard column widths in inches
    widths = {
        "single": 3.5,  # Single column (~89mm)
        "double": 7.0,  # Double column (~180mm)
        "full": 7.0,  # Full width (same as double)
    }

    width = widths[width_type]
    height = width / aspect_ratio

    return width, height
