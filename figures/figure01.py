"""Create Figure 1 showing consistent and inconsistent distributions.

This figure demonstrates scenarios where the predictive distribution (prior)
and normalized likelihood are consistent or inconsistent, using HPD overlap
as a diagnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats


# Fallback colors
WONG = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

def set_figure_defaults(context: str = "paper") -> None:
    """Set matplotlib defaults for publication figures.

    Font sizes meet Nature/Science minimums (5-7pt).
    TrueType font embedding (fonttype 42) required for journal submission.
    """
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,  # TrueType fonts for Nature/Science submission
            "ps.fonttype": 42,  # Required for proper font embedding
        }
    )

def save_figure(name: str) -> None:
    """Save figure as both PDF and PNG."""
    plt.savefig(f"{name}.pdf", dpi=450, bbox_inches="tight")
    plt.savefig(f"{name}.png", dpi=450, bbox_inches="tight")
    print(f"Saved {name}.pdf and {name}.png")


def compute_hpd_region(x: np.ndarray, pdf: np.ndarray, coverage: float = 0.95) -> np.ndarray:
    """Compute highest posterior density region for given coverage.

    Parameters
    ----------
    x : np.ndarray
        Domain values.
        Shape (n_points,)
    pdf : np.ndarray
        Probability density values (must be normalized).
        Shape (n_points,)
    coverage : float
        Desired coverage probability (default 0.95).

    Returns
    -------
    mask : np.ndarray
        Boolean mask indicating points in HPD region.
        Shape (n_points,)
    """
    # Normalize to ensure proper probability
    dx = x[1] - x[0]
    pdf_normalized = pdf / (np.sum(pdf) * dx)

    # Sort by density and find threshold
    sorted_pdf = np.sort(pdf_normalized)[::-1]  # Descending
    cumsum = np.cumsum(sorted_pdf) * dx
    threshold_idx = np.searchsorted(cumsum, coverage)
    if threshold_idx >= len(sorted_pdf):
        threshold_idx = len(sorted_pdf) - 1
    threshold = sorted_pdf[threshold_idx]

    return pdf_normalized >= threshold


def create_figure() -> None:
    """Create Figure 1 with four panels showing distribution consistency.

    This figure demonstrates scenarios where the predictive distribution (prior)
    and normalized likelihood are consistent or inconsistent, using HPD overlap
    as a diagnostic metric.
    """
    import warnings

    set_figure_defaults(context="paper")

    # Create 2x2 grid
    # Figure width: 7.0" matches Nature/Science full-page width (max 7.08")
    # DPI set during save (450), not here - prevents redundancy
    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 5.0), constrained_layout=True, sharex=True, sharey=True
    )

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Define x-axis
    x = np.linspace(-20, 20, 1000)

    # Colors: Wong palette - colorblind-friendly (verified for deuteranopia/protanopia)
    # These colors also work in grayscale and meet journal accessibility requirements
    color_predictive = WONG[5]  # Blue (#0072B2) - matches figure 2
    color_likelihood = WONG[1]  # Orange (#E69F00) - matches figure 2

    # Define scenarios
    scenarios = [
        {
            "title": "Inconsistent",
            "predictive": (0, 1.5),  # (mean, std)
            "likelihood": (5, 1.5),
        },
        {
            "title": "Consistent",
            "predictive": (0, 1.5),
            "likelihood": (2, 3.0),
        },
        {
            "title": "Consistent",
            "predictive": (0, 5.0),
            "likelihood": (5, 1.5),
        },
        {
            "title": "Consistent",
            "predictive": (0, 4.0),
            "likelihood": (5, 3.5),
        },
    ]

    for idx, (ax, scenario) in enumerate(zip(axes_flat, scenarios, strict=True)):
        # Generate distributions using scipy.stats.norm
        pred_mean, pred_std = scenario["predictive"]
        like_mean, like_std = scenario["likelihood"]

        pdf_predictive = stats.norm.pdf(x, loc=pred_mean, scale=pred_std)
        pdf_likelihood = stats.norm.pdf(x, loc=like_mean, scale=like_std)

        # Normalize likelihood (already normalized from scipy, but be explicit)
        dx = x[1] - x[0]
        pdf_likelihood = pdf_likelihood / (np.sum(pdf_likelihood) * dx)

        # Plot distributions
        ax.plot(
            x,
            pdf_predictive,
            color=color_predictive,
            linewidth=1.5,
            label="Predictive distribution",
        )
        ax.plot(
            x, pdf_likelihood, color=color_likelihood, linewidth=1.5, label="Normalized likelihood"
        )

        # Compute HPD regions
        hpd_predictive = compute_hpd_region(x, pdf_predictive, coverage=0.95)
        hpd_likelihood = compute_hpd_region(x, pdf_likelihood, coverage=0.95)

        # Find extent of HPD regions for visualization
        pred_regions = []
        like_regions = []

        # Extract contiguous regions
        for mask, regions_list in [(hpd_predictive, pred_regions), (hpd_likelihood, like_regions)]:
            in_region = False
            start = None
            for i, val in enumerate(mask):
                if val and not in_region:
                    start = x[i]
                    in_region = True
                elif not val and in_region:
                    regions_list.append((start, x[i - 1]))
                    in_region = False
            if in_region:
                regions_list.append((start, x[-1]))

        # Draw HPD regions as horizontal bars at bottom
        bar_height = 0.015
        y_pred = -0.08
        y_like = -0.05

        for start, end in pred_regions:
            ax.add_patch(
                Rectangle(
                    (start, y_pred),
                    end - start,
                    bar_height,
                    facecolor=color_predictive,
                    edgecolor=color_predictive,
                    linewidth=1.0,
                    clip_on=False,
                )
            )

        for start, end in like_regions:
            ax.add_patch(
                Rectangle(
                    (start, y_like),
                    end - start,
                    bar_height,
                    facecolor=color_likelihood,
                    edgecolor=color_likelihood,
                    linewidth=1.0,
                    clip_on=False,
                )
            )

        # Formatting
        ax.set_xlim(-20, 20)
        ax.set_ylim(-0.1, 0.45)
        ax.set_xlabel("Position (a.u.)", fontsize=7, labelpad=8)
        ax.set_ylabel("Probability Density", fontsize=7, labelpad=8)
        ax.set_title(scenario["title"], fontsize=8, fontweight="bold", pad=8)

        # Spines and ticks - following Tufte's minimal ink principle
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Set spine bounds to data range (Tufte range frames)
        ax.spines["left"].set_bounds(ax.get_ylim())
        ax.spines["bottom"].set_bounds(ax.get_xlim())
        ax.tick_params(labelsize=6)

        # Add panel label (a, b, c, d) outside plot area
        panel_labels = ["a", "b", "c", "d"]
        ax.text(
            -0.12,
            1.08,
            panel_labels[idx],
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

        # Add legend to first panel only
        if idx == 0:
            ax.legend(loc="upper left", fontsize=6, frameon=False)

    # Validate layout before saving
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig.canvas.draw()  # Force layout calculation
        if w:
            print(f"⚠️  Layout warning: {w[-1].message}")

    # Save to figures directory
    import os

    figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "figure01")
    save_figure(save_path)
    plt.close()
    print(f"\nFigure 1 saved to {save_path}.{{pdf,png}}")


if __name__ == "__main__":
    create_figure()
