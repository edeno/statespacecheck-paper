"""Create Figure 1 showing consistent and inconsistent distributions.

This figure demonstrates scenarios where the predictive distribution (prior)
and normalized likelihood are consistent or inconsistent, using HPD overlap
as a diagnostic.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy import stats

from statespacecheck_paper.plotting import compute_hpd_region
from statespacecheck_paper.style import WONG, save_figure, set_figure_defaults


def _create_panel(
    ax: Axes,
    x: np.ndarray,
    scenario: dict[str, Any],
    idx: int,
    color_predictive: str,
    color_likelihood: str,
) -> None:
    """Create a single panel showing predictive and likelihood distributions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x : np.ndarray, shape (n_points,)
        Position values for plotting.
    scenario : dict
        Dictionary with 'title', 'predictive', and 'likelihood' keys.
    idx : int
        Panel index (0-3) for labeling.
    color_predictive : str
        Color for predictive distribution.
    color_likelihood : str
        Color for likelihood distribution.
    """
    # Generate distributions
    pred_mean, pred_std = scenario["predictive"]
    like_mean, like_std = scenario["likelihood"]

    pdf_predictive = stats.norm.pdf(x, loc=pred_mean, scale=pred_std)
    pdf_likelihood = stats.norm.pdf(x, loc=like_mean, scale=like_std)

    # Normalize likelihood
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
    ax.plot(x, pdf_likelihood, color=color_likelihood, linewidth=1.5, label="Normalized likelihood")

    # Compute HPD regions
    hpd_predictive = compute_hpd_region(x, pdf_predictive, coverage=0.95)
    hpd_likelihood = compute_hpd_region(x, pdf_likelihood, coverage=0.95)

    # Extract contiguous regions
    pred_regions: list[tuple[float, float]] = []
    like_regions: list[tuple[float, float]] = []
    for mask, regions_list in [(hpd_predictive, pred_regions), (hpd_likelihood, like_regions)]:
        in_region = False
        start: float | None = None
        for i, val in enumerate(mask):
            if val and not in_region:
                start = float(x[i])
                in_region = True
            elif not val and in_region:
                if start is not None:
                    regions_list.append((start, float(x[i - 1])))
                in_region = False
        if in_region and start is not None:
            regions_list.append((start, float(x[-1])))

    # Draw HPD regions as horizontal bars
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

    # Spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.spines["left"].set_bounds(ylim[0], ylim[1])
    ax.spines["bottom"].set_bounds(xlim[0], xlim[1])
    ax.tick_params(labelsize=6)

    # Add panel label
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
        _create_panel(ax, x, scenario, idx, color_predictive, color_likelihood)

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
