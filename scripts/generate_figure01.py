"""Create Figure 1: Combined state space model schematic and distribution consistency.

This figure combines:
- Panel a: State space model schematic showing graphical model and filtering equations
- Panels b-e: Distribution comparison showing consistent/inconsistent scenarios

The schematic demonstrates the two-step Bayesian filtering process:
1. Prediction: Convolve previous posterior with transition to get predictive
2. Update: Multiply predictive with likelihood to get current posterior

The distribution panels show scenarios where predictive and normalized likelihood
are consistent or inconsistent, using HPD overlap as a diagnostic.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from scipy import stats

# Import schematic drawing functions
from state_space_schematic import draw_equation_boxes, draw_graphical_model

from statespacecheck_paper.plotting import compute_hpd_region
from statespacecheck_paper.style import COLORS, save_figure, set_figure_defaults


def _create_distribution_panel(
    ax: Axes,
    x: np.ndarray,
    scenario: dict[str, Any],
    panel_label: str,
    color_predictive: str,
    color_likelihood: str,
    show_legend: bool = False,
    show_ylabel: bool = True,
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
    panel_label : str
        Panel label (e.g., 'b', 'c', 'd', 'e').
    color_predictive : str
        Color for predictive distribution.
    color_likelihood : str
        Color for likelihood distribution.
    show_legend : bool, default False
        Whether to show legend on this panel.
    show_ylabel : bool, default True
        Whether to show y-axis label.
    """
    # Generate distributions
    pred_mean, pred_std = scenario["predictive"]
    like_mean, like_std = scenario["likelihood"]

    pdf_predictive = stats.norm.pdf(x, loc=pred_mean, scale=pred_std)
    pdf_likelihood = stats.norm.pdf(x, loc=like_mean, scale=like_std)

    # Normalize likelihood
    dx = x[1] - x[0]
    pdf_likelihood = pdf_likelihood / (np.sum(pdf_likelihood) * dx)

    # Plot distributions (matching style from panel b equation boxes)
    ax.plot(
        x,
        pdf_predictive,
        color=color_predictive,
        linewidth=1.2,
        label="Predictive distribution",
    )
    ax.fill_between(x, pdf_predictive, alpha=0.3, color=color_predictive)

    ax.plot(
        x,
        pdf_likelihood,
        color=color_likelihood,
        linewidth=1.2,
        label="Normalized likelihood",
    )
    ax.fill_between(x, pdf_likelihood, alpha=0.3, color=color_likelihood)

    # Compute HPD regions
    hpd_predictive = compute_hpd_region(x, pdf_predictive, coverage=0.95)
    hpd_likelihood = compute_hpd_region(x, pdf_likelihood, coverage=0.95)

    # Extract contiguous regions
    pred_regions: list[tuple[float, float]] = []
    like_regions: list[tuple[float, float]] = []
    for mask, regions_list in [
        (hpd_predictive, pred_regions),
        (hpd_likelihood, like_regions),
    ]:
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

    # Formatting - minimal style matching panel b
    ax.set_xlim(-20, 20)
    ax.set_ylim(-0.1, 0.30)  # Tighter y-limit to bring title closer to distributions
    ax.set_title(scenario["title"], fontsize=7, fontweight="normal", pad=2)

    # Turn off all axes (matching panel b style)
    ax.axis("off")

    # Add panel label
    ax.text(
        -0.12,
        1.08,
        panel_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # Add legend if requested
    if show_legend:
        ax.legend(loc="upper left", fontsize=6, frameon=False)


def create_figure() -> None:
    """Create Figure 1 combining schematic and distribution panels.

    This figure combines:
    - Panel a: State space model graphical model (top)
    - Panel b: Equation boxes (Prediction and Update steps)
    - Panel c: Distribution consistency examples (4 sub-panels, bottom)

    Returns
    -------
    None
        Saves figure to figures/main/figure01.{pdf,png}.
    """
    set_figure_defaults(context="paper")

    # Create figure with GridSpec for precise control
    # 3 rows: graphical model, equation boxes, distribution panels
    fig = plt.figure(figsize=(5.0, 6.5), dpi=450, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, wspace=0.01, hspace=0.02)

    # Create grid: 3 rows, 6 columns
    # Row 0: Graphical model (full width)
    # Row 1: Equation boxes (full width)
    # Row 2: margin + 4 distribution panels + margin
    # Panel B equation boxes span x=0.0 to x=7.15 in xlim=(-0.5, 9.5)
    # That's 5% left margin and 23.5% right margin
    # Width ratios: left_margin=0.05, 4 panels share 0.715, right_margin=0.235
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.6, 1.0, 0.35],  # Reduced panel A height to bring B closer
        width_ratios=[0.05, 0.715 / 4, 0.715 / 4, 0.715 / 4, 0.715 / 4, 0.235],
    )

    # Panel A: Graphical model spans all columns in top row
    axes = {}
    axes["A"] = fig.add_subplot(gs[0, :])

    # Panel B: Equation boxes spans all columns in middle row
    axes["B"] = fig.add_subplot(gs[1, :])

    # Panel C (sub-panels): Distribution panels aligned with equation boxes
    axes["C1"] = fig.add_subplot(gs[2, 1])
    axes["C2"] = fig.add_subplot(gs[2, 2])
    axes["C3"] = fig.add_subplot(gs[2, 3])
    axes["C4"] = fig.add_subplot(gs[2, 4])

    # =========================================================================
    # Panel A: Graphical model
    # =========================================================================
    draw_graphical_model(axes["A"])

    # =========================================================================
    # Panel B: Equation boxes
    # =========================================================================
    draw_equation_boxes(axes["B"])

    # =========================================================================
    # Panel C: Distribution consistency examples (4 sub-panels)
    # =========================================================================

    # Define x-axis for distributions
    x = np.linspace(-20, 20, 1000)

    # Colors from semantic COLORS system
    color_predictive = COLORS["predictive"]  # Blue (#0072B2)
    color_likelihood = COLORS["likelihood"]  # Orange (#E69F00)

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

    sub_panel_names = ["C1", "C2", "C3", "C4"]

    for panel_name, scenario in zip(sub_panel_names, scenarios, strict=True):
        _create_distribution_panel(
            axes[panel_name],
            x,
            scenario,
            "",  # No individual panel labels for sub-panels
            color_predictive,
            color_likelihood,
            show_legend=False,  # No legend - colors are explained in schematic
            show_ylabel=(panel_name == "C1"),  # Y-axis label only on first sub-panel
        )

    # Add panel labels using fig.text() at consistent x position
    # Use panel A's left edge as reference for all labels
    label_x = axes["A"].get_position().x0 - 0.08  # Further left
    for label, ax_key, y_offset in [("a", "A", 0.04), ("b", "B", 0.04), ("c", "C1", -0.02)]:
        fig.text(
            label_x,
            axes[ax_key].get_position().y1 + y_offset,
            label,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # Validate layout before saving
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig.canvas.draw()  # Force layout calculation
        for warning in w:
            print(f"⚠️  Layout warning: {warning.message}")

    # Save to figures/main directory
    save_figure("figures/main/figure01")
    plt.close()
    print("\nFigure 1 saved to figures/main/figure01.{pdf,png}")


if __name__ == "__main__":
    create_figure()
