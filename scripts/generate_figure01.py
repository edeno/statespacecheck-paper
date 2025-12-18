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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from statespacecheck_paper.plotting import create_distribution_comparison_panel
from statespacecheck_paper.schematic import draw_equation_boxes, draw_graphical_model
from statespacecheck_paper.style import COLORS, save_figure, set_figure_defaults


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
    fig: Figure = plt.figure(figsize=(5.0, 6.2), dpi=450, constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if isinstance(layout_engine, ConstrainedLayoutEngine):
        layout_engine.set(w_pad=0.01, h_pad=0.02, wspace=0.01, hspace=0.10)

    # Create grid: 3 rows, 4 columns (no margin column needed)
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.6, 0.7, 0.45],
        width_ratios=[1, 1, 1, 1],
    )

    # Panel A: Graphical model spans all columns in top row
    axes = {}
    axes["A"] = fig.add_subplot(gs[0, :])

    # Panel B: Equation boxes spans all columns in middle row
    axes["B"] = fig.add_subplot(gs[1, :])

    # Panel C: Create spanning axes for title, then sub-panels for content
    # The spanning axes is invisible but holds the "Goodness-of-Fit" title
    axes["C"] = fig.add_subplot(gs[2, :])
    axes["C"].axis("off")
    axes["C"].set_title("Goodness-of-Fit", fontsize=8, fontweight="bold", pad=4)

    # Sub-panels for distribution plots (using inset_axes for precise positioning)
    # Calculate sub-panel positions within the spanning axes
    sub_width = 0.23  # Width of each sub-panel as fraction of parent
    sub_gap = 0.02  # Gap between sub-panels
    sub_left_positions = [i * (sub_width + sub_gap) + 0.02 for i in range(4)]

    for i, key in enumerate(["C1", "C2", "C3", "C4"]):
        axes[key] = inset_axes(
            axes["C"],
            width="100%",
            height="100%",
            bbox_to_anchor=(sub_left_positions[i], 0.0, sub_width, 0.85),
            bbox_transform=axes["C"].transAxes,
            borderpad=0,
        )

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
    color_predictive = COLORS["predictive"]
    color_likelihood = COLORS["likelihood"]

    # Define scenarios: (title, predictive_params, likelihood_params)
    scenarios = [
        ("Inconsistent", (0, 1.5), (5, 1.5)),
        ("Consistent", (0, 1.5), (2, 3.0)),
        ("Consistent", (0, 5.0), (5, 1.5)),
        ("Consistent", (0, 4.0), (5, 3.5)),
    ]

    sub_panel_names = ["C1", "C2", "C3", "C4"]

    for i, (panel_name, (title, pred_params, like_params)) in enumerate(
        zip(sub_panel_names, scenarios, strict=True)
    ):
        create_distribution_comparison_panel(
            axes[panel_name],
            x,
            predictive_params=pred_params,
            likelihood_params=like_params,
            color_predictive=color_predictive,
            color_likelihood=color_likelihood,
            title=title,
            show_labels=(i == 0),  # Only show labels on first panel
        )

    # Draw canvas to finalize constrained_layout positions before querying them
    fig.canvas.draw()

    # Add shared x-axis label for Panel C
    c_pos = axes["C"].get_position()
    fig.text(
        (c_pos.x0 + c_pos.x1) / 2,
        c_pos.y0 - 0.02,
        "Latent state",
        ha="center",
        va="top",
        fontsize=7,
    )

    # Add panel labels (a, b, c) - now consistent since all panels use set_title()
    label_x = axes["C"].get_position().x0 - 0.02
    for label, ax_key in [("a", "A"), ("b", "B"), ("c", "C")]:
        fig.text(
            label_x,
            axes[ax_key].get_position().y1 + 0.01,
            label,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # Save to figures/main directory
    save_figure("figures/main/figure01")
    plt.close()
    print("\nFigure 1 saved to figures/main/figure01.{pdf,png}")


if __name__ == "__main__":
    create_figure()
