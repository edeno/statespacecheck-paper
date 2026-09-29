"""Figure-1 composition and generation recipe.

This figure combines (see :func:`compose_figure01` for the exact layout):
- Panel a: the state space model graphical model
- Panel b: the Bayesian filtering equation boxes (Prediction and Update steps)
- Panel c: distribution comparison (four sub-panels) showing consistent /
  inconsistent scenarios

The schematic demonstrates the two-step Bayesian filtering process:
1. Prediction: Convolve previous posterior with transition to get predictive
2. Update: Multiply predictive with likelihood to get current posterior

The distribution panels show scenarios where predictive and normalized likelihood
are consistent or inconsistent, using HPD overlap as a diagnostic.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import statespacecheck as ssc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import NDArray
from scipy import stats

from statespacecheck_paper.diagnostics import HPD_COVERAGE
from statespacecheck_paper.plotting import extract_contiguous_regions
from statespacecheck_paper.schematic import draw_equation_boxes, draw_graphical_model
from statespacecheck_paper.style import COLORS, save_figure, set_figure_defaults


def create_distribution_comparison_panel(
    ax: Axes,
    x: NDArray[np.floating],
    predictive_params: tuple[float, float],
    likelihood_params: tuple[float, float],
    color_predictive: str,
    color_likelihood: str,
    title: str | None = None,
    show_labels: bool = False,
    coverage: float = HPD_COVERAGE,
) -> None:
    """Create a panel comparing predictive and likelihood distributions.

    Shows both distributions with filled curves and HPD regions as
    horizontal bars below the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x : np.ndarray, shape (n_points,)
        Position values for plotting.
    predictive_params : tuple[float, float]
        (mean, std) for predictive Gaussian distribution.
    likelihood_params : tuple[float, float]
        (mean, std) for likelihood Gaussian distribution.
    color_predictive : str
        Color for predictive distribution.
    color_likelihood : str
        Color for likelihood distribution.
    title : str | None, optional
        Panel title.
    show_labels : bool, default False
        Whether to show "Predictive"/"Likelihood" text labels on curves.
    coverage : float, default ``HPD_COVERAGE``
        Coverage probability for HPD regions.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-20, 20, 1000)
    >>> create_distribution_comparison_panel(
    ...     ax, x,
    ...     predictive_params=(0, 1.5),
    ...     likelihood_params=(5, 1.5),
    ...     color_predictive="blue",
    ...     color_likelihood="orange",
    ...     title="Example",
    ... )
    >>> plt.close(fig)
    """
    # Generate distributions
    pred_mean, pred_std = predictive_params
    like_mean, like_std = likelihood_params

    pdf_predictive: NDArray[np.floating] = stats.norm.pdf(x, loc=pred_mean, scale=pred_std)
    pdf_likelihood: NDArray[np.floating] = stats.norm.pdf(x, loc=like_mean, scale=like_std)

    dx = float(x[1] - x[0])
    pdf_likelihood = pdf_likelihood / (np.sum(pdf_likelihood) * dx)

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

    # Compute HPD regions and extract contiguous intervals
    hpd_predictive = ssc.highest_density_region(pdf_predictive[np.newaxis], coverage=coverage)[0]
    hpd_likelihood = ssc.highest_density_region(pdf_likelihood[np.newaxis], coverage=coverage)[0]
    pred_regions = extract_contiguous_regions(hpd_predictive, x)
    like_regions = extract_contiguous_regions(hpd_likelihood, x)

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
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(-0.1, 0.30)  # Room for sub-panel titles
    if title:
        ax.set_title(title, fontweight="normal", pad=2)

    ax.axis("off")

    # Add direct labels on distribution curves
    if show_labels:
        # Label predictive on left side, likelihood on right side
        ax.text(
            -12,
            0.22,
            "Predictive",
            ha="center",
            va="bottom",
            color=color_predictive,
        )
        ax.text(
            16,
            0.22,
            "Likelihood",
            ha="center",
            va="bottom",
            color=color_likelihood,
        )


def compose_figure01() -> Figure:
    """Compose Figure 1 from the schematic and distribution renderers.

    This figure combines:
    - Panel a: State space model graphical model (top)
    - Panel b: Equation boxes (Prediction and Update steps)
    - Panel c: Distribution consistency examples (4 sub-panels, bottom)

    Returns
    -------
    matplotlib.figure.Figure
        The complete in-memory Figure-1 composition.
    """
    # Create figure with GridSpec for precise control
    # 3 rows: graphical model, equation boxes, distribution panels
    fig: Figure = plt.figure(figsize=(5.0, 5.1), dpi=450)

    # Create grid with minimal spacing between rows
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[0.55, 0.65, 0.35],
        width_ratios=[1, 1, 1, 1],
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.05,
        hspace=0.12,  # Moderate vertical space between panels
        wspace=0.02,
    )

    # Panel A: Graphical model spans all columns in top row
    axes: dict[str, Axes] = {}
    axes["graphical_model"] = fig.add_subplot(gs[0, :])

    # Panel B: Equation boxes spans all columns in middle row
    axes["filtering_equations"] = fig.add_subplot(gs[1, :])

    # Panel C: Create spanning axes for title, then sub-panels for content
    # The spanning axes is invisible but holds the "Goodness-of-Fit" title
    axes["goodness_of_fit"] = fig.add_subplot(gs[2, :])
    axes["goodness_of_fit"].axis("off")
    axes["goodness_of_fit"].set_title(
        "Goodness-of-Fit: Predictive vs. Likelihood",
        fontweight="bold",
        pad=4,
    )

    # Sub-panels for distribution plots (using inset_axes for precise positioning)
    # Calculate sub-panel positions within the spanning axes
    sub_width = 0.23  # Width of each sub-panel as fraction of parent
    sub_gap = 0.02  # Gap between sub-panels
    sub_left_positions = [i * (sub_width + sub_gap) + 0.02 for i in range(4)]

    distribution_axis_names = (
        "inconsistent_narrow",
        "consistent_broad",
        "consistent_nearby",
        "consistent_mixed_width",
    )
    for i, axis_name in enumerate(distribution_axis_names):
        axes[axis_name] = inset_axes(
            axes["goodness_of_fit"],
            width="100%",
            height="100%",
            bbox_to_anchor=(sub_left_positions[i], 0.0, sub_width, 0.85),
            bbox_transform=axes["goodness_of_fit"].transAxes,
            borderpad=0,
        )

    # =========================================================================
    # Panel A: Graphical model
    # =========================================================================
    draw_graphical_model(axes["graphical_model"])

    # =========================================================================
    # Panel B: Equation boxes
    # =========================================================================
    draw_equation_boxes(axes["filtering_equations"])

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
        ("Consistent", (0, 4.0), (5, 3.5)),
        ("Consistent", (0, 1.5), (2, 3.0)),
        ("Consistent", (0, 5.0), (5, 1.5)),
    ]

    for i, (axis_name, (title, pred_params, like_params)) in enumerate(
        zip(distribution_axis_names, scenarios, strict=True)
    ):
        create_distribution_comparison_panel(
            axes[axis_name],
            x,
            predictive_params=pred_params,
            likelihood_params=like_params,
            color_predictive=color_predictive,
            color_likelihood=color_likelihood,
            title=title,
            show_labels=(i == 0),  # Only show labels on first panel
        )

    # Add shared x-axis label for Panel C
    c_pos = axes["goodness_of_fit"].get_position()
    fig.text(
        (c_pos.x0 + c_pos.x1) / 2,
        c_pos.y0 - 0.02,
        "Latent state",
        ha="center",
        va="top",
    )

    # Add panel labels (a, b, c) - now consistent since all panels use set_title()
    label_x = axes["goodness_of_fit"].get_position().x0 - 0.02
    for label, axis_name in (
        ("a", "graphical_model"),
        ("b", "filtering_equations"),
        ("c", "goodness_of_fit"),
    ):
        fig.text(
            label_x,
            axes[axis_name].get_position().y1 + 0.01,
            label,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    return fig


def generate_figure01() -> None:
    """Compose Figure 1 with paper styling and save its PDF and PNG."""
    set_figure_defaults(context="paper")
    fig = compose_figure01()
    save_figure("manuscript/figures/main/figure01", close=True, fig=fig)
    print("\nFigure 1 saved to manuscript/figures/main/figure01.{pdf,png}")
