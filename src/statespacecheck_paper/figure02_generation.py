"""Figure-2 composition and generation recipe.

This figure explains the three diagnostic metrics (HPD overlap, predictive
checks, and KL divergence) using a shared synthetic example.

Layout (3 columns over a row of formulas with computed values):
    Outer columns: input distributions, intermediate computation, final result
    Middle column: the four steps of the predictive check

Columns (single panel label per column, left to right):
    a = HPD Overlap mechanics
    b = Predictive Check mechanics
    c = KL Divergence mechanics

Each column has its own rows, and its panels have semantic axis names (for
example, ``hpd_predictive`` and ``kl_pointwise``), so the on-page order and
each renderer's scientific role are visible directly in the layout recipe.

Per-panel renderers live in :mod:`statespacecheck_paper.figure02_panels`.
This module handles layout, panel labels, formula row, and saving.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

from statespacecheck_paper.figure02_panels import (
    create_shared_example,
    plot_hpd_intersection,
    plot_hpd_likelihood,
    plot_hpd_predictive,
    plot_kl_distributions,
    plot_kl_log_ratio,
    plot_kl_pointwise,
    plot_predictive_check_density_histogram,
    plot_predictive_check_mark_scores,
    plot_predictive_check_predictive_fan,
    plot_predictive_check_replicate_marks,
)
from statespacecheck_paper.paths import FIGURE_DIR
from statespacecheck_paper.style import FIGURE_DPI, save_figure, set_figure_defaults


def _add_column_group_backplates(
    fig: Figure,
    axes: Mapping[str, Axes],
) -> None:
    """Add subtle column backplates so each metric reads as one group."""
    fig.draw_without_rendering()
    to_figure = fig.transFigure.inverted()
    column_groups = (
        ("hpd_predictive", "hpd_likelihood", "hpd_overlap", "hpd_formula"),
        (
            "predictive_distribution",
            "predictive_marks",
            "predictive_scores",
            "predictive_histogram",
            "predictive_formula",
        ),
        ("kl_distributions", "kl_log_ratio", "kl_pointwise", "kl_formula"),
    )

    for keys in column_groups:
        # get_tightbbox() uses the figure's renderer after the draw above; it
        # can return None for an empty axes, so skip those.
        bboxes = [
            bbox.transformed(to_figure)
            for key in keys
            if (bbox := axes[key].get_tightbbox()) is not None
        ]
        if not bboxes:
            continue
        bbox = Bbox.union(bboxes)
        x_pad = 0.010
        y_pad = 0.008
        fig.add_artist(
            Rectangle(
                (bbox.x0 - x_pad, bbox.y0 - y_pad),
                bbox.width + 2 * x_pad,
                bbox.height + 2 * y_pad,
                transform=fig.transFigure,
                facecolor="#F8F9FB",
                edgecolor="#D9DEE7",
                linewidth=0.5,
                zorder=-1,
                clip_on=False,
            )
        )


def compose_figure02(rng: np.random.Generator | None = None) -> Figure:
    """Compose Figure 2 from one reproducible shared synthetic example.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Generator for the predictive-check samples. The canonical seed (42)
        is used when omitted; injection makes alternate examples testable.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Shared example data; the Monte Carlo p-value path uses ``rng``.
    data = create_shared_example(rng)

    # Three metric columns, separated by narrow spacers, over a row of formulas.
    # Each column has its own rows: the outer two have three panels and the middle
    # one has four, the steps of a procedure. Semantic axis names make the visual
    # reading order explicit.
    columns = (
        ("hpd_predictive", "hpd_likelihood", "hpd_overlap"),
        (
            "predictive_distribution",
            "predictive_marks",
            "predictive_scores",
            "predictive_histogram",
        ),
        ("kl_distributions", "kl_log_ratio", "kl_pointwise"),
    )
    formulas = ("hpd_formula", "predictive_formula", "kl_formula")
    fig = plt.figure(
        figsize=(6.2, 6.1),
        dpi=FIGURE_DPI,
        constrained_layout={"h_pad": 0.10, "w_pad": 0.04},
    )
    outer = fig.add_gridspec(2, 5, width_ratios=[2, 0.2, 2, 0.2, 2], height_ratios=[3, 0.27])
    axes: dict[str, Axes] = {}
    for grid_column, names, formula in zip((0, 2, 4), columns, formulas, strict=True):
        rows = outer[0, grid_column].subgridspec(len(names), 1)
        for row, name in enumerate(names):
            axes[name] = fig.add_subplot(rows[row])
        axes[formula] = fig.add_subplot(outer[1, grid_column])

    # KL Divergence column
    plot_kl_distributions(axes["kl_distributions"], data)
    plot_kl_log_ratio(axes["kl_log_ratio"], data)
    plot_kl_pointwise(axes["kl_pointwise"], data)

    # HPD Overlap column
    plot_hpd_predictive(axes["hpd_predictive"], data)
    plot_hpd_likelihood(axes["hpd_likelihood"], data)
    hpd_sizes = plot_hpd_intersection(axes["hpd_overlap"], data)

    # Predictive Check column
    plot_predictive_check_predictive_fan(axes["predictive_distribution"], data)
    plot_predictive_check_replicate_marks(axes["predictive_marks"], data)
    plot_predictive_check_mark_scores(axes["predictive_scores"], data)
    plot_predictive_check_density_histogram(axes["predictive_histogram"], data)

    column_titles = [
        ("hpd_predictive", "HPD overlap"),
        ("predictive_distribution", r"Predictive $\mathbfit{p}$-value"),
        ("kl_distributions", "KL divergence"),
    ]
    for ax_key, col_title in column_titles:
        ax = axes[ax_key]
        ax.text(
            0.5,
            1.18,
            col_title,
            transform=ax.transAxes,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    # KL Divergence formula
    axes["kl_formula"].axis("off")
    axes["kl_formula"].text(
        0.5,
        0.5,
        r"$D_{\mathrm{KL}}(P\,\|\,Q) = \int P \log(P/Q)\,dx_k$"
        f" = {data.kl_divergence:.2f}",
        transform=axes["kl_formula"].transAxes,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # HPD Overlap formula with notation = fraction with numbers = result.
    pred_size, like_size, intersection_size = hpd_sizes
    axes["hpd_formula"].axis("off")
    hpd_formula = (
        r"$\dfrac{|\mathrm{HPD}_P \cap \mathrm{HPD}_Q|}"
        r"{\min(|\mathrm{HPD}_P|, |\mathrm{HPD}_Q|)}$"
        f" = "
        rf"$\dfrac{{{intersection_size:.1f}}}{{{min(pred_size, like_size):.1f}}}$"
        f" = {data.hpd_overlap:.2f}"
    )
    axes["hpd_formula"].text(
        0.5,
        0.5,
        hpd_formula,
        transform=axes["hpd_formula"].transAxes,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Predictive Check formula
    axes["predictive_formula"].axis("off")
    axes["predictive_formula"].text(
        0.5,
        0.5,
        r"$p = \Pr(\log f_{\mathrm{pred}}(y) \geq \log f_{\mathrm{pred}}(\tilde{y}))$"
        f" = {data.predictive_pvalue:.2f}",
        transform=axes["predictive_formula"].transAxes,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Place panel letters a/b/c in the physical reading order.
    for label, ax_key in zip(
        ["a", "b", "c"],
        ["hpd_predictive", "predictive_distribution", "kl_distributions"],
        strict=True,
    ):
        ax = axes[ax_key]
        ax.text(
            -0.15,
            1.18,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # The 8 pt legends/titles need clear space above the data. Expand the
    # y-limits of the distribution panels (the formula and HPD-bar panels are
    # excluded) so the corner-anchored legends do not sit on the curves. The
    # panels showing the tall predictive peak need the most room.
    for key, headroom in (
        ("kl_distributions", 1.36),
        ("hpd_predictive", 1.36),
        ("predictive_distribution", 1.72),
        ("hpd_likelihood", 1.2),
        ("predictive_histogram", 1.28),
    ):
        lo, hi = axes[key].get_ylim()
        axes[key].set_ylim(lo, hi * headroom)

    _add_column_group_backplates(fig, axes)

    return fig


def generate_figure02() -> None:
    """Compose Figure 2 with paper styling and save its PDF and PNG."""
    set_figure_defaults()
    fig = compose_figure02()
    save_figure(FIGURE_DIR / "figure02", close=True, fig=fig)
    print(f"\nFigure 2 saved to {FIGURE_DIR / 'figure02'}.{{pdf,png}}")
