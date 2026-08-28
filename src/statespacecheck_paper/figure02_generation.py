"""Figure-2 composition and generation recipe.

This figure explains the three diagnostic metrics (HPD overlap, predictive
checks, and KL divergence) using a shared synthetic example.

Layout (3 columns x 4 rows):
    Row 1: Input distributions for each metric
    Row 2: Intermediate computation
    Row 3: Final result
    Row 4: Formula with computed value

Columns (single panel label per column, left to right):
    a = HPD Overlap mechanics
    b = Predictive Check mechanics
    c = KL Divergence mechanics

The subplot mosaic uses semantic axis names (for example,
``hpd_predictive`` and ``kl_pointwise``), so the on-page order and each
renderer's scientific role are visible directly in the layout recipe.

Per-panel renderers live in :mod:`statespacecheck_paper.figure02_panels`.
This module handles layout, panel labels, formula row, and saving.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from matplotlib.typing import HashableList

from statespacecheck_paper.figure02_panels import (
    create_shared_example,
    plot_hpd_intersection,
    plot_hpd_likelihood,
    plot_hpd_predictive,
    plot_kl_distributions,
    plot_kl_log_ratio,
    plot_kl_pointwise,
    plot_ppc_density_histogram,
    plot_ppc_likelihood_fan,
    plot_ppc_predictive_fan,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults


def _add_column_group_backplates(
    fig: Figure,
    axes: Mapping[Hashable, Axes],
) -> None:
    """Add subtle column backplates so each metric reads as one group."""
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    column_groups = (
        ("hpd_predictive", "hpd_likelihood", "hpd_overlap", "hpd_formula"),
        (
            "predictive_distribution",
            "predictive_simulations",
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

    # 3 metric columns × 4 rows (3 content + 1 formula); '.' spacer
    # columns visually separate the metrics without splitting them
    # across subfigures.
    # Semantic axis names make the visual reading order explicit. Each name is
    # repeated across two mosaic columns; ``.`` is the narrow inter-metric spacer.
    layout: list[list[Hashable]] = [
        [
            "hpd_predictive",
            "hpd_predictive",
            ".",
            "predictive_distribution",
            "predictive_distribution",
            ".",
            "kl_distributions",
            "kl_distributions",
        ],
        [
            "hpd_likelihood",
            "hpd_likelihood",
            ".",
            "predictive_simulations",
            "predictive_simulations",
            ".",
            "kl_log_ratio",
            "kl_log_ratio",
        ],
        [
            "hpd_overlap",
            "hpd_overlap",
            ".",
            "predictive_histogram",
            "predictive_histogram",
            ".",
            "kl_pointwise",
            "kl_pointwise",
        ],
        [
            "hpd_formula",
            "hpd_formula",
            ".",
            "predictive_formula",
            "predictive_formula",
            ".",
            "kl_formula",
            "kl_formula",
        ],
    ]
    fig, axes = plt.subplot_mosaic(
        # matplotlib's stub types the mosaic as ``list[HashableList[Hashable]]``;
        # a plain nested string list is the documented form but needs a cast to
        # satisfy list invariance.
        cast("list[HashableList[Hashable]]", layout),
        figsize=(7.15, 7.0),
        width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1],
        height_ratios=[1, 1, 1, 0.35],
        dpi=450,
        constrained_layout={"h_pad": 0.10, "w_pad": 0.04},
    )

    # KL Divergence column
    plot_kl_distributions(axes["kl_distributions"], data)
    plot_kl_log_ratio(axes["kl_log_ratio"], data)
    plot_kl_pointwise(axes["kl_pointwise"], data)

    # HPD Overlap column
    plot_hpd_predictive(axes["hpd_predictive"], data)
    plot_hpd_likelihood(axes["hpd_likelihood"], data)
    hpd_sizes = plot_hpd_intersection(axes["hpd_overlap"], data)

    # Predictive Check column
    plot_ppc_predictive_fan(axes["predictive_distribution"], data)
    plot_ppc_likelihood_fan(axes["predictive_simulations"], data)
    plot_ppc_density_histogram(axes["predictive_histogram"], data)

    column_titles = [
        ("hpd_predictive", "HPD Overlap"),
        ("predictive_distribution", "Predictive Check"),
        ("kl_distributions", "KL Divergence"),
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
        r"$D_{\mathrm{KL}} = \sum \mathrm{pred} \cdot \log(\mathrm{pred}/\mathrm{like})$"
        f" = {data.kl_value:.2f}",
        transform=axes["kl_formula"].transAxes,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # HPD Overlap formula with notation = fraction with numbers = result.
    pred_size, like_size, intersection_size = hpd_sizes
    axes["hpd_formula"].axis("off")
    hpd_formula = (
        r"$\frac{|H_{\mathrm{pred}} \cap H_{\mathrm{like}}|}"
        r"{\min(|H_{\mathrm{pred}}|, |H_{\mathrm{like}}|)}$"
        f" = "
        rf"$\frac{{{intersection_size:.1f}}}{{{min(pred_size, like_size):.1f}}}$"
        f" = {data.hpd_value:.2f}"
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
        f"$p = P(T^{{rep}} \\leq T^{{obs}})$ = {data.p_value:.2f}",
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
            1.08,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # The 8 pt legends/titles need clear space above the data. Expand the
    # y-limits of the distribution panels (the formula and HPD-bar panels are
    # excluded) so the corner-anchored legends no longer sit on the curves;
    # the data-max y-tick stays put, the extra room opens up above it.
    for key in (
        "kl_distributions",
        "hpd_predictive",
        "predictive_distribution",
        "predictive_simulations",
        "predictive_histogram",
    ):
        lo, hi = axes[key].get_ylim()
        axes[key].set_ylim(lo, hi * 1.28)

    _add_column_group_backplates(fig, axes)

    return fig


def generate_figure02() -> None:
    """Compose Figure 2 with paper styling and save its PDF and PNG."""
    set_figure_defaults(context="paper")
    fig = compose_figure02()
    save_figure("manuscript/figures/main/figure02", close=True, fig=fig)
    print("\nFigure 2 saved to manuscript/figures/main/figure02.{pdf,png}")
