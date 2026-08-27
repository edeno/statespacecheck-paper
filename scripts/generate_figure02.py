"""Create Figure 2: Diagnostic Metrics for State Space Models.

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

Panels are wired to the fixed axis keys A/D/G/J (KL), B/E/H/K (HPD), and
C/F/I/L (Predictive); the mosaic layout string positions those columns so
the on-page order reads HPD, Predictive, KL.

Per-panel renderers live in :mod:`statespacecheck_paper.figure02_panels`.
This script handles layout, panel labels, formula row, and saving.
"""

from __future__ import annotations

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
    plot_ppc_density_histogram,
    plot_ppc_likelihood_fan,
    plot_ppc_predictive_fan,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults


def _add_column_group_backplates(
    fig: Figure,
    axes: dict[str, Axes],
) -> None:
    """Add subtle column backplates so each metric reads as one group."""
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    column_groups = (
        ("A", "D", "G", "J"),
        ("B", "E", "H", "K"),
        ("C", "F", "I", "L"),
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


def create_figure() -> None:
    """Create Figure 2 with diagnostic metric mechanics."""
    set_figure_defaults(context="paper")
    rng = np.random.default_rng(42)

    # Shared example data; the Monte Carlo p-value path uses ``rng``.
    data = create_shared_example(rng)

    # 3 metric columns × 4 rows (3 content + 1 formula); '.' spacer
    # columns visually separate the metrics without splitting them
    # across subfigures.
    layout = """
        BB.CC.AA
        EE.FF.DD
        HH.II.GG
        KK.LL.JJ
        """
    fig, axes = plt.subplot_mosaic(
        layout,
        figsize=(7.15, 7.0),
        width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1],
        height_ratios=[1, 1, 1, 0.35],
        dpi=450,
        constrained_layout={"h_pad": 0.10, "w_pad": 0.04},
    )

    # KL Divergence column (A, D, G)
    plot_kl_distributions(axes["A"], data)
    plot_kl_log_ratio(axes["D"], data)
    plot_kl_pointwise(axes["G"], data)

    # HPD Overlap column (B, E, H)
    plot_hpd_predictive(axes["B"], data)
    plot_hpd_likelihood(axes["E"], data)
    hpd_sizes = plot_hpd_intersection(axes["H"], data)

    # Predictive Check column (C, F, I)
    plot_ppc_predictive_fan(axes["C"], data)
    plot_ppc_likelihood_fan(axes["F"], data)
    plot_ppc_density_histogram(axes["I"], data)

    column_titles = [("A", "KL Divergence"), ("B", "HPD Overlap"), ("C", "Predictive Check")]
    for ax_key, col_title in column_titles:
        ax = axes[ax_key]
        ax.text(
            0.5,
            1.18,
            col_title,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    # KL Divergence formula
    axes["J"].axis("off")
    axes["J"].text(
        0.5,
        0.5,
        r"$D_{\mathrm{KL}} = \sum \mathrm{pred} \cdot \log(\mathrm{pred}/\mathrm{like})$"
        f" = {data['kl_value']:.2f}",
        transform=axes["J"].transAxes,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # HPD Overlap formula with notation = fraction with numbers = result.
    pred_size, like_size, intersection_size = hpd_sizes
    axes["K"].axis("off")
    hpd_formula = (
        r"$\frac{|H_{\mathrm{pred}} \cap H_{\mathrm{like}}|}"
        r"{\min(|H_{\mathrm{pred}}|, |H_{\mathrm{like}}|)}$"
        f" = "
        rf"$\frac{{{intersection_size:.1f}}}{{{min(pred_size, like_size):.1f}}}$"
        f" = {data['hpd_value']:.2f}"
    )
    axes["K"].text(
        0.5,
        0.5,
        hpd_formula,
        transform=axes["K"].transAxes,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Predictive Check formula
    axes["L"].axis("off")
    axes["L"].text(
        0.5,
        0.5,
        f"$p = P(T^{{rep}} \\leq T^{{obs}})$ = {data['p_value']:.2f}",
        transform=axes["L"].transAxes,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Physical column order (left->right) is HPD (B), Predictive (C), KL (A);
    # place the panel letters a/b/c in that reading order.
    for label, ax_key in zip(["a", "b", "c"], ["B", "C", "A"], strict=True):
        ax = axes[ax_key]
        ax.text(
            -0.15,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    # The 8 pt legends/titles need clear space above the data. Expand the
    # y-limits of the distribution panels (the formula and HPD-bar panels are
    # excluded) so the corner-anchored legends no longer sit on the curves;
    # the data-max y-tick stays put, the extra room opens up above it.
    for key in ("A", "B", "C", "F", "I"):
        lo, hi = axes[key].get_ylim()
        axes[key].set_ylim(lo, hi * 1.28)

    _add_column_group_backplates(fig, axes)

    save_figure("manuscript/figures/main/figure02")
    print("\nFigure 2 saved to manuscript/figures/main/figure02.{pdf,png}")


if __name__ == "__main__":
    create_figure()
