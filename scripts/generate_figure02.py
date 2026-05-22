"""Create Figure 2: Diagnostic Metrics for State Space Models.

This figure explains the three diagnostic metrics (KL divergence, HPD overlap,
and predictive checks) using a shared synthetic example.

Layout (3 columns x 4 rows):
    Row 1: Input distributions for each metric
    Row 2: Intermediate computation
    Row 3: Final result
    Row 4: Formula with computed value

Columns (single panel label per column):
    a = KL Divergence mechanics
    b = HPD Overlap mechanics
    c = Predictive Check mechanics

Per-panel renderers live in :mod:`statespacecheck_paper.figure02_panels`.
This script handles layout, panel labels, formula row, and saving.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from statespacecheck_paper.figure02_panels import (
    create_shared_example,
    plot_hpd_panel_d,
    plot_hpd_panel_e,
    plot_hpd_panel_f,
    plot_kl_panel_a,
    plot_kl_panel_b,
    plot_kl_panel_c,
    plot_ppc_panel_g,
    plot_ppc_panel_h,
    plot_ppc_panel_i,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults


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
        AA.BB.CC
        DD.EE.FF
        GG.HH.II
        JJ.KK.LL
        """
    fig, axes = plt.subplot_mosaic(
        layout,
        figsize=(7.0, 5.5),
        width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1],
        height_ratios=[1, 1, 1, 0.35],
        dpi=450,
        constrained_layout={"h_pad": 0.08, "w_pad": 0.04},
    )

    # KL Divergence column (A, D, G)
    plot_kl_panel_a(axes["A"], data)
    plot_kl_panel_b(axes["D"], data)
    plot_kl_panel_c(axes["G"], data)

    # HPD Overlap column (B, E, H)
    plot_hpd_panel_d(axes["B"], data)
    plot_hpd_panel_e(axes["E"], data)
    hpd_sizes = plot_hpd_panel_f(axes["H"], data)

    # Predictive Check column (C, F, I)
    plot_ppc_panel_g(axes["C"], data)
    plot_ppc_panel_h(axes["F"], data)
    plot_ppc_panel_i(axes["I"], data)

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

    for label, ax_key in zip(["a", "b", "c"], ["A", "B", "C"], strict=True):
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

    save_figure("manuscript/figures/main/figure02")
    print("\nFigure 2 saved to manuscript/figures/main/figure02.{pdf,png}")


if __name__ == "__main__":
    create_figure()
