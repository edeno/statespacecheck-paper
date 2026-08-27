"""Figure-4 generation recipe: config → workflow → summary → layout → save.

A short, readable orchestration of the Figure-4 modules: build the canonical
:class:`Figure4Config` and :class:`Figure4Paths`, prepare the render data
(cached-or-computed decode + fresh track data), print the manuscript summary
scalars, compose the figure, and save it with the composition's tight bounding
box. Uses the fixed diagnostic thresholds (HPD overlap and the predictive
p-value at 0.05; KL has no natural fixed cutoff and is shown without one).
"""

from __future__ import annotations

from typing import Literal

from statespacecheck_paper.figure04_cache import Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_layout import Figure4DetailWindow, compose_figure04
from statespacecheck_paper.figure04_workflow import (
    compute_figure04_summary,
    format_figure04_summary,
    prepare_figure04_render_data,
)
from statespacecheck_paper.paths import ANIMAL_DATE_EPOCH, DATA_PATH
from statespacecheck_paper.style import save_figure, set_figure_defaults

# Diagnostic thresholds. HPD overlap and the predictive p-value use fixed
# cutoffs of 0.05. The KL divergence has no natural fixed cutoff, so it is
# shown without a threshold line or a flagged-region callout.
FIGURE4_DIAGNOSTIC_THRESHOLDS: dict[str, float] = {
    "hpd_overlap": 0.05,
    "predictive_pvalue": 0.05,
}
FIGURE4_METRIC_DIRECTIONS: dict[str, Literal["below", "above"]] = {
    "hpd_overlap": "below",
    "predictive_pvalue": "below",
}
# Manuscript detail view: a KL-divergence spike during immobility at a reward
# well, shown with 500 samples on either side (~2 seconds total at 500 Hz).
FIGURE4_DETAIL_WINDOW = Figure4DetailWindow(
    center_index=193_069,
    half_width_samples=500,
)


def generate_figure04(*, use_cache: bool = True) -> None:
    """Generate Figure 4 (real-data decoder diagnostics) and save it.

    Parameters
    ----------
    use_cache : bool, default True
        When True and a fingerprint-matching cache of decoder outputs exists
        under ``data/intermediates``, load it and skip the expensive fit/decode
        step. When False (``--force-recompute``), always recompute and
        overwrite the cache. A config / data / ``non_local_detector`` change
        invalidates the cache automatically. Fitting + decoding both models
        takes several minutes; figure-only edits (styling, thresholds) reuse
        the cache.
    """
    config = Figure4Config()
    paths = Figure4Paths(data_path=DATA_PATH, animal_date_epoch=ANIMAL_DATE_EPOCH)
    render_data = prepare_figure04_render_data(config, paths, use_cache=use_cache)

    summary = compute_figure04_summary(
        render_data,
        FIGURE4_DIAGNOSTIC_THRESHOLDS,
        FIGURE4_METRIC_DIRECTIONS,
    )
    print(f"\n{format_figure04_summary(summary)}")

    print("\nGenerating Figure 4...")
    set_figure_defaults(context="paper")
    composition = compose_figure04(
        render_data,
        diagnostic_thresholds=FIGURE4_DIAGNOSTIC_THRESHOLDS,
        detail_window=FIGURE4_DETAIL_WINDOW,
    )
    save_figure(
        "manuscript/figures/main/figure04",
        close=True,
        fig=composition.figure,
        bbox_inches=composition.bbox_inches,
    )
    print("Saved manuscript/figures/main/figure04.{pdf,png}")
    print("\nFigure 4 complete!")
