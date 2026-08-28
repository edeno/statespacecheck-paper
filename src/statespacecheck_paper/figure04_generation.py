"""Figure-4 generation recipe: config → workflow → summary → layout → save.

A short, readable orchestration of the Figure-4 modules: build the canonical
:class:`Figure4Config` and :class:`Figure4Paths`, prepare the render data
(cached-or-computed decode + fresh track data), print and serialize the
manuscript summary scalars, compose the figure, and save it with the
composition's tight bounding box. Uses the fixed diagnostic thresholds (HPD
overlap and the predictive
p-value at 0.05; KL has no natural fixed cutoff and is shown without one).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Literal

from statespacecheck_paper.figure04_cache import Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_layout import Figure4DetailWindow, compose_figure04
from statespacecheck_paper.figure04_workflow import (
    Figure4Summary,
    compute_figure04_summary,
    format_figure04_summary,
    prepare_figure04_render_data,
)
from statespacecheck_paper.paths import ANIMAL_DATE_EPOCH, DATA_PATH
from statespacecheck_paper.scientific_artifacts import write_json_artifact
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
FIGURE04_SUMMARY_PATH = Path("manuscript/figures/main/figure04_summary.json")


def figure04_summary_payload(
    *,
    config: Figure4Config,
    paths: Figure4Paths,
    summary: Figure4Summary,
) -> dict[str, object]:
    """Return the canonical Figure 4 reported statistics as JSON-ready data."""
    confusions: list[dict[str, object]] = []
    for confusion in summary.flag_confusions:
        rescue_rate = confusion.rescue_rate
        confusions.append(
            {
                **dataclasses.asdict(confusion),
                "rescue_rate": rescue_rate if math.isfinite(rescue_rate) else None,
            }
        )
    return {
        "schema_version": 1,
        "figure": "figure04",
        "dataset": {"animal_date_epoch": paths.animal_date_epoch},
        "configuration": dataclasses.asdict(config),
        "diagnostic_thresholds": dict(FIGURE4_DIAGNOSTIC_THRESHOLDS),
        "metric_flag_directions": dict(FIGURE4_METRIC_DIRECTIONS),
        "detail_window": dataclasses.asdict(FIGURE4_DETAIL_WINDOW),
        "diagnostic_means": {
            "continuous": dataclasses.asdict(summary.continuous),
            "continuous_fragmented": dataclasses.asdict(summary.continuous_fragmented),
        },
        "flag_confusions": confusions,
    }


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
    summary_path = write_json_artifact(
        FIGURE04_SUMMARY_PATH,
        figure04_summary_payload(config=config, paths=paths, summary=summary),
    )
    print(f"Saved canonical statistics to {summary_path}")

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
