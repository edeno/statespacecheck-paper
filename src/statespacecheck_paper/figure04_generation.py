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

from statespacecheck_paper.diagnostics import (
    FIXED_PREDICTIVE_PVALUE_CUTOFF,
    METRIC_FLAG_DIRECTIONS,
    FlagDirection,
)
from statespacecheck_paper.figure04_cache import Figure4CacheProvenance, Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_layout import Figure4DetailWindow, compose_figure04
from statespacecheck_paper.figure04_models import CONTINUOUS, CONTINUOUS_FRAGMENTED
from statespacecheck_paper.figure04_summary import (
    Figure4Summary,
    compute_figure04_summary,
    format_figure04_summary,
)
from statespacecheck_paper.figure04_workflow import prepare_figure04_render_data
from statespacecheck_paper.paths import (
    ANIMAL_DATE_EPOCH,
    DATA_PATH,
    FIGURE04_SUMMARY_PATH,
    FIGURE_DIR,
)
from statespacecheck_paper.scientific_artifacts import (
    inclusive_flag_rules,
    scientific_source_provenance,
    write_json_artifact,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

# Diagnostic thresholds. HPD overlap and the predictive p-value use fixed
# cutoffs of 0.05: the predictive p-value's is the paper-wide cutoff that
# Figure 3 also uses; the HPD-overlap cutoff is chosen for this recording
# (Figure 3 derives its HPD threshold from a baseline instead). The KL
# divergence has no natural fixed cutoff, so it is shown without a threshold
# line or a flagged-region callout.
FIGURE4_HPD_OVERLAP_CUTOFF = 0.05
FIGURE4_DIAGNOSTIC_THRESHOLDS: dict[str, float] = {
    "hpd_overlap": FIGURE4_HPD_OVERLAP_CUTOFF,
    "predictive_pvalue": FIXED_PREDICTIVE_PVALUE_CUTOFF,
}
FIGURE4_METRIC_DIRECTIONS: dict[str, FlagDirection] = {
    metric: METRIC_FLAG_DIRECTIONS[metric] for metric in FIGURE4_DIAGNOSTIC_THRESHOLDS
}
# Manuscript detail view: a KL-divergence spike during immobility at a reward
# well, shown with 500 samples on either side (~2 seconds total at 500 Hz).
FIGURE4_DETAIL_WINDOW = Figure4DetailWindow(
    center_index=193_069,
    half_width_samples=500,
)


def figure04_reported_statistics(summary: Figure4Summary) -> dict[str, object]:
    """Return the summary's reported statistics as they appear in the summary JSON.

    Parameters
    ----------
    summary : Figure4Summary
        Computed Figure-4 summary.

    Returns
    -------
    dict
        ``diagnostic_means`` (per decoder and metric), ``flag_confusion_models``
        (which decoder is the reference, model A, and which the comparison,
        model B), and ``flag_confusions`` (``rescued`` = flagged by the
        reference only, ``newly_flagged`` = by the comparison only, with
        ``rescue_rate``, ``None`` when undefined).
    """
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
        "diagnostic_means": {
            CONTINUOUS.id: dataclasses.asdict(summary.continuous),
            CONTINUOUS_FRAGMENTED.id: dataclasses.asdict(summary.continuous_fragmented),
        },
        "flag_confusion_models": {
            "reference": CONTINUOUS.id,
            "comparison": CONTINUOUS_FRAGMENTED.id,
        },
        "flag_confusions": confusions,
    }


def figure04_summary_payload(
    *,
    config: Figure4Config,
    paths: Figure4Paths,
    summary: Figure4Summary,
    cache_provenance: Figure4CacheProvenance,
) -> dict[str, object]:
    """Return the canonical Figure 4 reported statistics as JSON-ready data."""
    if cache_provenance.animal_date_epoch != paths.animal_date_epoch:
        raise ValueError(
            "Figure 4 cache provenance identifies a different dataset: "
            f"{cache_provenance.animal_date_epoch!r} != {paths.animal_date_epoch!r}."
        )
    statistics = figure04_reported_statistics(summary)
    return {
        "schema_version": 6,
        "figure": "figure04",
        "dataset": {
            "animal_date_epoch": paths.animal_date_epoch,
            "n_units": summary.n_units,
        },
        "configuration": dataclasses.asdict(config),
        "flag_rules": inclusive_flag_rules(
            FIGURE4_DIAGNOSTIC_THRESHOLDS,
            FIGURE4_METRIC_DIRECTIONS,
        ),
        "detail_window": dataclasses.asdict(FIGURE4_DETAIL_WINDOW),
        "diagnostic_means": statistics["diagnostic_means"],
        "flag_confusion_models": statistics["flag_confusion_models"],
        "flag_confusions": statistics["flag_confusions"],
        "provenance": {
            "source": scientific_source_provenance(),
            "figure04_decode_cache": cache_provenance.artifact_payload(),
        },
    }


def generate_figure04(*, use_cache: bool = True) -> None:
    """Generate Figure 4 (real-data decoder diagnostics) and save it.

    Parameters
    ----------
    use_cache : bool, default True
        When True, load the decode cache and the diagnostics cache from
        ``DATA_PATH / "intermediates"`` (``DATA_PATH`` is ``data/`` unless
        ``STATESPACECHECK_DATA_PATH`` is set) when their fingerprints match,
        and rebuild whichever does not. A stale decode cache (a config, input
        data, fitting implementation, or ``non_local_detector`` change) refits
        and re-decodes both models, which takes several minutes, and then
        recomputes the diagnostics; a stale diagnostics cache alone recomputes
        only the diagnostics from the cached predictions. When False
        (``--force-recompute``), refit, recompute, and overwrite both caches.
        Figure-only edits (styling, flag thresholds) reuse both caches.
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
        figure04_summary_payload(
            config=config,
            paths=paths,
            summary=summary,
            cache_provenance=render_data.cache_provenance,
        ),
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
        FIGURE_DIR / "figure04",
        close=True,
        fig=composition.figure,
        bbox_inches=composition.bbox_inches,
    )
    print(f"Saved {FIGURE_DIR / 'figure04'}.{{pdf,png}}")
    print("\nFigure 4 complete!")
