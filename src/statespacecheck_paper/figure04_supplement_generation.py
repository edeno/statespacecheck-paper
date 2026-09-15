"""Figure-4 supplement recipe: cached decode → broadening/rate analyses → figure + JSON.

Runs the full-recording follow-up analyses of
:mod:`statespacecheck_paper.figure04_broadening` on the same cached decode and
diagnostics the canonical Figure 4 uses, writes their machine-readable summary
(``figure04_supplement_summary.json``), and renders the supplementary figure.
The analyses change only the diagnostics' inputs or thresholds; no decoder is
refitted.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from statespacecheck_paper.figure04_broadening import (
    Figure4BroadeningConfig,
    compute_rate_and_behavior_association,
    compute_region_size_and_broadening,
)
from statespacecheck_paper.figure04_cache import Figure4CacheProvenance, Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_supplement_plotting import compose_figure04_supplement
from statespacecheck_paper.figure04_workflow import Figure4RenderData, prepare_figure04_render_data
from statespacecheck_paper.paths import ANIMAL_DATE_EPOCH, DATA_PATH
from statespacecheck_paper.scientific_artifacts import (
    scientific_source_provenance,
    write_json_artifact,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

FIGURE04_SUPPLEMENT_SUMMARY_PATH = Path(
    "manuscript/figures/supplementary/figure04_supplement_summary.json"
)
FIGURE04_SUPPLEMENT_FIGURE_PATH = "manuscript/figures/supplementary/figure04_supplement"


def figure04_supplement_summary_payload(
    *,
    config: Figure4Config,
    analysis_config: Figure4BroadeningConfig,
    paths: Figure4Paths,
    broadening_summary: dict[str, object],
    rate_summary: dict[str, object],
    cache_provenance: Figure4CacheProvenance,
) -> dict[str, object]:
    """Return the canonical Figure-4 supplement statistics as JSON-ready data."""
    if cache_provenance.animal_date_epoch != paths.animal_date_epoch:
        raise ValueError(
            "Figure 4 cache provenance identifies a different dataset: "
            f"{cache_provenance.animal_date_epoch!r} != {paths.animal_date_epoch!r}."
        )
    return {
        "schema_version": 1,
        "figure": "figure04_supplement",
        "dataset": {"animal_date_epoch": paths.animal_date_epoch},
        "configuration": dataclasses.asdict(config),
        "analysis_configuration": dataclasses.asdict(analysis_config),
        "region_size_and_broadening": broadening_summary,
        "rate_and_behavior": rate_summary,
        "provenance": {
            "source": scientific_source_provenance(),
            "figure04_decode_cache": cache_provenance.artifact_payload(),
        },
    }


def generate_figure04_supplement(
    *,
    render_data: Figure4RenderData | None = None,
    analysis_config: Figure4BroadeningConfig | None = None,
) -> None:
    """Compute the Figure-4 supplement analyses, save their JSON and the figure.

    Parameters
    ----------
    render_data : Figure4RenderData, optional
        Preloaded canonical render data; loaded from the caches when omitted.
    analysis_config : Figure4BroadeningConfig, optional
        Analysis settings; the defaults are the manuscript settings.
    """
    config = Figure4Config()
    paths = Figure4Paths(data_path=DATA_PATH, animal_date_epoch=ANIMAL_DATE_EPOCH)
    if analysis_config is None:
        analysis_config = Figure4BroadeningConfig()
    if render_data is None:
        render_data = prepare_figure04_render_data(config, paths, use_cache=True)
    if render_data.cache_provenance is None:
        raise RuntimeError("Figure 4 render data lacks cache provenance; refusing to write.")

    print("Computing predictive-region sizes, uniform-mixture rescue, coverage sensitivity...")
    broadening = compute_region_size_and_broadening(render_data, analysis_config)
    print("Computing firing-rate and behavior associations...")
    rate_summary = compute_rate_and_behavior_association(render_data, analysis_config)

    summary_path = write_json_artifact(
        FIGURE04_SUPPLEMENT_SUMMARY_PATH,
        figure04_supplement_summary_payload(
            config=config,
            analysis_config=analysis_config,
            paths=paths,
            broadening_summary=broadening.summary,
            rate_summary=rate_summary,
            cache_provenance=render_data.cache_provenance,
        ),
    )
    print(f"Saved canonical statistics to {summary_path}")

    set_figure_defaults(context="paper")
    fig = compose_figure04_supplement(broadening, rate_summary, analysis_config)
    save_figure(FIGURE04_SUPPLEMENT_FIGURE_PATH, close=True, fig=fig)
    print(f"Saved {FIGURE04_SUPPLEMENT_FIGURE_PATH}.{{pdf,png}}")
