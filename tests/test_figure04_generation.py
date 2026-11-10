"""Tests for the Figure-4 generation recipe and the thin CLI script."""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from statespacecheck_paper import figure04_generation
from statespacecheck_paper.figure04_cache import (
    Figure4Paths,
)
from statespacecheck_paper.figure04_decoder import Figure4Config, Figure4PackageDefaults
from statespacecheck_paper.figure04_diagnostics import FlagConfusion
from statespacecheck_paper.figure04_input import INPUT_FILE_SUFFIX
from statespacecheck_paper.figure04_layout import Figure4Composition
from statespacecheck_paper.figure04_protocol import FIGURE04_DETAIL_WINDOW
from statespacecheck_paper.figure04_summary import (
    Figure4DetailWindowComparison,
    Figure4DiagnosticMeans,
    Figure4ImmobilityComparison,
    Figure4KlDivergenceComparison,
    Figure4ReportedStatistics,
    Figure4Summary,
)
from statespacecheck_paper.paths import FIGURE04_SUMMARY_PATH, REPO_ROOT

from ._figure04 import synthetic_cache_provenance
from ._scripts import SCRIPTS_DIR


@pytest.fixture(scope="module")
def figure04_script() -> Iterator[ModuleType]:
    """Import the thin ``scripts/generate_figure04.py`` CLI module."""
    added = str(SCRIPTS_DIR) not in sys.path
    if added:
        sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        yield importlib.import_module("generate_figure04")
    finally:
        if added:
            sys.path.remove(str(SCRIPTS_DIR))


def test_generation_passes_figure_and_bbox_to_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """``generate_figure04`` must hand the composed figure and its tight bbox to
    ``save_figure`` explicitly (preserving the custom crop)."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    fig = plt.figure()
    composition = Figure4Composition(figure=fig, bbox_inches=Bbox.from_bounds(0, 0, 1, 1))

    provenance_token = object()
    monkeypatch.setattr(
        figure04_generation,
        "prepare_figure04_render_data",
        lambda *a, **k: SimpleNamespace(
            cache_provenance=provenance_token,
            analysis_results=SimpleNamespace(
                continuous_diagnostics=None, continuous_fragmented_diagnostics=None
            ),
            time=np.zeros(200_000),
        ),
    )
    monkeypatch.setattr(
        figure04_generation,
        "compute_figure04_reported_statistics",
        lambda *a, **k: SimpleNamespace(summary=object()),
    )
    monkeypatch.setattr(figure04_generation, "format_figure04_summary", lambda *a, **k: "summary")
    monkeypatch.setattr(
        figure04_generation,
        "figure04_summary_payload",
        lambda **kwargs: {"figure": "figure04"},
    )
    monkeypatch.setattr(
        figure04_generation,
        "write_json_artifact",
        lambda path, payload: path,
    )
    composed: dict[str, Any] = {}

    def _compose(*args: Any, **kwargs: Any) -> Figure4Composition:
        composed.update(args=args, kwargs=kwargs)
        return composition

    monkeypatch.setattr(figure04_generation, "compose_figure04", _compose)
    monkeypatch.setattr(figure04_generation, "set_figure_defaults", lambda *a, **k: None)

    saved: dict[str, Any] = {}
    monkeypatch.setattr(
        figure04_generation, "save_figure", lambda *a, **k: saved.update(args=a, kwargs=k)
    )

    figure04_generation.generate_figure04(use_cache=True)
    plt.close(fig)

    assert saved["kwargs"]["fig"] is fig
    assert saved["kwargs"]["bbox_inches"] is composition.bbox_inches
    assert saved["kwargs"]["close"] is True
    assert composed["kwargs"]["detail_window"] is FIGURE04_DETAIL_WINDOW


def test_summary_payload_contains_reported_counts_rates_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The JSON sidecar carries the values copied into the manuscript."""
    means_a = Figure4DiagnosticMeans(0.1, 2.0, 0.03)
    means_b = Figure4DiagnosticMeans(0.2, 1.0, 0.08)
    confusion = FlagConfusion(
        metric="hpd_overlap",
        threshold=0.05,
        n=100,
        both=2,
        rescued=18,
        newly_flagged=3,
        neither=77,
    )
    summary = Figure4Summary(means_a, means_b, (confusion,), n_units=7)
    kl_comparison = Figure4KlDivergenceComparison(0.9, 3.0, 2.0, 0.08, 1.5, 2.5, 30.0)
    detail_comparison = Figure4DetailWindowComparison(
        n_events=50,
        hpd_overlap=FlagConfusion("hpd_overlap", 0.05, 50, 1, 4, 0, 45),
        predictive_pvalue=FlagConfusion("predictive_pvalue", 0.05, 50, 2, 3, 1, 44),
        predictive_pvalue_higher=30,
        continuous_kl_median=3.0,
        continuous_fragmented_kl_median=2.0,
        median_speed_cm_per_s=0.5,
        fragmented_share_median=0.05,
        fragmented_share_peak=0.9,
        fragmented_probability_threshold=0.8,
        fragmented_share_above_threshold_samples=12,
    )
    immobility_comparison = Figure4ImmobilityComparison(
        speed_threshold_cm_per_s=4.0,
        immobile_flag_rates={"hpd_overlap": 0.04},
        moving_flag_rates={"hpd_overlap": 0.01},
        immobile_kl_median=3.5,
        moving_kl_median=2.9,
    )
    cache_provenance = synthetic_cache_provenance("epoch_x")
    source = {
        "statespacecheck_paper_version": "test",
        "statespacecheck_version": "test",
        "source_tree_sha256": "a" * 64,
        "uv_lock_sha256": "b" * 64,
    }
    monkeypatch.setattr(figure04_generation, "scientific_source_provenance", lambda: source)
    payload = cast(
        dict[str, Any],
        figure04_generation.figure04_summary_payload(
            config=Figure4Config(),
            paths=Figure4Paths(tmp_path, "epoch_x"),
            statistics=Figure4ReportedStatistics(
                summary=summary,
                n_units_with_spikes=5,
                epoch_duration_s=12.5,
                kl_divergence_comparison=kl_comparison,
                detail_window_comparison=detail_comparison,
                immobility_comparison=immobility_comparison,
                hpd_flagged_nonzero_overlap=2,
            ),
            cache_provenance=cache_provenance,
        ),
    )
    flag_rules = cast(dict[str, dict[str, str | float]], payload["flag_rules"])
    provenance = cast(dict[str, Any], payload["provenance"])

    assert payload["schema_version"] == figure04_generation.FIGURE04_SUMMARY_SCHEMA_VERSION
    # The recorded package defaults include the per-state classes the decode relies on.
    package_defaults = payload["configuration"]["package_defaults"]
    assert package_defaults["continuous_fragmented_transition_types"] == (
        ("RandomWalk", "Uniform"),
        ("Uniform", "Uniform"),
    )
    assert package_defaults["continuous_initial_conditions_types"] == ("UniformInitialConditions",)
    assert payload["dataset"] == {
        "animal_date_epoch": "epoch_x",
        "n_units": 7,
        "n_units_with_spikes": 5,
        "duration_s": 12.5,
    }
    assert flag_rules["hpd_overlap"] == {
        "comparison": "less_than_or_equal",
        "threshold": 0.05,
    }
    assert payload["diagnostic_means"]["continuous"]["hpd_overlap"] == pytest.approx(0.1)
    # Each flag confusion names its reference and comparison decoders.
    assert payload["flag_confusion_models"] == {
        "reference": "continuous",
        "comparison": "continuous_fragmented",
    }
    assert payload["flag_confusions"][0] == {
        "metric": "hpd_overlap",
        "threshold": 0.05,
        "n": 100,
        "both": 2,
        "rescued": 18,
        "newly_flagged": 3,
        "neither": 77,
        "rescued_fraction": 0.9,
    }
    assert payload["kl_divergence_comparison"] == {
        "lower_fraction": 0.9,
        "lower_continuous_median": 3.0,
        "lower_continuous_fragmented_median": 2.0,
        "higher_fraction": 0.08,
        "higher_continuous_median": 1.5,
        "higher_continuous_fragmented_median": 2.5,
        "higher_continuous_fragmented_max": 30.0,
    }
    assert payload["detail_window_comparison"] == {
        "n_events": 50,
        "hpd_overlap": {
            "metric": "hpd_overlap",
            "threshold": 0.05,
            "n": 50,
            "both": 1,
            "rescued": 4,
            "newly_flagged": 0,
            "neither": 45,
        },
        "predictive_pvalue": {
            "metric": "predictive_pvalue",
            "threshold": 0.05,
            "n": 50,
            "both": 2,
            "rescued": 3,
            "newly_flagged": 1,
            "neither": 44,
        },
        "predictive_pvalue_higher": 30,
        "continuous_kl_median": 3.0,
        "continuous_fragmented_kl_median": 2.0,
        "median_speed_cm_per_s": 0.5,
        "fragmented_share_median": 0.05,
        "fragmented_share_peak": 0.9,
        "fragmented_probability_threshold": 0.8,
        "fragmented_share_above_threshold_samples": 12,
    }
    assert payload["immobility_comparison"] == {
        "speed_threshold_cm_per_s": 4.0,
        "immobile_flag_rates": {"hpd_overlap": 0.04},
        "moving_flag_rates": {"hpd_overlap": 0.01},
        "immobile_kl_median": 3.5,
        "moving_kl_median": 2.9,
    }
    assert payload["hpd_flagged_nonzero_overlap"] == 2
    assert provenance["source"] == source
    cache_provenance_payload = provenance["figure04_caches"]
    assert cache_provenance_payload["fingerprint_sha256"] == "c" * 64
    assert cache_provenance_payload["diagnostics_fingerprint_sha256"] == "e" * 64
    assert cache_provenance_payload["diagnostics_config"]["hpd_coverage"] == 0.95
    assert cache_provenance_payload["input_file_sha256"] == {
        f"epoch_x{INPUT_FILE_SUFFIX}": "d" * 64
    }


def test_committed_summary_records_the_current_schema_and_package_defaults() -> None:
    """The committed summary uses the current layout and records exactly the
    package defaults the code checks the decoders against."""
    committed = json.loads((REPO_ROOT / FIGURE04_SUMMARY_PATH).read_text(encoding="utf-8"))
    assert committed["schema_version"] == figure04_generation.FIGURE04_SUMMARY_SCHEMA_VERSION
    expected = json.loads(json.dumps(dataclasses.asdict(Figure4PackageDefaults())))
    assert committed["configuration"]["package_defaults"] == expected


def test_committed_summary_states_the_nld_version_it_was_decoded_with() -> None:
    """The version the summary's package defaults record is the installed version
    the decode cache recorded."""
    committed = json.loads((REPO_ROOT / FIGURE04_SUMMARY_PATH).read_text(encoding="utf-8"))
    stated = committed["configuration"]["package_defaults"]["non_local_detector_version"]
    decoded_with = committed["provenance"]["figure04_caches"]["non_local_detector_version"]
    assert stated == decoded_with


def test_cli_force_recompute_forwards_use_cache(
    figure04_script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--force-recompute`` maps to ``use_cache=False``; its absence to True."""
    calls: list[bool] = []
    monkeypatch.setattr(
        figure04_script, "generate_figure04", lambda *, use_cache: calls.append(use_cache)
    )

    figure04_script.main(["--force-recompute"])
    figure04_script.main([])
    assert calls == [False, True]
