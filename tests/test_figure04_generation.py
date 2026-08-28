"""Tests for the Figure-4 generation recipe and the thin CLI script."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from statespacecheck_paper import figure04_generation
from statespacecheck_paper.figure04_cache import Figure4CacheProvenance, Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_diagnostics import FlagConfusion
from statespacecheck_paper.figure04_layout import Figure4Composition
from statespacecheck_paper.figure04_workflow import (
    Figure4DiagnosticMeans,
    Figure4Summary,
)
from statespacecheck_paper.load_local_data import EXPORT_FILE_SUFFIXES

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def figure04_script() -> Iterator[ModuleType]:
    """Import the thin ``scripts/generate_figure04.py`` CLI module."""
    added = str(_SCRIPTS_DIR) not in sys.path
    if added:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        yield importlib.import_module("generate_figure04")
    finally:
        if added:
            sys.path.remove(str(_SCRIPTS_DIR))


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
        lambda *a, **k: SimpleNamespace(cache_provenance=provenance_token),
    )
    monkeypatch.setattr(figure04_generation, "compute_figure04_summary", lambda *a, **k: object())
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
    assert composed["kwargs"]["detail_window"] is figure04_generation.FIGURE4_DETAIL_WINDOW


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
        a_only=18,
        b_only=3,
        neither=77,
    )
    summary = Figure4Summary(means_a, means_b, (confusion,))
    cache_provenance = Figure4CacheProvenance(
        fingerprint_sha256="c" * 64,
        schema_version=4,
        animal_date_epoch="epoch_x",
        export_checksums=tuple((suffix, "d" * 64) for suffix in EXPORT_FILE_SUFFIXES),
        non_local_detector_version="1.2.3",
    )
    source = {
        "statespacecheck_paper_version": "test",
        "source_tree_sha256": "a" * 64,
        "uv_lock_sha256": "b" * 64,
    }
    monkeypatch.setattr(figure04_generation, "scientific_source_provenance", lambda: source)
    payload = cast(
        dict[str, Any],
        figure04_generation.figure04_summary_payload(
            config=Figure4Config(),
            paths=Figure4Paths(tmp_path, "epoch_x"),
            summary=summary,
            cache_provenance=cache_provenance,
        ),
    )
    flag_rules = cast(dict[str, dict[str, str | float]], payload["flag_rules"])
    provenance = cast(dict[str, Any], payload["provenance"])

    assert payload["schema_version"] == 2
    assert payload["dataset"] == {"animal_date_epoch": "epoch_x"}
    assert flag_rules["hpd_overlap"] == {
        "comparison": "less_than_or_equal",
        "threshold": 0.05,
    }
    assert payload["diagnostic_means"]["continuous"]["hpd_overlap"] == pytest.approx(0.1)
    assert payload["flag_confusions"][0] == {
        "metric": "hpd_overlap",
        "threshold": 0.05,
        "n": 100,
        "both": 2,
        "a_only": 18,
        "b_only": 3,
        "neither": 77,
        "rescue_rate": 0.9,
    }
    assert provenance["source"] == source
    decode_provenance = provenance["figure04_decode_cache"]
    assert decode_provenance["fingerprint_sha256"] == "c" * 64
    assert set(decode_provenance["export_file_sha256"]) == {
        f"epoch_x{suffix}" for suffix in EXPORT_FILE_SUFFIXES
    }


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
