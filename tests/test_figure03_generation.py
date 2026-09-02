"""Tests for the importable Figure-3 generation recipe."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from statespacecheck_paper import figure03_generation
from statespacecheck_paper.diagnostics import DiagnosticThresholds
from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_summary import Figure3RealizationSummary


def test_generation_threads_one_config_through_simulation_summary_and_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The displayed realization and pooled summary share the same config."""
    config = Figure3Config(drift_momentum=0.91)
    simulation_result = SimpleNamespace(
        true_position=np.zeros(5),
        spike_counts=np.zeros((5, 2), dtype=np.int64),
        diagnostics=object(),
        sparse_place_field_centers=np.array([0.5]),
    )
    realization_summary = Figure3RealizationSummary(
        diagnostic_thresholds=DiagnosticThresholds(
            hpd_overlap=0.05,
            kl_divergence=2.0,
            predictive_pvalue=0.05,
        ),
        median_flag_percentages=np.zeros((3, 6)),
        median_decoding_accuracy=np.zeros((1, 6)),
        n_realizations=7,
    )

    seen: dict[str, Any] = {}

    def _simulate(received: Figure3Config) -> SimpleNamespace:
        seen["simulation_config"] = received
        return simulation_result

    monkeypatch.setattr(figure03_generation, "run_figure03_simulation", _simulate)
    monkeypatch.setattr(
        figure03_generation,
        "estimate_realization_summary",
        lambda received, *, n_realizations: (
            seen.update(summary_config=received, n_realizations=n_realizations)
            or realization_summary
        ),
    )
    monkeypatch.setattr(figure03_generation, "set_figure_defaults", lambda **kwargs: None)

    fig = plt.figure()

    def _compose(**kwargs: Any) -> Figure:
        seen["compose_kwargs"] = kwargs
        return fig

    monkeypatch.setattr(figure03_generation, "compose_figure03", _compose)
    monkeypatch.setattr(
        figure03_generation,
        "save_figure",
        lambda *args, **kwargs: seen.update(save_args=args, save_kwargs=kwargs),
    )
    monkeypatch.setattr(
        figure03_generation,
        "write_json_artifact",
        lambda path, payload: (seen.update(summary_path=path, summary_payload=payload) or path),
    )

    figure03_generation.generate_figure03(config, n_realizations=7)
    plt.close(fig)

    assert seen["simulation_config"] is config
    assert seen["summary_config"] is config
    assert seen["n_realizations"] == 7
    assert seen["compose_kwargs"]["config"] is config
    assert seen["save_kwargs"]["fig"] is fig
    assert seen["save_kwargs"]["close"] is True
    assert seen["summary_path"] == figure03_generation.FIGURE03_SUMMARY_PATH
    assert seen["summary_payload"]["realizations"]["count"] == 7


def test_default_recipe_uses_manuscript_drift_momentum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting config selects the documented manuscript configuration."""
    seen: dict[str, Figure3Config] = {}

    def _stop_after_config(config: Figure3Config) -> None:
        seen["config"] = config
        raise RuntimeError("stop after config")

    monkeypatch.setattr(figure03_generation, "run_figure03_simulation", _stop_after_config)
    with pytest.raises(RuntimeError, match="stop after config"):
        figure03_generation.generate_figure03(n_realizations=1)
    assert seen["config"].drift_momentum == pytest.approx(0.88)


def test_summary_payload_preserves_labels_rules_and_source_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = Figure3RealizationSummary(
        diagnostic_thresholds=DiagnosticThresholds(
            hpd_overlap=0.0,
            kl_divergence=2.0,
            predictive_pvalue=0.05,
        ),
        median_flag_percentages=np.zeros((3, 6)),
        median_decoding_accuracy=np.zeros((1, 6)),
        n_realizations=2,
    )
    source = {
        "statespacecheck_paper_version": "test",
        "source_tree_sha256": "a" * 64,
        "uv_lock_sha256": "b" * 64,
    }
    monkeypatch.setattr(figure03_generation, "scientific_source_provenance", lambda: source)

    payload = cast(
        dict[str, Any],
        figure03_generation.figure03_summary_payload(Figure3Config(), summary),
    )
    flag_rules = cast(dict[str, dict[str, str | float]], payload["flag_rules"])

    assert payload["schema_version"] == 3
    assert payload["accuracy_metric_order"] == ["median_absolute_error"]
    assert np.asarray(payload["median_decoding_accuracy"]).shape == (1, 6)
    assert payload["condition_labels"] == [
        "Well-specified",
        "Remap",
        "History-dep.",
        "Replay",
        "Drift",
        "Sparse population",
    ]
    assert flag_rules == {
        "hpd_overlap": {"comparison": "less_than_or_equal", "threshold": 0.0},
        "predictive_pvalue": {"comparison": "less_than_or_equal", "threshold": 0.05},
        "kl_divergence": {"comparison": "greater_than_or_equal", "threshold": 2.0},
    }
    assert payload["provenance"] == {"source": source}
