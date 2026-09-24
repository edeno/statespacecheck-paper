"""Tests for the website data export (``statespacecheck_paper.site_export``).

Covers the encoding helpers, the playground and parity-fixture builders, the
Figure-3 scenario payloads (against the real seed-1 realization), the Figure-4
replay payload (against synthetic render data), and that the committed files
under ``site/`` are current with the committed figure summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.diagnostics import (
    compute_normalized_event_likelihood,
    compute_spike_event_diagnostics_from_rates,
)
from statespacecheck_paper.figure03_generation import FIGURE03_CONDITION_IDS
from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_simulation import (
    Figure3SimulationResult,
    build_figure03_rate_tables,
    run_figure03_simulation,
)
from statespacecheck_paper.figure03_summary import build_summary_conditions
from statespacecheck_paper.figure04_diagnostics import mean_per_spike_likelihood_by_time
from statespacecheck_paper.figure04_layout import Figure4DetailWindow
from statespacecheck_paper.reported_values import (
    FIGURE03_SUMMARY_PATH,
    FIGURE04_SUMMARY_PATH,
    macro_sections,
)
from statespacecheck_paper.site_export import (
    PARITY_FIXTURE_PATH,
    SCENARIO_WINDOWS,
    SITE_DATA_DIR,
    decode_display_rows,
    encode_display_rows,
    flag_events,
    gaussian_predictive,
    manifest_payload,
    metric_parity_fixture,
    playground_ensembles,
    playground_payload,
    replay_payload,
    scenario_payloads,
)
from tests.test_figure04_layout import _compose_render_data

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def figure03_summary() -> dict[str, Any]:
    return _load(FIGURE03_SUMMARY_PATH)


@pytest.fixture(scope="module")
def figure04_summary() -> dict[str, Any]:
    return _load(FIGURE04_SUMMARY_PATH)


@pytest.fixture(scope="module")
def simulation() -> Figure3SimulationResult:
    return run_figure03_simulation(Figure3Config())


@pytest.fixture(scope="module")
def scenarios(
    simulation: Figure3SimulationResult, figure03_summary: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return scenario_payloads(simulation, figure03_summary)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def test_display_rows_round_trip_scales_each_row_to_its_max() -> None:
    values = np.array([[0.0, 1.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0], [1e-30, 3e-30, 2e-30, 0.0]])
    decoded = decode_display_rows(encode_display_rows(values), n_bins=4)
    np.testing.assert_array_equal(decoded[0], [0, 64, 128, 255])
    np.testing.assert_array_equal(decoded[1], [0, 0, 0, 0])
    np.testing.assert_array_equal(decoded[2], [85, 255, 170, 0])


@pytest.mark.parametrize(
    "values",
    [np.array([1.0, 2.0]), np.array([[1.0, np.nan]]), np.array([[1.0, -1.0]])],
)
def test_display_rows_reject_invalid_input(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        encode_display_rows(values)


def test_flag_events_is_inclusive_in_both_directions() -> None:
    values = np.array([0.04, 0.05, 0.06])
    below = {"comparison": "less_than_or_equal", "threshold": 0.05}
    above = {"comparison": "greater_than_or_equal", "threshold": 0.05}
    np.testing.assert_array_equal(flag_events(values, below), [True, True, False])
    np.testing.assert_array_equal(flag_events(values, above), [False, True, True])
    with pytest.raises(ValueError, match="Unknown flag comparison"):
        flag_events(values, {"comparison": "equal", "threshold": 0.05})


# ---------------------------------------------------------------------------
# Playground and parity fixture
# ---------------------------------------------------------------------------


def test_gaussian_predictive_is_normalized_and_centered() -> None:
    bins = np.arange(0.0, 101.0)
    predictive = gaussian_predictive(bins, 40.0, 5.0)
    assert predictive.sum() == pytest.approx(1.0)
    assert bins[np.argmax(predictive)] == 40.0
    with pytest.raises(ValueError, match="std must be positive"):
        gaussian_predictive(bins, 40.0, 0.0)


def test_playground_ensembles_are_the_figure3_decoder_tables(
    simulation: Figure3SimulationResult,
) -> None:
    config = Figure3Config()
    assert config.place_field_centers is not None
    sparse_centers = np.asarray(simulation.sparse_place_field_centers)
    position_bins, cell_centers, ensembles = playground_ensembles(config, sparse_centers)
    tables = build_figure03_rate_tables(
        position_bins, config.place_field_centers, sparse_centers, config
    )
    place_cells, sparse_epoch = ensembles
    np.testing.assert_array_equal(place_cells.rates, tables.baseline_firing_rates)
    np.testing.assert_array_equal(sparse_epoch.rates, tables.sparse_population_firing_rates)
    n_place = len(config.place_field_centers)
    assert place_cells.selectable_cells == tuple(range(n_place))
    assert sparse_epoch.selectable_cells == tuple(range(n_place, n_place + sparse_centers.size))
    # Each selectable cell's rate peaks at its field center (to the grid).
    for ensemble in ensembles:
        for cell in ensemble.selectable_cells:
            peak = position_bins[np.argmax(ensemble.rates[:, cell])]
            assert abs(peak - cell_centers[cell]) <= 0.5


def test_playground_payload_carries_simulation_flag_rules(
    simulation: Figure3SimulationResult, figure03_summary: dict[str, Any]
) -> None:
    payload = playground_payload(
        Figure3Config(), np.asarray(simulation.sparse_place_field_centers), figure03_summary
    )
    assert payload["flag_rules"] == figure03_summary["flag_rules"]
    assert [e["ensemble_id"] for e in payload["ensembles"]] == ["place_cells", "sparse_epoch"]
    for ensemble in payload["ensembles"]:
        assert len(ensemble["rates"]) == len(payload["position_bins"])


def test_parity_fixture_matches_the_python_diagnostics(
    simulation: Figure3SimulationResult,
) -> None:
    fixture = metric_parity_fixture(
        Figure3Config(), np.asarray(simulation.sparse_place_field_centers)
    )
    for case in fixture["cases"][::29]:
        diagnostics = compute_spike_event_diagnostics_from_rates(
            np.asarray(fixture["predictives"][case["predictive"]])[np.newaxis, :],
            np.asarray(fixture["ensembles"][case["ensemble"]]),
            np.array([0]),
            np.array([case["cell"]]),
        )
        assert case["expected"]["hpd_overlap"] == diagnostics.event_hpd_overlap[0]
        assert case["expected"]["kl_divergence"] == diagnostics.event_kl_divergence[0]
        assert case["expected"]["predictive_pvalue"] == diagnostics.event_predictive_pvalue[0]
    expected = [case["expected"] for case in fixture["cases"]]
    assert {0.0, 1.0} <= {e["hpd_overlap"] for e in expected}
    # Cases include spikes flagged by KL alone (consistent by the other two).
    assert any(
        e["hpd_overlap"] == 1.0 and e["predictive_pvalue"] > 0.5 and e["kl_divergence"] > 10
        for e in expected
    )


# ---------------------------------------------------------------------------
# Scenario player (Figure 3)
# ---------------------------------------------------------------------------


def test_scenario_windows_cover_every_condition_and_overlap_its_scored_steps() -> None:
    config = Figure3Config()
    conditions = dict(zip(FIGURE03_CONDITION_IDS, build_summary_conditions(config), strict=True))
    assert [window.condition_id for window in SCENARIO_WINDOWS] == list(FIGURE03_CONDITION_IDS)
    n_time = config.phase_boundaries[-1]
    for window in SCENARIO_WINDOWS:
        assert 0 <= window.start < window.stop <= n_time
        assert any(
            t0 < window.stop and t1 > window.start
            for t0, t1 in conditions[window.condition_id].step_windows
        ), window


def test_scenario_events_match_the_decoded_diagnostics(
    simulation: Figure3SimulationResult,
    scenarios: dict[str, dict[str, Any]],
    figure03_summary: dict[str, Any],
) -> None:
    diagnostics = simulation.diagnostics
    n_bins = simulation.position_bins.size
    for window in SCENARIO_WINDOWS:
        payload = scenarios[window.condition_id]
        events = payload["events"]
        in_window = (diagnostics.event_time_ind >= window.start) & (
            diagnostics.event_time_ind < window.stop
        )
        np.testing.assert_array_equal(
            np.asarray(events["t"]) + window.start, diagnostics.event_time_ind[in_window]
        )
        np.testing.assert_array_equal(events["cell"], diagnostics.event_cell_ind[in_window])
        for metric in ("hpd_overlap", "predictive_pvalue", "kl_divergence"):
            full = np.asarray(getattr(diagnostics, f"event_{metric}"))[in_window]
            np.testing.assert_allclose(events[metric], full, rtol=5e-4, atol=0)
            np.testing.assert_array_equal(
                events["flagged"][metric],
                flag_events(full, figure03_summary["flag_rules"][metric]),
            )
        # Each event's likelihood row decodes to its own quantized likelihood.
        rows = decode_display_rows(payload["likelihood_rows"], n_bins)
        expected = decode_display_rows(
            encode_display_rows(diagnostics.per_spike_likelihood[in_window]), n_bins
        )
        np.testing.assert_array_equal(rows[events["likelihood_row"]], expected)
        predictive = decode_display_rows(payload["predictive"]["rows"], n_bins)
        assert predictive.shape == (window.stop - window.start, n_bins)
        # One color scale for every window: Figure 3a's 0 to 97.5th percentile.
        assert payload["predictive"]["range"] == [
            0.0,
            float(np.nanquantile(diagnostics.predictive, 0.975)),
        ]
        np.testing.assert_allclose(
            payload["predictive"]["row_max"],
            diagnostics.predictive[window.start : window.stop].max(axis=1),
            rtol=1e-5,
        )
        assert len(payload["true_position"]) == window.stop - window.start


def test_scenario_summaries_come_from_the_figure_summary(
    scenarios: dict[str, dict[str, Any]], figure03_summary: dict[str, Any]
) -> None:
    metric_order = figure03_summary["metric_order"]
    for column, condition_id in enumerate(figure03_summary["condition_order"]):
        summary = scenarios[condition_id]["summary"]
        for metric, value in summary["median_flag_percent"].items():
            row = metric_order.index(metric)
            assert value == figure03_summary["median_flag_percentages"][row][column]
        assert (
            summary["median_absolute_error"]
            == figure03_summary["median_decoding_accuracy"][0][column]
        )


# ---------------------------------------------------------------------------
# Replay comparison (Figure 4)
# ---------------------------------------------------------------------------


def test_replay_payload_slices_both_models_to_the_detail_window() -> None:
    render_data = _compose_render_data()
    summary = {
        "flag_rules": {
            "hpd_overlap": {"comparison": "less_than_or_equal", "threshold": 0.05},
            "predictive_pvalue": {"comparison": "less_than_or_equal", "threshold": 0.05},
        },
        "provenance": {"figure04_decode_cache": {"fingerprint_sha256": "abc"}},
    }
    window = Figure4DetailWindow(center_index=20, half_width_samples=10)
    payload = replay_payload(render_data, summary, window)
    time_slice = window.to_slice(render_data.time.size)
    n_time = time_slice.stop - time_slice.start
    decode = render_data.decode_results
    n_bins = decode.diagnostic_position_bins.size

    assert len(payload["time"]) == n_time
    assert payload["time"][0] == 0.0
    likelihood, has_spikes = mean_per_spike_likelihood_by_time(
        decode.spike_counts[time_slice], decode.diagnostic_place_fields
    )
    np.testing.assert_array_equal(
        decode_display_rows(payload["likelihood"], n_bins),
        decode_display_rows(encode_display_rows(likelihood), n_bins),
    )
    assert payload["has_spikes"] == has_spikes.tolist()
    np.testing.assert_array_equal(
        decode_display_rows(payload["unit_likelihoods"], n_bins),
        decode_display_rows(
            encode_display_rows(
                compute_normalized_event_likelihood(decode.diagnostic_place_fields)
            ),
            n_bins,
        ),
    )
    for name, diagnostics in (
        ("continuous", decode.continuous_diagnostics),
        ("continuous_fragmented", decode.continuous_fragmented_diagnostics),
    ):
        model = payload["models"][name]
        rows = decode_display_rows(model["predictive"]["rows"], n_bins)
        assert rows.shape == (n_time, n_bins)
        low, high = model["predictive"]["range"]
        assert low < high
        # Events are windowed and located by the decoder bin that counted them.
        in_window = (diagnostics.event_time_ind >= time_slice.start) & (
            diagnostics.event_time_ind < time_slice.stop
        )
        np.testing.assert_array_equal(
            np.asarray(model["events"]["bin"]) + time_slice.start,
            diagnostics.event_time_ind[in_window],
        )
        assert len(model["events"]["t"]) == int(in_window.sum())
        assert set(model["events"]["flagged"]) == {"hpd_overlap", "predictive_pvalue"}
    # Units are ranked by place-field peak.
    assert sorted(payload["unit_rank"]) == list(range(decode.place_field_peaks.size))
    assert payload["decode_cache_fingerprint"] == "abc"


# ---------------------------------------------------------------------------
# Committed site files are current
# ---------------------------------------------------------------------------


def test_committed_manifest_matches_the_figure_summaries(
    figure03_summary: dict[str, Any], figure04_summary: dict[str, Any]
) -> None:
    committed = _load(SITE_DATA_DIR / "manifest.json")
    assert committed == manifest_payload(figure03_summary, figure04_summary)
    macros = {
        macro.name: macro.value
        for _, section in macro_sections(figure03_summary, figure04_summary)
        for macro in section
    }
    assert committed["macros"] == macros


def test_committed_parity_fixture_is_current(simulation: Figure3SimulationResult) -> None:
    committed = _load(PARITY_FIXTURE_PATH)
    fresh = metric_parity_fixture(
        Figure3Config(), np.asarray(simulation.sparse_place_field_centers)
    )
    np.testing.assert_allclose(committed["ensembles"], fresh["ensembles"], rtol=1e-12)
    np.testing.assert_allclose(committed["predictives"], fresh["predictives"], rtol=1e-12)
    assert len(committed["cases"]) == len(fresh["cases"])
    for old, new in zip(committed["cases"], fresh["cases"], strict=True):
        assert (old["predictive"], old["ensemble"], old["cell"]) == (
            new["predictive"],
            new["ensemble"],
            new["cell"],
        )
        for metric, value in new["expected"].items():
            assert old["expected"][metric] == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_committed_scenarios_are_current(scenarios: dict[str, dict[str, Any]]) -> None:
    for condition_id, fresh in scenarios.items():
        committed = _load(SITE_DATA_DIR / f"scenario_{condition_id}.json")
        for key in ("start", "stop", "scored_windows", "summary", "label", "model_component"):
            assert committed[key] == fresh[key], (condition_id, key)
        assert committed["events"]["t"] == fresh["events"]["t"], condition_id
        assert committed["events"]["cell"] == fresh["events"]["cell"], condition_id
        assert committed["events"]["flagged"] == fresh["events"]["flagged"], condition_id


def test_committed_replay_matches_the_figure4_decode(figure04_summary: dict[str, Any]) -> None:
    committed = _load(SITE_DATA_DIR / "replay.json")
    fingerprint = figure04_summary["provenance"]["figure04_decode_cache"]["fingerprint_sha256"]
    assert committed["decode_cache_fingerprint"] == fingerprint
    assert committed["flag_rules"] == figure04_summary["flag_rules"]
    continuous, fragmented = (
        committed["models"][name]["events"] for name in ("continuous", "continuous_fragmented")
    )
    # Both decoders are scored on the same spikes.
    assert continuous["t"] == fragmented["t"]
    assert continuous["cell"] == fragmented["cell"]
