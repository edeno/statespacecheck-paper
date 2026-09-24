"""Tests for the website data export (``statespacecheck_paper.site_export``).

Covers the encoding helpers, the filter explainer, the playground and
parity-fixture builders, the Figure-3 scenario payloads (against the real seed-1
realization), the Figure-4 replay payload (against synthetic render data), and
that the committed files under ``site/`` are current with the committed figure
summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.decoding import decode_with_diagnostics
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
from statespacecheck_paper.simulation import gaussian_transition_matrix, place_field_rates
from statespacecheck_paper.site_export import (
    FILTER_EXPLAINER,
    PARITY_FIXTURE_PATH,
    SCENARIO_WINDOWS,
    SITE_DATA_DIR,
    FilterExplainerSequence,
    decode_display_rows,
    encode_display_rows,
    filter_explainer_payload,
    filter_explainer_sequence,
    flag_events,
    gaussian_predictive,
    manifest_payload,
    metric_parity_fixture,
    playground_ensembles,
    playground_payload,
    replay_payload,
    scenario_payloads,
)
from statespacecheck_paper.style import COLORS, METRIC_SPECS
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
# Filter explainer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def explainer_sequence() -> FilterExplainerSequence:
    return filter_explainer_sequence(Figure3Config())


def test_filter_explainer_is_decoded_by_the_papers_filter(
    explainer_sequence: FilterExplainerSequence,
) -> None:
    config = Figure3Config()
    assert config.place_field_centers is not None
    sequence = explainer_sequence
    centers = np.asarray(config.place_field_centers)
    rates = place_field_rates(
        sequence.position_bins, centers, config.place_field_std, FILTER_EXPLAINER.rate_scale
    )
    np.testing.assert_array_equal(sequence.rates, rates)
    decoded = decode_with_diagnostics(
        sequence.spike_counts,
        sequence.position_bins,
        gaussian_transition_matrix(sequence.position_bins, FILTER_EXPLAINER.step_std),
        centers,
        config.place_field_std,
        FILTER_EXPLAINER.rate_scale,
    )
    np.testing.assert_array_equal(sequence.decoded.predictive, decoded.predictive)
    np.testing.assert_array_equal(sequence.decoded.posterior, decoded.posterior)

    payload = filter_explainer_payload(config)
    n_bins = sequence.position_bins.size
    for key in ("predictive", "posterior"):
        np.testing.assert_array_equal(
            decode_display_rows(payload[key]["rows"], n_bins),
            decode_display_rows(encode_display_rows(getattr(decoded, key)), n_bins),
        )
    # Both rows share one scale, so the page can compare their heights.
    assert payload["predictive"]["range"] == payload["posterior"]["range"]
    # The fields ship with each cell's peak, so the page draws their true heights.
    fields = payload["place_fields"]
    np.testing.assert_array_equal(
        decode_display_rows(fields["rows"], n_bins),
        decode_display_rows(encode_display_rows(sequence.rates.T), n_bins),
    )
    np.testing.assert_allclose(fields["row_max"], sequence.rates.max(axis=0), rtol=1e-5)
    assert fields["range"] == [0.0, sequence.rates.max()]
    assert payload["events"]["t"] == decoded.event_time_ind.tolist()
    assert payload["cell_centers"] == sorted(payload["cell_centers"])


def test_filter_explainer_sequence_tells_the_story_on_the_page(
    explainer_sequence: FilterExplainerSequence,
) -> None:
    """The page text describes this sequence: check the claims it makes."""
    sequence = explainer_sequence
    counts = sequence.spike_counts
    decoded = sequence.decoded
    centers = np.asarray(Figure3Config().place_field_centers)
    # It opens on one spike, whose prediction is the flat initial distribution.
    assert counts[0].sum() == 1
    np.testing.assert_allclose(decoded.predictive[0], 1.0 / sequence.position_bins.size)
    # The animal runs smoothly up the track and back.
    position = sequence.true_position
    assert position[0] == pytest.approx(FILTER_EXPLAINER.run_low)
    assert position.max() == pytest.approx(FILTER_EXPLAINER.run_high, abs=1e-3)
    assert np.abs(np.diff(position, n=2)).max() < 0.1
    # The prediction visibly spreads over at least one long gap between spikes.
    conflict = FILTER_EXPLAINER.conflict_step
    spike_steps = np.flatnonzero(counts.sum(axis=1))
    assert np.diff(spike_steps[spike_steps <= conflict]).max() >= 30
    # Before the conflict, although the decoder ignores the animal's momentum,
    # no spike is inconsistent with the prediction.
    assert np.all(decoded.event_hpd_overlap[decoded.event_time_ind < conflict] > 0)
    # Two cells whose fields lie far from the animal fire together, and each
    # spike's likelihood is disjoint from the prediction.
    assert counts[conflict].sum() == 2
    assert np.all(
        np.abs(centers[list(sequence.conflict_cells)] - sequence.true_position[conflict]) > 30
    )
    at_conflict = decoded.event_time_ind == conflict
    np.testing.assert_array_equal(decoded.event_hpd_overlap[at_conflict], 0.0)
    # Elsewhere the spikes come from the model, so the decoder tracks the
    # animal before the conflict and recovers by the end.
    mean = decoded.posterior @ sequence.position_bins
    assert np.median(np.abs(mean[:conflict] - sequence.true_position[:conflict])) < 5
    assert abs(mean[-1] - sequence.true_position[-1]) < 5


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
    np.testing.assert_array_equal(position_bins, simulation.position_bins)
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
            == figure03_summary["median_decoding_accuracy"][
                figure03_summary["accuracy_metric_order"].index("median_absolute_error")
            ][column]
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
        "provenance": {
            "figure04_decode_cache": {
                "fingerprint_sha256": "abc",
                "diagnostics_fingerprint_sha256": "def",
            }
        },
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
    assert payload["diagnostics_fingerprint"] == "def"
    # The raster covers the same bins as the events: [time[start], time[stop]).
    t_end = render_data.time[time_slice.stop]
    for unit, times in enumerate(render_data.recording.spike_times):
        in_bins = (times >= render_data.time[time_slice.start]) & (times < t_end)
        assert len(payload["spike_times"][unit]) == int(in_bins.sum())


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


def _assert_rows_close(committed: str, fresh: str, n_bins: int) -> None:
    """Quantized rows may differ by one level across platforms, never more."""
    difference = decode_display_rows(committed, n_bins).astype(int) - decode_display_rows(
        fresh, n_bins
    ).astype(int)
    assert np.abs(difference).max() <= 1


def test_committed_scenarios_are_current(scenarios: dict[str, dict[str, Any]]) -> None:
    for condition_id, fresh in scenarios.items():
        committed = _load(SITE_DATA_DIR / f"scenario_{condition_id}.json")
        n_bins = len(fresh["position_bins"])
        for key in ("start", "stop", "scored_windows", "summary", "label", "model_component"):
            assert committed[key] == fresh[key], (condition_id, key)
        for key in ("t", "cell", "likelihood_row", "flagged"):
            assert committed["events"][key] == fresh["events"][key], (condition_id, key)
        for metric in ("hpd_overlap", "predictive_pvalue", "kl_divergence"):
            np.testing.assert_allclose(
                committed["events"][metric], fresh["events"][metric], rtol=1e-3
            )
        for key in ("position_bins", "true_position", "posterior_mean", "cell_centers"):
            np.testing.assert_allclose(committed[key], fresh[key], atol=0.011)
        np.testing.assert_allclose(
            committed["predictive"]["row_max"], fresh["predictive"]["row_max"], rtol=1e-5
        )
        np.testing.assert_allclose(
            committed["predictive"]["range"], fresh["predictive"]["range"], rtol=1e-9
        )
        _assert_rows_close(committed["predictive"]["rows"], fresh["predictive"]["rows"], n_bins)
        _assert_rows_close(committed["likelihood_rows"], fresh["likelihood_rows"], n_bins)


def test_committed_filter_data_is_current() -> None:
    committed = _load(SITE_DATA_DIR / "filter.json")
    fresh = filter_explainer_payload(Figure3Config())
    assert committed.keys() == fresh.keys()
    n_bins = len(fresh["position_bins"])
    for key in ("predictive", "posterior", "place_fields"):
        _assert_rows_close(committed[key]["rows"], fresh[key]["rows"], n_bins)
        np.testing.assert_allclose(committed[key]["row_max"], fresh[key]["row_max"], rtol=1e-5)
        np.testing.assert_allclose(committed[key]["range"], fresh[key]["range"], rtol=1e-9)
    _assert_rows_close(committed["likelihood"], fresh["likelihood"], n_bins)
    assert committed["events"]["t"] == fresh["events"]["t"]
    assert committed["events"]["cell"] == fresh["events"]["cell"]
    np.testing.assert_allclose(
        committed["events"]["hpd_overlap"], fresh["events"]["hpd_overlap"], rtol=1e-3
    )
    for key, values in fresh["moments"].items():
        np.testing.assert_allclose(committed["moments"][key], values, atol=0.011)
    for key in ("position_bins", "cell_centers", "true_position", "exposure"):
        np.testing.assert_allclose(committed[key], fresh[key], atol=0.011)
    for key in ("conflict_step", "coverage"):
        assert committed[key] == fresh[key], key


def test_committed_playground_is_current(
    simulation: Figure3SimulationResult, figure03_summary: dict[str, Any]
) -> None:
    committed = _load(SITE_DATA_DIR / "playground.json")
    fresh = playground_payload(
        Figure3Config(), np.asarray(simulation.sparse_place_field_centers), figure03_summary
    )
    for key in ("flag_rules", "coverage", "position_bins", "cell_centers"):
        assert committed[key] == fresh[key], key
    for old, new in zip(committed["ensembles"], fresh["ensembles"], strict=True):
        assert old["ensemble_id"] == new["ensemble_id"]
        assert old["selectable_cells"] == new["selectable_cells"]
        np.testing.assert_allclose(old["rates"], new["rates"], rtol=1e-12)


def test_committed_replay_matches_the_figure4_decode(figure04_summary: dict[str, Any]) -> None:
    committed = _load(SITE_DATA_DIR / "replay.json")
    decode_cache = figure04_summary["provenance"]["figure04_decode_cache"]
    # A decoder or a diagnostics change must be followed by a re-export.
    assert committed["decode_cache_fingerprint"] == decode_cache["fingerprint_sha256"]
    assert committed["diagnostics_fingerprint"] == decode_cache["diagnostics_fingerprint_sha256"]
    assert committed["flag_rules"] == figure04_summary["flag_rules"]
    continuous, fragmented = (
        committed["models"][name]["events"] for name in ("continuous", "continuous_fragmented")
    )
    # Both decoders are scored on the same spikes.
    assert continuous["t"] == fragmented["t"]
    assert continuous["cell"] == fragmented["cell"]


def test_site_stylesheet_uses_the_paper_palette() -> None:
    """The site's CSS color tokens match the figures' colors in ``style``."""
    css = (REPO_ROOT / "site/css/style.css").read_text(encoding="utf-8").lower()
    tokens = {
        "--predictive": COLORS["predictive"],
        "--likelihood": COLORS["likelihood"],
        "--posterior": COLORS["posterior"],
        "--position": COLORS["ground_truth"],
        "--threshold": COLORS["threshold"],
        **{f"--{css_name}": spec.color for css_name, spec in _METRIC_CSS.items()},
    }
    for token, color in tokens.items():
        assert f"{token}: {color.lower()};" in css, token


_METRIC_CSS = {
    css_name: next(spec for spec in METRIC_SPECS if spec.name == metric)
    for css_name, metric in (
        ("hpd", "hpd_overlap"),
        ("pvalue", "predictive_pvalue"),
        ("kl", "kl_divergence"),
    )
}


def test_simulation_hpd_threshold_matches_the_page_text(figure03_summary: dict[str, Any]) -> None:
    """site/index.html says the simulation's HPD cutoff flags only disjoint regions."""
    rule = figure03_summary["flag_rules"]["hpd_overlap"]
    assert rule == {"comparison": "less_than_or_equal", "threshold": 0.0}
