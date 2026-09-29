"""Tests for the website data export (``statespacecheck_paper.site_export``).

Covers the encoding helpers, the filter explainer, the playground and
parity-fixture builders, the Figure-3 condition payloads (against the real seed-1
realization), the Figure-4 recording payload (against synthetic render data), and
that the committed files under ``site/`` are current with the committed figure
summaries.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import statespacecheck as ssc
from numpy.typing import NDArray

from statespacecheck_paper.decoding import decode_with_diagnostics
from statespacecheck_paper.diagnostics import (
    METRIC_FLAG_DIRECTIONS,
    compute_spike_event_diagnostics_from_rates,
    flag_mask,
)
from statespacecheck_paper.figure03_generation import conditions_by_id
from statespacecheck_paper.figure03_protocol import STEP_SECONDS, Figure3Config
from statespacecheck_paper.figure03_simulation import (
    Figure3SimulationResult,
    all_place_field_centers,
    build_figure03_rate_tables,
    run_figure03_simulation,
)
from statespacecheck_paper.figure04_diagnostics import mean_event_likelihood_by_time
from statespacecheck_paper.figure04_layout import Figure4DetailWindow
from statespacecheck_paper.figure04_models import FIGURE04_MODELS, figure04_model
from statespacecheck_paper.figure04_workflow import Figure4RenderData
from statespacecheck_paper.number_format import significant, whole_percent
from statespacecheck_paper.paths import (
    FIGURE03_SUMMARY_PATH,
    FIGURE04_SUMMARY_PATH,
    PARITY_FIXTURE_PATH,
    REPO_ROOT,
    SITE_DATA_DIR,
)
from statespacecheck_paper.scientific_artifacts import inclusive_flag_rules
from statespacecheck_paper.simulation import gaussian_transition_matrix, place_field_rates
from statespacecheck_paper.site_export import (
    CONDITION_WINDOWS,
    FILTER_EXPLAINER,
    PLAYGROUND_PRESETS,
    FilterExplainerSequence,
    PlaygroundPreset,
    condition_payloads,
    encode_display_rows,
    filter_explainer_payload,
    filter_explainer_sequence,
    flag_events,
    flag_threshold_text,
    gaussian_predictive,
    manifest_payload,
    metric_parity_fixture,
    page_values,
    playground_ensembles,
    playground_payload,
    recording_payload,
)
from statespacecheck_paper.style import COLORS, METRIC_NAMES, METRIC_SPECS, PREDICTIVE_VMAX_QUANTILE
from tests.test_figure04_layout import _compose_render_data


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    return loaded


def decode_display_rows(encoded: str, n_bins: int) -> NDArray[np.uint8]:
    """Invert ``encode_display_rows`` to a ``(n_rows, n_bins)`` array."""
    flat = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    if flat.size % n_bins:
        raise ValueError(f"{flat.size} encoded values do not divide into rows of {n_bins}")
    return flat.reshape(-1, n_bins)


@pytest.fixture(scope="module")
def figure03_summary() -> dict[str, Any]:
    return _load(FIGURE03_SUMMARY_PATH)


@pytest.fixture(scope="module")
def figure04_summary() -> dict[str, Any]:
    return _load(FIGURE04_SUMMARY_PATH)


@pytest.fixture(scope="module")
def config() -> Figure3Config:
    return Figure3Config()


@pytest.fixture(scope="module")
def simulation(config: Figure3Config) -> Figure3SimulationResult:
    return run_figure03_simulation(config)


@pytest.fixture(scope="module")
def sparse_centers(simulation: Figure3SimulationResult) -> NDArray[np.float64]:
    return np.asarray(simulation.sparse_place_field_centers, dtype=np.float64)


@pytest.fixture(scope="module")
def parity_fixture(config: Figure3Config, sparse_centers: NDArray[np.float64]) -> dict[str, Any]:
    return metric_parity_fixture(config, sparse_centers)


@pytest.fixture(scope="module")
def site_conditions(
    simulation: Figure3SimulationResult, figure03_summary: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return condition_payloads(simulation, figure03_summary)


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


def test_flag_events_reads_the_rules_the_summaries_record() -> None:
    values = np.array([0.04, 0.05, 0.06, np.nan])
    rules = inclusive_flag_rules(
        dict.fromkeys(METRIC_FLAG_DIRECTIONS, 0.05), METRIC_FLAG_DIRECTIONS
    )
    for metric, direction in METRIC_FLAG_DIRECTIONS.items():
        np.testing.assert_array_equal(
            flag_events(values, rules[metric]), flag_mask(values, 0.05, direction)
        )


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
    assert np.all(np.abs(centers[np.flatnonzero(counts[conflict])] - position[conflict]) > 30)
    at_conflict = decoded.event_time_ind == conflict
    np.testing.assert_array_equal(decoded.event_hpd_overlap[at_conflict], 0.0)
    # Elsewhere the spikes come from the model, so the decoder tracks the
    # animal before the conflict and recovers by the end: no later spike is
    # inconsistent with the prediction, and playback pauses only at the conflict.
    assert np.all(decoded.event_hpd_overlap[decoded.event_time_ind > conflict] > 0)
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


def test_playground_ensembles_are_the_figure03_decoder_tables(
    simulation: Figure3SimulationResult,
    config: Figure3Config,
    sparse_centers: NDArray[np.float64],
) -> None:
    assert config.place_field_centers is not None
    ensembles = playground_ensembles(config, sparse_centers)
    position_bins = config.position_bins
    cell_centers = all_place_field_centers(config, sparse_centers)
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
    config: Figure3Config, sparse_centers: NDArray[np.float64], figure03_summary: dict[str, Any]
) -> None:
    payload = playground_payload(config, sparse_centers, figure03_summary)
    assert payload["flag_rules"] == figure03_summary["flag_rules"]
    assert payload["flag_threshold_text"] == flag_threshold_text(figure03_summary)
    assert [e["ensemble_id"] for e in payload["ensembles"]] == ["place_cells", "sparse_epoch"]
    for ensemble in payload["ensembles"]:
        assert len(ensemble["rates"]) == len(payload["position_bins"])


def test_flag_thresholds_print_by_the_reporting_policy(figure03_summary: dict[str, Any]) -> None:
    """Pooled-baseline thresholds print to two significant figures, cutoffs in full."""
    rules = figure03_summary["flag_rules"]
    assert flag_threshold_text(figure03_summary) == {
        "hpd_overlap": significant(rules["hpd_overlap"]["threshold"]),
        "kl_divergence": significant(rules["kl_divergence"]["threshold"]),
        "predictive_pvalue": "0.05",
    }
    # The committed KL threshold, 4.138..., reads 4.1 on the page.
    assert flag_threshold_text(figure03_summary)["kl_divergence"] == "4.1"
    assert flag_threshold_text(figure03_summary)["hpd_overlap"] == "0"


# Each playground example's button label (site/index.html) and the diagnostics
# that label claims flag its spike under the Figure-3 flag rules.
_PRESET_CLAIMS: dict[str, tuple[str, set[str]]] = {
    "consistent": ("Consistent", set()),
    "conflicting": ("Conflicting", {"hpd_overlap", "predictive_pvalue", "kl_divergence"}),
    # A narrow prediction inside the spike's broader likelihood is consistent.
    "nested": ("Narrow prediction", set()),
    # Only KL divergence responds to the difference in spread.
    "broad": ("Broad prediction, narrow spike", {"kl_divergence"}),
    # HPD overlap catches the conflict the p-value misses (KL flags it too).
    "pvalue_miss": ("p-value misses a conflict", {"hpd_overlap", "kl_divergence"}),
}


def test_playground_buttons_are_the_exported_presets() -> None:
    html = (REPO_ROOT / "site" / "index.html").read_text(encoding="utf-8")
    buttons = dict(re.findall(r'data-preset="([^"]+)">([^<]+)<', html))
    assert buttons == {preset_id: label for preset_id, (label, _) in _PRESET_CLAIMS.items()}
    assert [preset.preset_id for preset in PLAYGROUND_PRESETS] == list(_PRESET_CLAIMS)


@pytest.mark.parametrize("preset", PLAYGROUND_PRESETS, ids=lambda preset: preset.preset_id)
def test_playground_preset_flags_what_its_label_claims(
    preset: PlaygroundPreset,
    config: Figure3Config,
    sparse_centers: NDArray[np.float64],
    figure03_summary: dict[str, Any],
) -> None:
    ensemble = {e.ensemble_id: e for e in playground_ensembles(config, sparse_centers)}[
        preset.ensemble_id
    ]
    assert preset.cell in ensemble.selectable_cells
    diagnostics = compute_spike_event_diagnostics_from_rates(
        gaussian_predictive(config.position_bins, preset.mean, preset.std)[np.newaxis, :],
        ensemble.rates,
        np.array([0], dtype=np.intp),
        np.array([preset.cell], dtype=np.intp),
        include_dense_matrices=False,
    )
    flagged = {
        metric
        for metric, rule in figure03_summary["flag_rules"].items()
        if flag_events(getattr(diagnostics, f"event_{metric}"), rule)[0]
    }
    assert flagged == _PRESET_CLAIMS[preset.preset_id][1]


def test_parity_fixture_matches_the_python_diagnostics(parity_fixture: dict[str, Any]) -> None:
    fixture = parity_fixture
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
# Condition player (Figure 3)
# ---------------------------------------------------------------------------


def test_condition_windows_cover_every_condition_and_overlap_its_scored_steps(
    config: Figure3Config,
) -> None:
    conditions = conditions_by_id(config)
    assert [window.condition_id for window in CONDITION_WINDOWS] == list(conditions)
    n_time = config.phase_boundaries[-1]
    for window in CONDITION_WINDOWS:
        assert 0 <= window.start < window.stop <= n_time
        assert any(
            t0 < window.stop and t1 > window.start
            for t0, t1 in conditions[window.condition_id].step_windows
        ), window


def test_condition_events_match_the_decoded_diagnostics(
    simulation: Figure3SimulationResult,
    site_conditions: dict[str, dict[str, Any]],
    figure03_summary: dict[str, Any],
) -> None:
    diagnostics = simulation.diagnostics
    n_bins = simulation.position_bins.size
    for window in CONDITION_WINDOWS:
        payload = site_conditions[window.condition_id]
        events = payload["events"]
        in_window = (diagnostics.event_time_ind >= window.start) & (
            diagnostics.event_time_ind < window.stop
        )
        np.testing.assert_array_equal(
            np.asarray(events["t"]) + window.start, diagnostics.event_time_ind[in_window]
        )
        np.testing.assert_array_equal(events["cell"], diagnostics.event_cell_ind[in_window])
        assert payload["step_seconds"] == STEP_SECONDS
        for metric in METRIC_NAMES:
            full = np.asarray(getattr(diagnostics, f"event_{metric}"))[in_window]
            np.testing.assert_allclose(events[metric], full, rtol=5e-4, atol=0)
            np.testing.assert_array_equal(
                events["flagged"][metric],
                flag_events(full, figure03_summary["flag_rules"][metric]),
            )
        # Each event's likelihood row decodes to its own quantized likelihood.
        rows = decode_display_rows(payload["likelihood_rows"], n_bins)
        expected = decode_display_rows(
            encode_display_rows(diagnostics.event_likelihood[in_window]), n_bins
        )
        np.testing.assert_array_equal(rows[events["likelihood_row"]], expected)
        predictive = decode_display_rows(payload["predictive"]["rows"], n_bins)
        assert predictive.shape == (window.stop - window.start, n_bins)
        # One color scale for every window, the one Figure 3a uses.
        assert payload["predictive"]["range"] == [
            0.0,
            float(np.nanquantile(diagnostics.predictive, PREDICTIVE_VMAX_QUANTILE)),
        ]
        np.testing.assert_allclose(
            payload["predictive"]["row_max"],
            diagnostics.predictive[window.start : window.stop].max(axis=1),
            rtol=1e-5,
        )
        assert len(payload["true_position"]) == window.stop - window.start


def test_condition_summaries_come_from_the_figure_summary(
    site_conditions: dict[str, dict[str, Any]], figure03_summary: dict[str, Any]
) -> None:
    """The page quotes the summary's medians, rounded with the manuscript's policy."""
    metric_order = figure03_summary["metric_order"]
    error_row = figure03_summary["error_metric_order"].index("median_absolute_error")
    for column, condition_id in enumerate(figure03_summary["condition_order"]):
        summary = site_conditions[condition_id]["summary"]
        assert set(summary["median_flag_percent_text"]) == set(METRIC_NAMES)
        for metric, text in summary["median_flag_percent_text"].items():
            row = metric_order.index(metric)
            assert text == whole_percent(figure03_summary["median_flag_percentages"][row][column])
        assert summary["median_absolute_error_text"] == significant(
            figure03_summary["median_decoding_error"][error_row][column]
        )


# ---------------------------------------------------------------------------
# Recording comparison (Figure 4)
# ---------------------------------------------------------------------------


def _recording_summary(render_data: Figure4RenderData) -> dict[str, Any]:
    """A Figure-4 summary whose cache fingerprints match ``render_data``."""
    provenance = render_data.cache_provenance
    return {
        "flag_rules": {
            "hpd_overlap": {"comparison": "less_than_or_equal", "threshold": 0.05},
            "predictive_pvalue": {"comparison": "less_than_or_equal", "threshold": 0.05},
        },
        "provenance": {
            "figure04_caches": {
                "fingerprint_sha256": provenance.fingerprint_sha256,
                "diagnostics_fingerprint_sha256": provenance.diagnostics_fingerprint_sha256,
            }
        },
    }


def test_recording_payload_slices_both_models_to_the_detail_window() -> None:
    render_data = _compose_render_data()
    summary = _recording_summary(render_data)
    window = Figure4DetailWindow(center_index=20, half_width_samples=10)
    payload = recording_payload(render_data, summary, window)
    time_slice = window.to_slice(render_data.time.size)
    n_time = time_slice.stop - time_slice.start
    decode = render_data.decode_results
    n_bins = decode.diagnostic_position_bins.size

    assert len(payload["time"]) == n_time
    assert payload["time"][0] == 0.0
    likelihood, has_spikes = mean_event_likelihood_by_time(
        decode.spike_counts[time_slice], decode.diagnostic_place_fields
    )
    np.testing.assert_array_equal(
        decode_display_rows(payload["likelihood"], n_bins),
        decode_display_rows(encode_display_rows(likelihood), n_bins),
    )
    assert payload["has_spikes"] == has_spikes.tolist()
    np.testing.assert_array_equal(
        decode_display_rows(payload["cell_likelihoods"], n_bins),
        decode_display_rows(
            encode_display_rows(ssc.event_likelihood(decode.diagnostic_place_fields)),
            n_bins,
        ),
    )
    for name, diagnostics in (
        ("continuous", decode.continuous_diagnostics),
        ("continuous_fragmented", decode.continuous_fragmented_diagnostics),
    ):
        model = payload["models"][name]
        # The page takes the model labels from the export, not its own copy.
        assert (model["label"], model["short_label"]) == (
            figure04_model(name).label,
            figure04_model(name).short_label,
        )
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
    # Cells are ranked by place-field peak.
    assert sorted(payload["cell_rank"]) == list(range(decode.place_field_peaks.size))
    assert payload["decode_cache_fingerprint"] == render_data.cache_provenance.fingerprint_sha256
    assert (
        payload["diagnostics_fingerprint"]
        == render_data.cache_provenance.diagnostics_fingerprint_sha256
    )
    # The raster covers the same bins as the events: [time[start], time[stop]).
    t_end = render_data.time[time_slice.stop]
    for cell, times in enumerate(render_data.recording.spike_times):
        in_bins = (times >= render_data.time[time_slice.start]) & (times < t_end)
        assert len(payload["spike_times"][cell]) == int(in_bins.sum())


@pytest.mark.parametrize("key", ["fingerprint_sha256", "diagnostics_fingerprint_sha256"])
def test_recording_payload_rejects_a_summary_from_another_decode(key: str) -> None:
    """The window is not labeled with fingerprints of a decode other than the exported one."""
    render_data = _compose_render_data()
    summary = _recording_summary(render_data)
    summary["provenance"]["figure04_caches"][key] = "0" * 64
    window = Figure4DetailWindow(center_index=20, half_width_samples=10)
    with pytest.raises(ValueError, match=rf"\({key} differ\)"):
        recording_payload(render_data, summary, window)


# ---------------------------------------------------------------------------
# Committed site files are current
# ---------------------------------------------------------------------------


def test_page_values_spell_the_recording_window_length(figure04_summary: dict[str, Any]) -> None:
    """The page's "two-second window" is the Figure-4 detail window at the decoder's bin rate."""
    assert page_values(figure04_summary) == {"RecordingWindowSecondsWord": "two"}

    summary = copy.deepcopy(figure04_summary)
    summary["detail_window"]["half_width_samples"] = 1_500
    assert page_values(summary) == {"RecordingWindowSecondsWord": "six"}
    summary["detail_window"]["half_width_samples"] = 600
    with pytest.raises(ValueError, match="whole seconds"):
        page_values(summary)


def test_committed_manifest_matches_the_figure_summaries(
    figure03_summary: dict[str, Any], figure04_summary: dict[str, Any]
) -> None:
    committed = _load(SITE_DATA_DIR / "manifest.json")
    assert committed == manifest_payload(figure03_summary, figure04_summary)


def test_committed_parity_fixture_is_current(parity_fixture: dict[str, Any]) -> None:
    committed = _load(PARITY_FIXTURE_PATH)
    fresh = parity_fixture
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


def _assert_heatmap_close(committed: dict[str, Any], fresh: dict[str, Any], n_bins: int) -> None:
    """A committed ``heatmap_payload`` matches a fresh one up to platform rounding."""
    _assert_rows_close(committed["rows"], fresh["rows"], n_bins)
    np.testing.assert_allclose(committed["row_max"], fresh["row_max"], rtol=1e-5)
    np.testing.assert_allclose(committed["range"], fresh["range"], rtol=1e-9)


def test_committed_condition_payloads_are_current(
    site_conditions: dict[str, dict[str, Any]],
) -> None:
    for condition_id, fresh in site_conditions.items():
        committed = _load(SITE_DATA_DIR / f"condition_{condition_id}.json")
        assert committed.keys() == fresh.keys(), condition_id
        n_bins = len(fresh["position_bins"])
        for key in (
            "start",
            "stop",
            "step_seconds",
            "scored_windows",
            "summary",
            "model_component",
        ):
            assert committed[key] == fresh[key], (condition_id, key)
        for key in ("t", "cell", "flagged"):
            assert committed["events"][key] == fresh["events"][key], (condition_id, key)
        for metric in METRIC_NAMES:
            np.testing.assert_allclose(
                committed["events"][metric], fresh["events"][metric], rtol=1e-3
            )
        for key in ("position_bins", "true_position", "cell_centers"):
            np.testing.assert_allclose(committed[key], fresh[key], atol=0.011)
        _assert_heatmap_close(committed["predictive"], fresh["predictive"], n_bins)
        # The row table deduplicates float likelihoods, so rows that coincide on
        # one platform can differ in the last bit on another, changing the table's
        # length and every later index. Compare the row each event points to.
        committed_rows, fresh_rows = (
            decode_display_rows(payload["likelihood_rows"], n_bins)[
                payload["events"]["likelihood_row"]
            ].astype(int)
            for payload in (committed, fresh)
        )
        assert committed_rows.shape == fresh_rows.shape, condition_id
        assert np.abs(committed_rows - fresh_rows).max() <= 1, condition_id


def test_committed_filter_data_is_current() -> None:
    committed = _load(SITE_DATA_DIR / "filter.json")
    fresh = filter_explainer_payload(Figure3Config())
    assert committed.keys() == fresh.keys()
    n_bins = len(fresh["position_bins"])
    for key in ("predictive", "posterior", "place_fields"):
        _assert_heatmap_close(committed[key], fresh[key], n_bins)
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
    config: Figure3Config, sparse_centers: NDArray[np.float64], figure03_summary: dict[str, Any]
) -> None:
    committed = _load(SITE_DATA_DIR / "playground.json")
    fresh = playground_payload(config, sparse_centers, figure03_summary)
    assert committed.keys() == fresh.keys()
    for key in (
        "flag_rules",
        "flag_threshold_text",
        "presets",
        "coverage",
        "position_bins",
        "cell_centers",
    ):
        assert committed[key] == fresh[key], key
    for old, new in zip(committed["ensembles"], fresh["ensembles"], strict=True):
        assert old["ensemble_id"] == new["ensemble_id"]
        assert old["selectable_cells"] == new["selectable_cells"]
        np.testing.assert_allclose(old["rates"], new["rates"], rtol=1e-12)


def test_committed_recording_matches_the_figure04_decode(figure04_summary: dict[str, Any]) -> None:
    committed = _load(SITE_DATA_DIR / "recording.json")
    caches = figure04_summary["provenance"]["figure04_caches"]
    # A decoder or a diagnostics change must be followed by a re-export.
    assert committed["decode_cache_fingerprint"] == caches["fingerprint_sha256"]
    assert committed["diagnostics_fingerprint"] == caches["diagnostics_fingerprint_sha256"]
    assert committed["flag_rules"] == figure04_summary["flag_rules"]
    assert {
        name: (model["label"], model["short_label"]) for name, model in committed["models"].items()
    } == {model.id: (model.label, model.short_label) for model in FIGURE04_MODELS}
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
