"""Behavior tests for the figure-3 metric-dissociation phases.

These tests verify the scientific claims of the figure-3 simulation:

- The unperturbed phases are an exactly matched null: the trajectory is a
  draw from the decoder's own transition matrix (on its grid) and the spikes
  come from its own rate tables.
- The remap phase diagnostics use the decoder's remapped likelihood,
  rather than an oracle baseline rate table.
- The reflected-map phase is a coherent wrong map: internal
  prediction/likelihood agreement stays near baseline while the decoded
  position is far from the truth.
- In the matched low-information regime, isolated spikes from sparse narrow
  cells elevate KL while HPD overlap and the rank-based p-value flag few
  events, and every flag there is a false positive against a correct model.
- The history-dependent firing phase has a rate-matched marginal and produces
  per-spike metrics comparable to baseline.
- Thresholds are calibrated on independent matched-null sessions and the
  rank-based p-value is super-uniform under that null.

If any of these assertions ever flips, the figure no longer tells the
story the paper claims; CI flags the regression.
"""

from __future__ import annotations

import numpy as np
import pytest
import statespacecheck as ssc

from statespacecheck_paper.diagnostics import (
    DiagnosticThresholds,
    compute_spike_event_diagnostics_from_rates,
)
from statespacecheck_paper.figure03_protocol import (
    DEFAULT_HISTORY_RATE_MATCHING_GAIN,
    PHASE_LABELS,
    Figure3Config,
    PhaseBoundary,
)
from statespacecheck_paper.figure03_simulation import (
    Figure3SimulationResult,
    _single_out_and_back_sweep,
    build_figure03_rate_tables,
    estimate_history_rate_matching_gain,
    reflect_place_field_centers,
    remap_place_field_centers,
    run_figure03_simulation,
    run_matched_null_simulation,
    simulate_drift_phase,
    sparse_population_rates,
)
from statespacecheck_paper.figure03_summary import (
    CONDITION_IDS,
    RANK_CALIBRATION_ALPHAS,
    Figure3Calibration,
    Figure3RealizationSummary,
    estimate_calibration_thresholds,
    estimate_realization_summary,
)
from statespacecheck_paper.simulation import gaussian_transition_matrix, place_field_rates


def _moderate_params() -> Figure3Config:
    """Phase sizes large enough for per-phase medians to be stable but
    small enough to keep the test fast.
    """
    return Figure3Config(
        phase_boundaries=(600, 900, 1100, 1400, 1600, 1900, 2100, 3100, 3300, 4300),
    )


def _per_phase_medians(
    sim: Figure3SimulationResult,
) -> dict[str, tuple[float, float, float]]:
    """Return (kl_med, hpd_med, sp_med) per phase label (last occurrence wins)."""
    metrics = sim.diagnostics
    boundaries = np.asarray(sim.phase_boundaries)
    labels = sim.phase_labels
    event_phase = np.searchsorted(boundaries, metrics.event_time_ind, side="right")
    out: dict[str, tuple[float, float, float]] = {}
    for i, label in enumerate(labels):
        mask = event_phase == i
        if not mask.any():
            continue
        kl = float(np.nanmedian(metrics.event_kl_divergence[mask]))
        hpd = float(np.nanmedian(metrics.event_hpd_overlap[mask]))
        sp = float(np.nanmedian(metrics.event_predictive_pvalue[mask]))
        out[label] = (kl, hpd, sp)
    return out


@pytest.fixture(scope="module")
def sim() -> Figure3SimulationResult:
    return run_figure03_simulation(_moderate_params(), seed=0)


@pytest.fixture(scope="module")
def calibration() -> Figure3Calibration:
    return estimate_calibration_thresholds(
        _moderate_params(), n_calibration_realizations=4, first_calibration_seed=500
    )


@pytest.fixture(scope="module")
def summary(calibration: Figure3Calibration) -> Figure3RealizationSummary:
    return estimate_realization_summary(
        _moderate_params(), calibration=calibration, n_realizations=5, first_random_seed=0
    )


def _column(summary: Figure3RealizationSummary, condition_id: str) -> np.ndarray:
    return summary.median_flag_percentages[:, CONDITION_IDS.index(condition_id)]


def test_phase_labels_and_boundaries(sim: Figure3SimulationResult) -> None:
    """``run_figure03_simulation`` emits every canonical phase in order
    and a timeline that ends at the SPARSE_POP_END boundary.
    """
    params = sim.config
    assert sim.phase_labels == PHASE_LABELS
    assert len(PHASE_LABELS) == 10
    for misfit in (
        "Remap Misfit",
        "History-Dependent Firing",
        "Drift Misfit",
        "Reflected Map Misfit",
        "Sparse Population",
    ):
        assert PHASE_LABELS.count(misfit) == 1
    boundaries = np.asarray(sim.phase_boundaries)
    end = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    assert boundaries[-1] == end
    assert np.all(np.diff(boundaries) > 0)
    assert sim.true_position.shape == (end,)
    assert sim.represented_position.shape == (end,)


def test_matched_phases_follow_the_decoder_transition_on_the_grid(
    sim: Figure3SimulationResult,
) -> None:
    """Outside the drift phase the trajectory is grid-valued and every step is
    a transition with positive probability under the decoder's matrix; the
    first sample is a draw from the decoder's uniform initial law."""
    params = sim.config
    bnd = params.phase_boundaries
    transition = gaussian_transition_matrix(sim.position_bins, params.prediction_step_std)
    x = sim.true_position
    drift = slice(bnd[PhaseBoundary.RECOVERY2_END], bnd[PhaseBoundary.DRIFT_END])
    on_grid = np.isin(x, sim.position_bins)
    assert on_grid[: drift.start].all() and on_grid[drift.stop :].all()
    index = np.abs(x[:, None] - sim.position_bins[None, :]).argmin(axis=1)
    baseline = slice(0, bnd[PhaseBoundary.REMAP_START])
    steps = transition[index[baseline][1:], index[baseline][:-1]]
    assert np.all(steps > 0.0)
    # A different seed draws a different initial grid position (uniform law).
    other = run_figure03_simulation(_moderate_params(), seed=3)
    assert other.true_position[0] != x[0] or other.true_position[1] != x[1]


def test_represented_position_differs_only_in_the_replay_sweep(
    sim: Figure3SimulationResult,
) -> None:
    from statespacecheck_paper.figure03_protocol import compute_replay_step_window

    r0, r1 = compute_replay_step_window(sim.config)
    differs = ~np.isclose(sim.represented_position, sim.true_position)
    assert differs[r0:r1].any()
    assert not differs[:r0].any() and not differs[r1:].any()
    # The physical position is fixed during the sweep.
    assert np.allclose(sim.true_position[r0:r1], sim.true_position[r0])


@pytest.mark.parametrize(
    ("start", "expected_endpoint"),
    [(20.0, 100.0), (80.0, 0.0)],
)
def test_replay_is_one_out_and_back_sweep(start: float, expected_endpoint: float) -> None:
    """Replay visits the farther endpoint once and returns without oscillating."""
    sweep = _single_out_and_back_sweep(start, 2_000, 0.0, 100.0, 0.5)

    assert sweep.shape == (2_000,)
    assert sweep[0] == pytest.approx(start)
    assert sweep[-1] == pytest.approx(start)
    turn = int(np.argmax(sweep) if expected_endpoint == 100.0 else np.argmin(sweep))
    assert sweep[turn] == pytest.approx(expected_endpoint)

    outbound_diff = np.diff(sweep[: turn + 1])
    inbound_diff = np.diff(sweep[turn:])
    if expected_endpoint == 100.0:
        assert np.all(outbound_diff >= 0.0)
        assert np.all(inbound_diff <= 0.0)
    else:
        assert np.all(outbound_diff <= 0.0)
        assert np.all(inbound_diff >= 0.0)


def test_short_replay_sweep_respects_speed_cap() -> None:
    """Short custom timelines make a smaller excursion instead of moving too fast."""
    sweep = _single_out_and_back_sweep(20.0, 100, 0.0, 100.0, 0.5)

    assert sweep.max() < 100.0
    assert np.max(np.abs(np.diff(sweep))) <= 0.5 + np.finfo(float).eps
    assert sweep[-1] == pytest.approx(sweep[0])


def test_replay_generative_and_decoder_share_tuning_model(sim: Figure3SimulationResult) -> None:
    """Replay is a correctly-specified observation model with a disclosed gain.

    The generative sweep spikes fire at the elevated ``replay_place_field_rate_scale``
    through the ordinary position-tuning model; the decoder's replay-window
    rate table must use the *same* tuning model and the *same* elevated
    scale (a fourfold gain in the manuscript configuration).
    """
    params = sim.config
    assert params.place_field_centers is not None
    rate_tables = build_figure03_rate_tables(
        sim.position_bins,
        params.place_field_centers,
        np.asarray(sim.sparse_place_field_centers),
        params,
    )
    n_normal = len(params.place_field_centers)
    decoder_replay_normal = rate_tables.replay_firing_rates[:, :n_normal]
    generative_replay = place_field_rates(
        sim.position_bins,
        params.place_field_centers,
        params.place_field_std,
        params.replay_place_field_rate_scale,
    )
    np.testing.assert_allclose(decoder_replay_normal, generative_replay)
    assert params.replay_place_field_rate_scale == pytest.approx(
        4.0 * params.place_field_rate_scale
    )


def test_remap_phase_uses_decoder_likelihood(sim: Figure3SimulationResult) -> None:
    """Remap diagnostics must use the same remapped rates as the decoder."""
    params = sim.config
    assert params.place_field_centers is not None
    start = params.phase_boundaries[PhaseBoundary.REMAP_START]
    end = params.phase_boundaries[PhaseBoundary.REMAP_END]
    in_window = (sim.diagnostics.event_time_ind >= start) & (sim.diagnostics.event_time_ind < end)
    assert in_window.any(), "test simulation produced no remap-window spike events"

    remapped_normal_rates = place_field_rates(
        sim.position_bins,
        remap_place_field_centers(
            params.place_field_centers, params.place_field_remapping, active=True
        ),
        params.place_field_std,
        params.place_field_rate_scale,
    )
    sparse_rates, _ = sparse_population_rates(sim.position_bins, params)
    baseline_sparse_firing_rates = params.sparse_cell_baseline_rate_fraction * sparse_rates
    remapped_firing_rates = np.hstack([remapped_normal_rates, baseline_sparse_firing_rates])
    expected = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        remapped_firing_rates,
        sim.diagnostics.event_time_ind[in_window],
        sim.diagnostics.event_cell_ind[in_window],
        coverage=params.hpd_coverage,
    )
    assert expected.per_spike_likelihood is not None
    np.testing.assert_allclose(
        sim.diagnostics.per_spike_likelihood[in_window], expected.per_spike_likelihood
    )
    predictive = sim.diagnostics.predictive[sim.diagnostics.event_time_ind[in_window]]
    np.testing.assert_allclose(
        sim.diagnostics.event_hpd_overlap[in_window],
        ssc.hpd_overlap(predictive, sim.diagnostics.per_spike_likelihood[in_window], coverage=0.95),
    )
    for name in ("event_hpd_overlap", "event_kl_divergence", "event_predictive_pvalue"):
        np.testing.assert_allclose(
            getattr(sim.diagnostics, name)[in_window],
            getattr(expected, name),
            err_msg=f"remap-window {name} did not use decoder rates",
        )
    baseline_rates = np.hstack(
        [
            place_field_rates(
                sim.position_bins,
                params.place_field_centers,
                params.place_field_std,
                params.place_field_rate_scale,
            ),
            baseline_sparse_firing_rates,
        ]
    )
    oracle = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        baseline_rates,
        sim.diagnostics.event_time_ind[in_window],
        sim.diagnostics.event_cell_ind[in_window],
    )
    for name in ("event_hpd_overlap", "event_kl_divergence", "event_predictive_pvalue"):
        assert not np.allclose(getattr(expected, name), getattr(oracle, name))


def test_reflected_map_is_coherent_and_uses_decoder_likelihood(
    sim: Figure3SimulationResult,
) -> None:
    """The reflected map mirrors every center; its diagnostics use that map.

    Coherence: the decoded state is far from the truth (a mirror image)
    while the per-spike metrics stay near the matched-null values.
    """
    params = sim.config
    assert params.place_field_centers is not None
    reflected = reflect_place_field_centers(
        params.place_field_centers, params.position_min, params.position_max
    )
    np.testing.assert_allclose(
        reflected, params.position_min + params.position_max - params.place_field_centers
    )
    start = params.phase_boundaries[PhaseBoundary.RECOVERY3_END]
    end = params.phase_boundaries[PhaseBoundary.REFLECT_END]
    in_window = (sim.diagnostics.event_time_ind >= start) & (sim.diagnostics.event_time_ind < end)
    assert in_window.any()
    rate_tables = build_figure03_rate_tables(
        sim.position_bins,
        params.place_field_centers,
        np.asarray(sim.sparse_place_field_centers),
        params,
    )
    expected = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        rate_tables.reflected_firing_rates,
        sim.diagnostics.event_time_ind[in_window],
        sim.diagnostics.event_cell_ind[in_window],
        coverage=params.hpd_coverage,
    )
    np.testing.assert_allclose(
        sim.diagnostics.event_hpd_overlap[in_window], expected.event_hpd_overlap
    )
    # Decoded position is mirrored: large error, but internal agreement near baseline.
    posterior_mean = sim.diagnostics.posterior @ sim.position_bins
    # Skip the first 100 steps of the window (the transient after the map switch).
    window = slice(start + 100, end)
    error = np.median(np.abs(posterior_mean[window] - sim.true_position[window]))
    mirror_error = np.median(
        np.abs(posterior_mean[window] - (params.position_max - sim.true_position[window]))
    )
    assert mirror_error < error
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, _ = medians["Matched Null Baseline"]
    refl_kl, refl_hpd, _ = medians["Reflected Map Misfit"]
    assert refl_hpd > 0.8 * base_hpd
    assert refl_kl < 3.0 * base_kl


def test_sparse_population_is_a_matched_low_information_regime(
    sim: Figure3SimulationResult,
) -> None:
    """The last phase keeps the matched trajectory and a correctly modeled
    sparse observation model: quiet ordinary ensemble, heterogeneous sparse
    cells, the decoder's own rate tables, and no transition change."""
    params = sim.config
    w0 = params.phase_boundaries[PhaseBoundary.RECOVERY4_END]
    w1 = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    n_sparse = len(sim.sparse_place_field_centers)
    n_normal = sim.spike_counts.shape[1] - n_sparse
    assert n_sparse == len(params.sparse_place_field_centers)
    assert np.isin(sim.true_position[w0:w1], sim.position_bins).all()
    assert sim.spike_counts[w0:w1, :n_normal].sum() == 0
    assert sim.spike_counts[w0:w1, n_normal:].sum() > 0

    in_window = (sim.diagnostics.event_time_ind >= w0) & (sim.diagnostics.event_time_ind < w1)
    assert in_window.any()
    assert np.all(sim.diagnostics.event_cell_ind[in_window] >= n_normal)

    sparse_rates, centers = sparse_population_rates(sim.position_bins, params)
    np.testing.assert_allclose(centers, sim.sparse_place_field_centers)
    # Heterogeneous gains: the peak rates differ across sparse cells.
    peaks = sparse_rates.max(axis=0)
    assert np.unique(np.round(peaks, 12)).size > 1
    expected = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        sparse_rates,
        sim.diagnostics.event_time_ind[in_window],
        (sim.diagnostics.event_cell_ind[in_window] - n_normal).astype(np.intp),
        coverage=params.hpd_coverage,
    )
    np.testing.assert_allclose(
        sim.diagnostics.per_spike_likelihood[in_window], expected.per_spike_likelihood
    )
    event_time = int(sim.diagnostics.event_time_ind[np.flatnonzero(in_window)[0]])
    transition = gaussian_transition_matrix(sim.position_bins, params.prediction_step_std)
    expected_predictive = transition @ sim.diagnostics.posterior[event_time - 1]
    expected_predictive /= expected_predictive.sum()
    np.testing.assert_allclose(sim.diagnostics.predictive[event_time], expected_predictive)
    rate_tables = build_figure03_rate_tables(
        sim.position_bins, params.place_field_centers, centers, params
    )
    np.testing.assert_allclose(
        rate_tables.sparse_population_firing_rates[:, n_normal:], sparse_rates
    )
    np.testing.assert_allclose(
        rate_tables.baseline_firing_rates[:, n_normal:],
        params.sparse_cell_baseline_rate_fraction * sparse_rates,
    )


def test_history_rate_matching_gain_is_pinned_and_matches_marginal_rate() -> None:
    """The pinned gain reproduces the independent calibration estimate and
    brings the history process's marginal rate to the Poisson baseline."""
    config = Figure3Config()
    gain, ratio = estimate_history_rate_matching_gain(config, n_calibration_seeds=4)
    assert abs(ratio - 1.0) < 2e-3
    assert gain == pytest.approx(DEFAULT_HISTORY_RATE_MATCHING_GAIN, rel=0.03)
    # Without matching the modulated process fires faster than the Poisson baseline.
    _, unmatched = estimate_history_rate_matching_gain(config, n_calibration_seeds=2, tolerance=1.0)
    assert unmatched >= 1.0


def test_history_dependent_firing_per_spike_metrics_near_baseline(
    sim: Figure3SimulationResult,
) -> None:
    """Per-spike spatial diagnostics largely miss the (rate-matched)
    history-dependent misspecification."""
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, base_sp = medians["Matched Null Baseline"]
    hd_kl, hd_hpd, hd_sp = medians["History-Dependent Firing"]
    assert hd_kl < 3 * base_kl
    assert hd_hpd > 0.9 * base_hpd
    assert 0.5 * base_sp < hd_sp < 1.5 * base_sp


def test_drift_phase_inflates_kl(sim: Figure3SimulationResult) -> None:
    medians = _per_phase_medians(sim)
    base_kl, _, _ = medians["Matched Null Baseline"]
    drift_kl, _, _ = medians["Drift Misfit"]
    assert drift_kl > 1.2 * base_kl


def test_simulate_drift_phase() -> None:
    params = Figure3Config()
    x = simulate_drift_phase(500, x_last=40.0, config=params, rng=np.random.default_rng(0))
    assert x.shape == (500,)
    assert x[0] == pytest.approx(40.0)
    assert np.all(x >= params.position_min) and np.all(x <= params.position_max)
    repeat = simulate_drift_phase(500, x_last=40.0, config=params, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(x, repeat)


def test_matched_null_simulation_is_a_draw_from_the_baseline_decoder() -> None:
    """The calibration session uses the decoder's grid, transition, and rates only."""
    config = _moderate_params()
    null = run_matched_null_simulation(config, seed=11, n_time_steps=400)
    assert null.true_position.shape == (400,)
    assert np.isin(null.true_position, null.position_bins).all()
    n_normal = len(config.place_field_centers)
    assert null.spike_counts.shape[1] == n_normal + len(config.sparse_place_field_centers)
    assert null.diagnostics.predictive.shape == (400, null.position_bins.size)
    np.testing.assert_allclose(null.diagnostics.predictive[0], 1.0 / null.position_bins.size)
    repeat = run_matched_null_simulation(config, seed=11, n_time_steps=400)
    np.testing.assert_array_equal(null.spike_counts, repeat.spike_counts)


class TestCalibration:
    def test_thresholds_and_realized_null_rates(self, calibration: Figure3Calibration) -> None:
        t = calibration.diagnostic_thresholds
        assert t.predictive_pvalue == 0.05
        assert 0.0 <= t.hpd_overlap <= 1.0 and np.isfinite(t.kl_divergence)
        assert calibration.pooled_null_flag_percentages.shape == (3,)
        assert calibration.per_realization_null_flag_percentages.shape == (4, 3)
        # Realized rates are at least the nominal levels (inclusive rule).
        assert calibration.pooled_null_flag_percentages[0] >= 1.0 - 1e-9
        assert calibration.pooled_null_flag_percentages[2] >= 1.0 - 1e-9
        assert 0.0 <= calibration.hpd_threshold_tie_percent <= 100.0

    def test_rank_pvalue_is_super_uniform_under_the_null(
        self, calibration: Figure3Calibration
    ) -> None:
        """Empirical tails must not exceed alpha beyond binomial sampling error."""
        n = calibration.n_pooled_events
        for alpha, tail in zip(
            RANK_CALIBRATION_ALPHAS, calibration.rank_pvalue_tail_percentages, strict=True
        ):
            slack = 100.0 * 3.0 * np.sqrt(alpha * (1 - alpha) / n)
            assert tail <= 100.0 * alpha + slack

    def test_calibration_seeds_must_not_overlap_evaluation(
        self, calibration: Figure3Calibration
    ) -> None:
        with pytest.raises(ValueError, match="overlap"):
            estimate_realization_summary(
                _moderate_params(),
                calibration=calibration,
                n_realizations=2,
                first_random_seed=calibration.first_calibration_seed,
            )


class TestEstimateRealizationSummary:
    def test_shapes_and_determinism(
        self, summary: Figure3RealizationSummary, calibration: Figure3Calibration
    ) -> None:
        n_cols = len(CONDITION_IDS)
        assert summary.n_realizations == 5
        assert summary.median_flag_percentages.shape == (3, n_cols)
        assert summary.flag_percentages_by_realization.shape == (5, 3, n_cols)
        assert summary.event_counts_by_realization.shape == (5, n_cols)
        assert summary.pooled_flag_percentages.shape == (3, n_cols)
        assert summary.median_decoding_accuracy.shape == (4, n_cols)
        assert summary.decoding_accuracy_by_realization.shape == (5, 4, n_cols)
        assert summary.replay_represented_accuracy.shape == (5, 2)
        assert summary.phase_spike_rates_by_realization.shape == (5, n_cols)
        assert summary.sparse_cell_flag_percentages_by_realization.shape == (5, 3)
        assert np.all(summary.median_flag_percentages >= 0.0)
        assert np.all(summary.median_flag_percentages <= 100.0)
        # Every non-sparse column has events in every realization; the short test
        # timeline's sparse regime can be empty (NaN percentage, excluded from medians).
        assert np.all(summary.event_counts_by_realization[:, :-1] > 0)
        coverage = summary.median_decoding_accuracy[1]
        assert np.all((coverage >= 0.0) & (coverage <= 100.0))
        repeat = estimate_realization_summary(
            _moderate_params(), calibration=calibration, n_realizations=5, first_random_seed=0
        )
        np.testing.assert_array_equal(
            summary.median_flag_percentages, repeat.median_flag_percentages
        )

    def test_remap_is_strongly_flagged_by_all_three(
        self, summary: Figure3RealizationSummary
    ) -> None:
        null = _column(summary, "matched_null")
        remap = _column(summary, "remap")
        drift = _column(summary, "drift")
        assert np.all(remap > 10.0)
        assert np.all(remap > 2.0 * null)
        assert np.all(remap > drift)

    def test_replay_is_not_flagged_and_tracks_the_represented_trajectory(
        self, summary: Figure3RealizationSummary
    ) -> None:
        replay = _column(summary, "replay")
        remap = _column(summary, "remap")
        assert np.all(replay < 15.0) and np.all(replay < 0.5 * remap)
        col = CONDITION_IDS.index("replay")
        physical_error = summary.median_decoding_accuracy[0, col]
        represented_error = np.median(summary.replay_represented_accuracy[:, 0])
        assert represented_error < physical_error

    def test_reflected_map_is_unflagged_but_wrong(self, summary: Figure3RealizationSummary) -> None:
        """The coherent wrong map: near-baseline flags, large error, low coverage."""
        refl = _column(summary, "reflected_map")
        remap = _column(summary, "remap")
        col = CONDITION_IDS.index("reflected_map")
        null_col = CONDITION_IDS.index("matched_null")
        # The short test window still carries the transient at the map
        # switch, so the bound is relative to the incoherent remap rather than
        # the near-baseline level the full-length figure reaches.
        assert np.all(refl < 0.75 * remap)
        assert (
            summary.median_decoding_accuracy[0, col]
            > 5.0 * summary.median_decoding_accuracy[0, null_col]
        )
        assert (
            summary.median_decoding_accuracy[1, col] < summary.median_decoding_accuracy[1, null_col]
        )

    def test_sparse_population_column_flags_kl_far_more_than_the_others(
        self, summary: Figure3RealizationSummary
    ) -> None:
        sparse = _column(summary, "sparse_population")
        null = _column(summary, "matched_null")
        assert sparse[2] > 15.0 and sparse[2] > 2.0 * null[2]
        assert sparse[0] < 15.0 and sparse[1] < 15.0
        assert sparse[2] > 3.0 * max(sparse[0], sparse[1])
        # The predictive region is broader than in the matched null.
        col = CONDITION_IDS.index("sparse_population")
        null_col = CONDITION_IDS.index("matched_null")
        assert (
            summary.median_decoding_accuracy[3, col] > summary.median_decoding_accuracy[3, null_col]
        )
        assert np.any(summary.sparse_cell_event_counts_by_realization > 0)

    def test_history_dependent_column_is_missed_with_matched_rate(
        self, summary: Figure3RealizationSummary
    ) -> None:
        hist = _column(summary, "history_dependent")
        assert np.all(hist < 5.0)
        rates = np.median(summary.phase_spike_rates_by_realization, axis=0)
        hist_rate = rates[CONDITION_IDS.index("history_dependent")]
        null_rate = rates[CONDITION_IDS.index("matched_null")]
        assert abs(hist_rate / null_rate - 1.0) < 0.15

    def test_rejects_nonpositive_realizations(self, calibration: Figure3Calibration) -> None:
        with pytest.raises(ValueError, match="n_realizations"):
            estimate_realization_summary(
                _moderate_params(), calibration=calibration, n_realizations=0
            )


class TestFigure3RealizationSummaryInvariants:
    def test_shape_mismatch_raises(self, summary: Figure3RealizationSummary) -> None:
        import dataclasses

        with pytest.raises(ValueError, match="event_counts_by_realization"):
            dataclasses.replace(summary, event_counts_by_realization=np.zeros((2, 2), dtype=int))
        with pytest.raises(ValueError, match="n_realizations"):
            dataclasses.replace(summary, n_realizations=0)

    def test_thresholds_property_matches_calibration(
        self, summary: Figure3RealizationSummary
    ) -> None:
        assert isinstance(summary.diagnostic_thresholds, DiagnosticThresholds)
        assert summary.diagnostic_thresholds == summary.calibration.diagnostic_thresholds
