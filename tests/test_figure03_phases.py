"""Behavior tests for the figure-3 metric-dissociation phases.

These tests verify the scientific claims of the figure-3 simulation:

- The remap phase diagnostics use the decoder's remapped likelihood,
  rather than an oracle baseline expected-count table.
- In the sparse-population control, isolated spikes from a small population of
  narrow cells clustered at one location elevate KL while HPD overlap and the
  rank-based p-value remain consistent.
- The history-dependent firing phase produces per-spike metrics
  comparable to baseline — i.e., the per-spike spatial diagnostics
  largely *miss* a purely temporal misspecification (the deliberate
  demonstration of metric scope).

If any of these assertions ever flips, the figure no longer tells the
story the paper claims; CI flags the regression.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import statespacecheck as ssc
from numpy.typing import NDArray
from scipy.special import xlogy

from statespacecheck_paper.diagnostics import (
    HPD_COVERAGE,
    METRIC_FLAG_DIRECTIONS,
    DiagnosticThresholds,
    compute_spike_event_diagnostics_from_rates,
    flag_mask,
)
from statespacecheck_paper.figure03_protocol import (
    Figure3Config,
    PhaseBoundary,
    compute_replay_step_window,
)
from statespacecheck_paper.figure03_simulation import (
    Figure3ExpectedCountTables,
    Figure3SimulationResult,
    _single_out_and_back_sweep,
    build_figure03_expected_count_tables,
    build_sparse_population,
    remap_place_field_centers,
    run_figure03_simulation,
    simulate_momentum_phase,
    simulate_sparse_approach_phase,
)
from statespacecheck_paper.figure03_summary import (
    Figure3RealizationSummary,
    _realization_values,
    build_summary_conditions,
    estimate_realization_summary,
    median_standard_error,
)
from statespacecheck_paper.simulation import gaussian_transition_matrix, place_field_expected_counts


def _moderate_params() -> Figure3Config:
    """Phase sizes large enough for per-phase medians to be stable but
    small enough to keep the test fast (~3 s on a laptop).
    """
    return Figure3Config(
        phase_boundaries=(600, 900, 1100, 1400, 1600, 1900, 2100, 3100),
        sparse_approach_duration_steps=100,
    )


def _seed0_params() -> Figure3Config:
    """``_moderate_params`` whose realizations start at seed 0."""
    return dataclasses.replace(_moderate_params(), random_seed=0)


def _per_phase_medians(
    sim: Figure3SimulationResult,
) -> dict[PhaseBoundary, tuple[float, float, float]]:
    """Return (kl_med, hpd_med, sp_med) per phase.

    Each phase is keyed by the boundary that ends it: the opening baseline by
    ``PhaseBoundary.REMAP_START``, the sparse population by ``SPARSE_POP_END``.
    """
    metrics = sim.diagnostics
    boundaries = np.asarray(sim.config.phase_boundaries)
    event_phase = np.searchsorted(boundaries, metrics.event_time_ind, side="right")
    out: dict[PhaseBoundary, tuple[float, float, float]] = {}
    for label in PhaseBoundary:
        mask = event_phase == label
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


def test_phase_boundaries_end_the_timeline(sim: Figure3SimulationResult) -> None:
    """One boundary per phase, and a simulated timeline that ends at the
    SPARSE_POP_END boundary.
    """
    params = sim.config
    assert len(PhaseBoundary) == len(params.phase_boundaries) == 8
    end = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    x_true = np.asarray(sim.physical_position)
    assert x_true.shape[0] == end


def _expected_count_tables(sim: Figure3SimulationResult) -> Figure3ExpectedCountTables:
    """Return the decoder's tables; each has shape ``(n_bins, n_cells)``."""
    return build_figure03_expected_count_tables(
        sim.position_bins,
        sim.config.place_field_centers,
        np.asarray(sim.sparse_place_field_centers),
        sim.config,
    )


def _nearest_bin(sim: Figure3SimulationResult, position: NDArray[np.floating]) -> NDArray[np.intp]:
    """Return the position bin nearest each of ``position``, shape ``(n_time,)``."""
    return np.abs(sim.position_bins[None, :] - position[:, None]).argmin(axis=1)


def _replay_sweep(sim: Figure3SimulationResult) -> NDArray[np.floating]:
    """Rebuild the represented sweep, which starts from the immobile position."""
    config = sim.config
    r0, r1 = compute_replay_step_window(config)
    return _single_out_and_back_sweep(
        float(sim.physical_position[r0]),
        r1 - r0,
        float(config.position_min),
        float(config.position_max),
        config.replay_speed_per_step,
    )


def _assert_count_matches_poisson_total(observed: int, expected: float, label: str) -> None:
    """Check a Poisson total against its mean, within 4.5 standard deviations."""
    assert abs(observed - expected) <= 4.5 * np.sqrt(max(expected, 1.0)), (
        f"{label}: observed {observed} spikes, the decoder expects {expected:.1f}"
    )


def test_decoder_updates_with_the_declared_table_on_every_step(
    sim: Figure3SimulationResult,
) -> None:
    """Every step's stored update is the Poisson update under its phase's table.

    ``posterior[t]`` is ``predictive[t]`` times the likelihood of that step's
    spikes, so the stored outputs reveal the table the decoder actually used. A
    window that is missing, starts or ends late, or reaches only the diagnostics
    fails here, whichever tables those windows were configured with.
    """
    config = sim.config
    bounds = config.phase_boundaries
    tables = _expected_count_tables(sim)
    remap = (bounds[PhaseBoundary.REMAP_START], bounds[PhaseBoundary.REMAP_END])
    replay = compute_replay_step_window(config)
    sparse = (bounds[PhaseBoundary.RECOVERY3_END], bounds[PhaseBoundary.SPARSE_POP_END])
    windows = (
        (0, remap[0], tables.baseline_expected_counts_per_step),
        (*remap, tables.remapped_expected_counts_per_step),
        (remap[1], replay[0], tables.baseline_expected_counts_per_step),
        (*replay, tables.replay_expected_counts_per_step),
        (replay[1], sparse[0], tables.baseline_expected_counts_per_step),
        (*sparse, tables.sparse_population_expected_counts_per_step),
    )
    for start, end, table in windows:
        log_likelihood = xlogy(sim.spike_counts[start:end, None, :], table).sum(axis=2) - table.sum(
            axis=1
        )
        weights = sim.diagnostics.predictive[start:end] * np.exp(
            log_likelihood - log_likelihood.max(axis=1, keepdims=True)
        )
        np.testing.assert_allclose(
            sim.diagnostics.posterior[start:end],
            weights / weights.sum(axis=1, keepdims=True),
            atol=1e-10,
            err_msg=f"steps [{start}, {end})",
        )


def test_prediction_is_the_declared_walk_on_every_step(sim: Figure3SimulationResult) -> None:
    """No phase changes the decoder's transition: each prediction is the walk
    applied to the preceding posterior."""
    transition = gaussian_transition_matrix(sim.position_bins, sim.config.prediction_step_std)
    expected = sim.diagnostics.posterior[:-1] @ transition.T
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(sim.diagnostics.predictive[1:], expected, atol=1e-12)


def test_spikes_fire_at_the_rates_the_decoder_declares(sim: Figure3SimulationResult) -> None:
    """Spike totals match the decoder's table at the generating position.

    This compares the simulated spikes, not the configuration, so a control
    whose generator drifts from the decoder's rates fails here. The remap window
    is included with the baseline table because its spikes stay ordinary; only
    the decoder is wrong there. The history-dependent window is left out because
    its spikes deliberately do not follow the table.
    """
    config = sim.config
    bounds = config.phase_boundaries
    tables = _expected_count_tables(sim)
    n_ordinary = len(config.place_field_centers)
    replay = compute_replay_step_window(config)
    sparse = (bounds[PhaseBoundary.RECOVERY3_END], bounds[PhaseBoundary.SPARSE_POP_END])
    baseline = tables.baseline_expected_counts_per_step
    position = sim.physical_position
    sweep = _replay_sweep(sim)
    windows = (
        ("baseline", 0, bounds[PhaseBoundary.RECOVERY1_END], baseline, position),
        ("recovery 2", bounds[PhaseBoundary.HIST_DEP_END], replay[0], baseline, position),
        ("replay", *replay, tables.replay_expected_counts_per_step, None),
        ("momentum", replay[1], sparse[0], baseline, position),
        ("sparse", *sparse, tables.sparse_population_expected_counts_per_step, position),
    )
    for label, start, end, table, generating in windows:
        bins = _nearest_bin(sim, sweep if generating is None else generating[start:end])
        expected = table[bins, :n_ordinary].sum()
        observed = int(sim.spike_counts[start:end, :n_ordinary].sum())
        _assert_count_matches_poisson_total(observed, float(expected), label)


def test_sparse_cells_fire_at_the_rates_the_decoder_declares() -> None:
    """Sparse-cell spikes match both decoder tables, at their baseline gain before
    the window and at the full rate within it.

    The sparse rates are too low to measure within one realization, so a long
    stay at the sparse location gives the generator enough spikes.
    """
    config = Figure3Config()
    n_baseline, n_active = 400_000, 200_000
    position = np.full(n_baseline + n_active, config.sparse_position)
    spikes, centers = build_sparse_population(
        position, config, 0, n_baseline, n_baseline + n_active
    )
    tables = build_figure03_expected_count_tables(
        config.position_bins, config.place_field_centers, centers, config
    )
    n_ordinary = len(config.place_field_centers)
    position_bin = int(np.abs(config.position_bins - config.sparse_position).argmin())

    for label, count, table, window in (
        ("baseline", n_baseline, tables.baseline_expected_counts_per_step, spikes[:n_baseline]),
        (
            "active",
            n_active,
            tables.sparse_population_expected_counts_per_step,
            spikes[n_baseline:],
        ),
    ):
        expected = count * table[position_bin, n_ordinary:].sum()
        _assert_count_matches_poisson_total(int(window.sum()), float(expected), label)


def test_clean_phases_walk_at_the_decoders_step_size(sim: Figure3SimulationResult) -> None:
    """The well-specified windows' trajectory has the decoder's step size."""
    config = sim.config
    bounds = config.phase_boundaries
    replay_start, _ = compute_replay_step_window(config)
    increments = np.concatenate(
        [
            np.diff(sim.physical_position[start:end])
            for start, end in (
                (0, bounds[PhaseBoundary.RECOVERY1_END]),
                (bounds[PhaseBoundary.HIST_DEP_END], replay_start),
            )
        ]
    )
    assert increments.mean() == pytest.approx(0.0, abs=0.1)
    assert increments.std() == pytest.approx(config.prediction_step_std, rel=0.1)


def test_each_phase_starts_where_the_last_one_ended(sim: Figure3SimulationResult) -> None:
    """Crossing a phase boundary is one ordinary step, not a jump."""
    position = sim.physical_position
    boundaries = np.asarray(sim.config.phase_boundaries[:-1])
    jumps = np.abs(position[boundaries] - position[boundaries - 1])
    assert np.all(jumps <= 5.0 * sim.config.prediction_step_std), f"got {jumps}"


def test_history_phase_bursts_and_suppresses_spikes(sim: Figure3SimulationResult) -> None:
    """The misfit is present: a spike is followed by silence, then a burst.

    For each spike, compares the same cell's spikes in the following steps with
    the rates the decoder expects there; the "missed by the diagnostics" tests
    would pass trivially if the phase were ordinary Poisson firing.
    """
    config = sim.config
    bounds = config.phase_boundaries
    n_ordinary = len(config.place_field_centers)
    expected_counts = _expected_count_tables(sim).baseline_expected_counts_per_step[
        _nearest_bin(sim, sim.physical_position)
    ]
    burst_start, burst_end = config.history_burst_window

    def following_spikes(start: int, end: int, lag_start: int, lag_end: int) -> tuple[int, float]:
        observed, expected = 0, 0.0
        for time_ind, cell_ind in np.argwhere(sim.spike_counts[start:end, :n_ordinary]):
            following = np.arange(start + time_ind + lag_start, start + time_ind + lag_end + 1)
            following = following[following < end]
            observed += int(sim.spike_counts[following, cell_ind].sum())
            expected += float(expected_counts[following, cell_ind].sum())
        return observed, expected

    history = (bounds[PhaseBoundary.RECOVERY1_END], bounds[PhaseBoundary.HIST_DEP_END])
    baseline = (0, bounds[PhaseBoundary.REMAP_START])

    observed, expected = following_spikes(*history, burst_start, burst_end)
    assert observed > 1.5 * expected, f"history burst: {observed} vs expected {expected:.1f}"
    observed, expected = following_spikes(*baseline, burst_start, burst_end)
    assert observed < 1.25 * expected, f"baseline burst: {observed} vs expected {expected:.1f}"
    observed, expected = following_spikes(*history, 1, config.history_refractory_steps)
    assert expected > 10.0
    assert observed == 0, f"history refractory: {observed} spikes, {expected:.1f} expected"


def test_momentum_phase_has_persistent_velocity(sim: Figure3SimulationResult) -> None:
    """The misfit is present: momentum steps are correlated, unlike the decoder's walk."""
    bounds = sim.config.phase_boundaries
    position = sim.physical_position

    def lag_one_correlation(start: int, end: int) -> float:
        steps = np.diff(position[start:end])
        return float(np.corrcoef(steps[:-1], steps[1:])[0, 1])

    momentum = lag_one_correlation(
        bounds[PhaseBoundary.RECOVERY2_END], bounds[PhaseBoundary.MOMENTUM_END]
    )
    baseline = lag_one_correlation(0, bounds[PhaseBoundary.REMAP_START])

    assert momentum > 0.7, f"momentum steps should persist; got lag-1 correlation {momentum:.2f}"
    assert abs(baseline) < 0.15, f"baseline steps should not persist; got {baseline:.2f}"


def test_replay_decoded_position_leaves_the_immobile_animal(sim: Figure3SimulationResult) -> None:
    """The animal stays still while the decoder follows the sweep away from it."""
    r0, r1 = compute_replay_step_window(sim.config)
    physical = sim.physical_position[r0:r1]
    posterior_mean = sim.diagnostics.posterior[r0:r1] @ sim.position_bins
    sweep = _replay_sweep(sim)

    np.testing.assert_allclose(physical, physical[0])
    assert np.max(np.abs(posterior_mean - physical)) > 10.0
    assert np.median(np.abs(posterior_mean - sweep)) < np.median(np.abs(posterior_mean - physical))


@pytest.mark.parametrize("initial_position", [0.0, 50.0])
def test_trajectory_starts_from_the_configured_initial_position(initial_position: float) -> None:
    """The first sample is one random-walk increment away from ``initial_position``."""
    config = Figure3Config(
        phase_boundaries=(200, 300, 400, 500, 600, 700, 800, 900),
        initial_position=initial_position,
    )
    first = float(run_figure03_simulation(config, seed=0).physical_position[0])

    # Reflection at the track ends keeps |step| as the distance from x_0.
    assert abs(first - initial_position) <= 5.0 * config.prediction_step_std


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
    """Replay's observation model is correctly specified.

    The generative sweep spikes fire at the elevated ``replay_place_field_rate_scale``
    through the ordinary position-tuning model; the decoder's replay-window
    expected-count table must use the *same* tuning model (centers, width) and the
    *same* elevated scale. The invariant is this parameterization
    equivalence, not array equality of the continuous sweep evaluation
    against the grid expected-count table. When both match, the window carries no
    observation misfit; the decoder's random-walk transition still differs
    from the deterministic sweep.
    """
    params = sim.config
    assert params.place_field_centers is not None
    expected_count_tables = build_figure03_expected_count_tables(
        sim.position_bins,
        params.place_field_centers,
        np.asarray(sim.sparse_place_field_centers),
        params,
    )
    n_normal = len(params.place_field_centers)
    decoder_replay_normal = expected_count_tables.replay_expected_counts_per_step[:, :n_normal]
    generative_replay = place_field_expected_counts(
        sim.position_bins,
        params.place_field_centers,
        params.place_field_std,
        params.replay_place_field_rate_scale,
    )
    np.testing.assert_allclose(decoder_replay_normal, generative_replay)
    # The elevated replay scale is genuinely above the ordinary rate scale,
    # so this equivalence is not vacuously true at the baseline rate.
    baseline_normal = place_field_expected_counts(
        sim.position_bins,
        params.place_field_centers,
        params.place_field_std,
        params.place_field_rate_scale,
    )
    assert params.replay_place_field_rate_scale > params.place_field_rate_scale
    assert decoder_replay_normal.max() > baseline_normal.max()


def test_remap_phase_uses_decoder_likelihood(sim: Figure3SimulationResult) -> None:
    """Remap diagnostics must use the same remapped rates as the decoder.

    This guards against combining the decoder's predictive distribution
    with unavailable baseline/oracle place fields during the remap window.
    """
    params = sim.config
    assert params.place_field_centers is not None
    start = params.phase_boundaries[PhaseBoundary.REMAP_START]
    end = params.phase_boundaries[PhaseBoundary.REMAP_END]
    in_window = (sim.diagnostics.event_time_ind >= start) & (sim.diagnostics.event_time_ind < end)
    assert in_window.any(), "test simulation produced no remap-window spike events"

    remapped_normal_rates = place_field_expected_counts(
        sim.position_bins,
        remap_place_field_centers(
            params.place_field_centers, params.place_field_remapping, active=True
        ),
        params.place_field_std,
        params.place_field_rate_scale,
    )
    sparse_scale = (
        params.sparse_cell_peak_rate_per_step * np.sqrt(2.0 * np.pi) * params.sparse_place_field_std
    )
    baseline_sparse_expected_counts = (
        params.sparse_cell_baseline_rate_fraction
        * place_field_expected_counts(
            sim.position_bins,
            np.asarray(sim.sparse_place_field_centers),
            params.sparse_place_field_std,
            sparse_scale,
        )
    )
    remapped_expected_counts_per_step = np.hstack(
        [remapped_normal_rates, baseline_sparse_expected_counts]
    )
    expected = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        remapped_expected_counts_per_step,
        sim.diagnostics.event_time_ind[in_window],
        sim.diagnostics.event_cell_ind[in_window],
    )
    assert expected.event_likelihood is not None

    np.testing.assert_allclose(
        sim.diagnostics.event_likelihood[in_window],
        expected.event_likelihood,
    )
    predictive = sim.diagnostics.predictive[sim.diagnostics.event_time_ind[in_window]]
    np.testing.assert_allclose(
        sim.diagnostics.event_hpd_overlap[in_window],
        ssc.hpd_overlap(
            predictive,
            sim.diagnostics.event_likelihood[in_window],
            coverage=HPD_COVERAGE,
        ),
        err_msg="remap HPD was not computed from the displayed event likelihood",
    )
    np.testing.assert_allclose(
        sim.diagnostics.event_kl_divergence[in_window],
        ssc.kl_divergence(
            predictive,
            sim.diagnostics.event_likelihood[in_window],
        ),
        err_msg="remap KL was not computed from the displayed event likelihood",
    )
    for name in (
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_predictive_pvalue",
    ):
        np.testing.assert_allclose(
            getattr(sim.diagnostics, name)[in_window],
            getattr(expected, name),
            err_msg=f"remap-window {name} did not use decoder rates",
        )

    # Confirm that this fixture distinguishes decoder-rate diagnostics from
    # an oracle computation with the unperturbed place fields, which the
    # decoder does not use during the remap.
    baseline_rates = np.hstack(
        [
            place_field_expected_counts(
                sim.position_bins,
                params.place_field_centers,
                params.place_field_std,
                params.place_field_rate_scale,
            ),
            baseline_sparse_expected_counts,
        ]
    )
    oracle = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        baseline_rates,
        sim.diagnostics.event_time_ind[in_window],
        sim.diagnostics.event_cell_ind[in_window],
    )
    for name in (
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_predictive_pvalue",
    ):
        assert not np.allclose(getattr(expected, name), getattr(oracle, name)), (
            f"test fixture does not distinguish decoder and oracle values for {name}"
        )


def test_sparse_population_dissociates_kl_from_other_metrics() -> None:
    """Load-bearing: isolated sparse-population spikes elevate KL while HPD
    overlap and the predictive p-value remain consistent.

    The phase has about 12 spikes per realization even at this length, too few for
    one realization's median, so the claim is about events pooled over five seeds.
    """
    params = dataclasses.replace(
        _moderate_params(),
        phase_boundaries=(1200, 1500, 1700, 2000, 2200, 2500, 2700, 5700),
    )
    realizations = [run_figure03_simulation(params, seed=seed).diagnostics for seed in range(5)]
    boundaries = np.asarray(params.phase_boundaries)

    def pooled_median(values: str, phase: PhaseBoundary) -> float:
        pooled = [
            getattr(metrics, values)[
                np.searchsorted(boundaries, metrics.event_time_ind, side="right") == phase
            ]
            for metrics in realizations
        ]
        return float(np.nanmedian(np.concatenate(pooled)))

    base_kl = pooled_median("event_kl_divergence", PhaseBoundary.REMAP_START)
    base_hpd = pooled_median("event_hpd_overlap", PhaseBoundary.REMAP_START)
    sparse_kl = pooled_median("event_kl_divergence", PhaseBoundary.SPARSE_POP_END)
    sparse_hpd = pooled_median("event_hpd_overlap", PhaseBoundary.SPARSE_POP_END)
    sparse_p = pooled_median("event_predictive_pvalue", PhaseBoundary.SPARSE_POP_END)

    # Over 40 seeds, any five consecutive seeds give a KL ratio of 3.95-6.67, an
    # HPD ratio of 1.00, and a median p-value of 0.40-0.59.
    assert sparse_kl > 3 * base_kl, (
        "sparse-population spikes should inflate KL by >3x; "
        f"got base={base_kl:.3f}, sparse={sparse_kl:.3f}"
    )
    assert sparse_hpd >= 0.9 * base_hpd, (
        "sparse-population spikes should preserve HPD overlap; "
        f"got base={base_hpd:.3f}, sparse={sparse_hpd:.3f}"
    )
    # The rank-based p-value stays consistent: well clear of the 0.05 flag
    # threshold (the panel-(b) column test pins the ~0% flag rate directly).
    assert sparse_p > 0.2, (
        "sparse-population rank-based p-values should stay well above the 0.05 "
        f"flag threshold; got {sparse_p:.3f}"
    )


def test_sparse_population_is_a_correctly_modeled_low_activity_regime(
    sim: Figure3SimulationResult,
) -> None:
    """The last phase is a fixed immobile stop, not a transition perturbation.

    The ordinary ensemble is quiet, the sparse population fires, and both its
    likelihood and the decoder prediction use the declared model. This pins
    the KL dissociation to sparse information rather than hidden mismatch.
    """
    params = sim.config
    w0 = params.phase_boundaries[PhaseBoundary.RECOVERY3_END]
    w1 = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    n_sparse = len(sim.sparse_place_field_centers)
    n_normal = sim.spike_counts.shape[1] - n_sparse
    np.testing.assert_allclose(sim.physical_position[w0:w1], params.sparse_position)
    # The ordinary ensemble is silent; only the sparse-population cells fire.
    assert sim.spike_counts[w0:w1, :n_normal].sum() == 0
    assert sim.spike_counts[w0:w1, n_normal:].sum() > 0
    assert n_sparse == params.sparse_cell_count

    in_window = (sim.diagnostics.event_time_ind >= w0) & (sim.diagnostics.event_time_ind < w1)
    assert in_window.any(), "the sparse population produced no sparse-window diagnostic events"
    assert np.all(sim.diagnostics.event_cell_ind[in_window] >= n_normal)

    sparse_scale = (
        params.sparse_cell_peak_rate_per_step * np.sqrt(2.0 * np.pi) * params.sparse_place_field_std
    )
    sparse_population_expected_counts_per_step = place_field_expected_counts(
        sim.position_bins,
        np.asarray(sim.sparse_place_field_centers),
        params.sparse_place_field_std,
        sparse_scale,
    )
    expected = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        sparse_population_expected_counts_per_step,
        sim.diagnostics.event_time_ind[in_window],
        # Re-index the sparse-cell columns to 0..n_sparse-1 for the K-column
        # sparse expected-count table.
        (sim.diagnostics.event_cell_ind[in_window] - n_normal).astype(np.intp),
    )
    np.testing.assert_allclose(
        sim.diagnostics.event_likelihood[in_window],
        expected.event_likelihood,
    )

    # No phase-specific transition is introduced: the stored prediction is
    # exactly the standard transition applied to the preceding posterior.
    event_time = int(sim.diagnostics.event_time_ind[np.flatnonzero(in_window)[0]])
    transition = gaussian_transition_matrix(sim.position_bins, params.prediction_step_std)
    expected_predictive = transition @ sim.diagnostics.posterior[event_time - 1]
    expected_predictive /= expected_predictive.sum()
    np.testing.assert_allclose(
        sim.diagnostics.predictive[event_time],
        expected_predictive,
    )

    # Parameterization equivalence: the decoder's sparse expected-count tables use
    # the same centers/width/scale as the generative model
    # (``sparse_population_expected_counts_per_step``),
    # with the declared baseline gain below the window and the full elevated
    # rate within it. The gain is intentionally nonzero, so this is a
    # rate-regime check (low baseline vs elevated), not a zero-count check.
    expected_count_tables = build_figure03_expected_count_tables(
        sim.position_bins,
        params.place_field_centers,
        np.asarray(sim.sparse_place_field_centers),
        params,
    )
    decoder_elevated_sparse = expected_count_tables.sparse_population_expected_counts_per_step[
        :, n_normal:
    ]
    decoder_baseline_sparse = expected_count_tables.baseline_expected_counts_per_step[:, n_normal:]
    np.testing.assert_allclose(decoder_elevated_sparse, sparse_population_expected_counts_per_step)
    np.testing.assert_allclose(
        decoder_baseline_sparse,
        params.sparse_cell_baseline_rate_fraction * sparse_population_expected_counts_per_step,
    )
    # Rate regime: the baseline gain is < 1, so out-of-window rates are
    # uniformly below the elevated in-window peak.
    assert params.sparse_cell_baseline_rate_fraction < 1.0
    peak = float(sparse_population_expected_counts_per_step.max())
    assert float(decoder_baseline_sparse.max()) < peak
    assert float(decoder_elevated_sparse.max()) == pytest.approx(peak)


def test_history_dependent_firing_per_spike_metrics_near_baseline(
    sim: Figure3SimulationResult,
) -> None:
    """Load-bearing scientific claim: per-spike spatial diagnostics
    largely *miss* temporal (history-dependent) misspecification.

    The bursting + refractory misfit lives in the joint distribution of
    spike trains, not in any individual spike's spatial likelihood. We
    therefore expect the per-spike metrics in the history-dependent
    phase to stay close to baseline rather than crossing the flagging
    thresholds the way remap/momentum do.
    """
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, base_sp = medians[PhaseBoundary.REMAP_START]
    hd_kl, hd_hpd, hd_sp = medians[PhaseBoundary.HIST_DEP_END]

    # All three per-spike metrics stay near baseline — the temporal
    # misfit barely registers. Bounds are absolute (vs. baseline), not
    # relative to the remap collapse: the claim is "near baseline", and
    # a midpoint-vs-remap bound would be trivially satisfied because the
    # remap inflation is ~30x.
    assert hd_kl < 2 * base_kl, (
        "per-spike KL in the history-dependent phase should stay within "
        f"2x baseline; got baseline={base_kl:.3f}, hist-dep={hd_kl:.3f}"
    )
    assert hd_hpd > 0.9 * base_hpd, (
        "per-spike HPD overlap in the history-dependent phase should stay within "
        f"10% of baseline; got baseline={base_hpd:.3f}, hist-dep={hd_hpd:.3f}"
    )
    # predictive_pvalue stays in a band around baseline (neither collapsing like
    # remap nor spuriously inflating).
    assert 0.5 * base_sp < hd_sp < 1.5 * base_sp, (
        "per-spike predictive_pvalue in the history-dependent phase should stay "
        f"within +/-50% of baseline; got baseline={base_sp:.3f}, "
        f"hist-dep={hd_sp:.3f}"
    )


def test_momentum_phase_inflates_kl() -> None:
    """The momentum misfit (persistent-velocity trajectory vs. memoryless decoder)
    raises the typical per-spike KL above baseline. One realization does not show
    this at test scale, where a single seed's ratio can fall below 1, so the claim
    is about the median over several seeds, with longer baseline and momentum phases.
    """
    params = dataclasses.replace(
        _moderate_params(),
        phase_boundaries=(1200, 1500, 1700, 2000, 2200, 3100, 3300, 4300),
    )
    ratios = []
    for seed in range(5):
        medians = _per_phase_medians(run_figure03_simulation(params, seed=seed))
        ratios.append(
            medians[PhaseBoundary.MOMENTUM_END][0] / medians[PhaseBoundary.REMAP_START][0]
        )
    # Over 40 seeds, the median of any five consecutive seeds spans 1.32-1.76 for
    # this momentum window and 0.94-1.31 with no momentum (1.15 for these seeds), so the
    # bound catches a momentum window that has become a plain walk.
    assert np.median(ratios) > 1.25, f"momentum/baseline KL ratios: {np.round(ratios, 2)}"


def test_simulate_momentum_phase() -> None:
    """The momentum trajectory starts at ``x_last``, stays on the track, and is
    reproducible under a fixed seed."""
    params = Figure3Config()
    x = simulate_momentum_phase(500, x_last=40.0, config=params, rng=np.random.default_rng(0))

    assert x.shape == (500,)
    assert x[0] == pytest.approx(40.0)
    assert np.all(x >= params.position_min) and np.all(x <= params.position_max)

    repeat = simulate_momentum_phase(500, x_last=40.0, config=params, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(x, repeat)


def test_simulate_sparse_approach_phase() -> None:
    """The approach ends exactly at ``params.sparse_position``, stays on the
    track, and is reproducible under a fixed seed."""
    params = Figure3Config()  # sparse_approach_duration_steps=1000
    n = 1500
    x = simulate_sparse_approach_phase(n, x_last=10.0, config=params, rng=np.random.default_rng(0))

    assert x.shape == (n,)
    assert x[-1] == pytest.approx(params.sparse_position)
    assert np.all(x >= params.position_min) and np.all(x <= params.position_max)

    repeat = simulate_sparse_approach_phase(
        n, x_last=10.0, config=params, rng=np.random.default_rng(0)
    )
    np.testing.assert_array_equal(x, repeat)


# ---------------------------------------------------------------------------
# estimate_realization_summary: pooled thresholds + median per-phase fractions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def summary() -> Figure3RealizationSummary:
    """Five pooled realizations, shared by the scientific-claim guards below."""
    return estimate_realization_summary(_seed0_params(), n_realizations=5)


class TestEstimateRealizationSummary:
    def test_shapes(self, summary: Figure3RealizationSummary) -> None:
        """The summary is (3 metrics x 6 columns: well-specified, remap,
        history, replay, momentum, sparse population) and fractions are percentages."""
        assert isinstance(summary, Figure3RealizationSummary)
        assert summary.n_realizations == 5
        assert summary.median_flag_percentages.shape == (3, 6)
        assert summary.median_decoding_error.shape == (6,)
        assert np.all(summary.median_decoding_error >= 0.0)
        # Percentages in [0, 100].
        assert np.all(summary.median_flag_percentages >= 0.0)
        assert np.all(summary.median_flag_percentages <= 100.0)
        # Threshold is a valid, finite DiagnosticThresholds with the fixed p-value cutoff.
        assert summary.diagnostic_thresholds.predictive_pvalue == 0.05
        assert np.isfinite(summary.diagnostic_thresholds.kl_divergence)

    def test_same_seeds_reproduce_the_same_result_at_any_worker_count(self) -> None:
        params = _seed0_params()
        first = estimate_realization_summary(params, n_realizations=2, n_jobs=1)
        repeat = estimate_realization_summary(params, n_realizations=2, n_jobs=2)
        np.testing.assert_array_equal(
            first.realization_flag_percentages, repeat.realization_flag_percentages
        )
        np.testing.assert_array_equal(
            first.realization_decoding_error, repeat.realization_decoding_error
        )
        assert first.diagnostic_thresholds == repeat.diagnostic_thresholds

    def test_realizations_are_stored_in_seed_order(self) -> None:
        """Row i of the published per-realization arrays is seed ``first + i``."""
        params = _seed0_params()
        pooled = estimate_realization_summary(params, n_realizations=2, n_jobs=1)
        second_alone = estimate_realization_summary(
            dataclasses.replace(params, random_seed=1), n_realizations=1, n_jobs=1
        )
        np.testing.assert_array_equal(
            pooled.realization_decoding_error[1], second_alone.realization_decoding_error[0]
        )

    def test_baseline_pool_stops_before_the_first_remap_step(self) -> None:
        """The first remap step's spikes are not baseline events."""
        params = Figure3Config(phase_boundaries=(200, 300, 400, 500, 600, 700, 800, 900))
        baseline_end = params.phase_boundaries[PhaseBoundary.REMAP_START]
        seed, diagnostics = next(
            (seed, diagnostics)
            for seed in range(20)
            if np.any(
                (
                    diagnostics := run_figure03_simulation(params, seed=seed).diagnostics
                ).event_time_ind
                == baseline_end
            )
        )

        values = _realization_values(params, seed, build_summary_conditions(params))

        expected = int(np.sum(diagnostics.event_time_ind < baseline_end))
        assert values.baseline_values["kl_divergence"].size == expected

    def test_baseline_flagged_fractions_score_the_pooled_baseline(self) -> None:
        """Each fraction is the share of pooled opening-baseline events flagged.

        Flags are inclusive, so every event tied at a threshold counts; this
        recomputes the fraction from the realizations' baseline events.
        """
        params = _seed0_params()
        summary = estimate_realization_summary(params, n_realizations=2)
        baseline_end = params.phase_boundaries[PhaseBoundary.REMAP_START]
        diagnostics = [run_figure03_simulation(params, seed=seed).diagnostics for seed in (0, 1)]
        for metric, direction in METRIC_FLAG_DIRECTIONS.items():
            pooled = np.concatenate(
                [
                    np.asarray(getattr(d, "event_" + metric))[d.event_time_ind < baseline_end]
                    for d in diagnostics
                ]
            )
            threshold = getattr(summary.diagnostic_thresholds, metric)
            expected = float(np.mean(flag_mask(pooled, threshold, direction)))
            assert summary.baseline_flagged_fractions[metric] == expected

    def test_remap_column_is_most_flagged(self, summary: Figure3RealizationSummary) -> None:
        """Scientific regression guard: across realizations, the remap column
        (index 1) is flagged far more than the well-specified column (index 0)
        for every metric — the headline 'all three detect remap' result, on
        the median across realizations."""
        for row in range(3):
            assert summary.median_flag_percentages[row, 1] > summary.median_flag_percentages[row, 0]

    def test_replay_is_not_flagged(self, summary: Figure3RealizationSummary) -> None:
        """Scientific claim: the replay event (column 3) is *not* a
        misspecification. The decoder tracks the swept trajectory, so every
        metric stays low — far below the remap positive control — even though
        the decoded position departs from the (fixed) physical position."""
        replay = summary.median_flag_percentages[:, 3]
        remap = summary.median_flag_percentages[:, 1]
        assert np.all(replay < 15.0), f"replay should stay low; got {replay}"
        assert np.all(replay < 0.5 * remap), (
            f"replay must flag far less than the remap misfit; got replay={replay}, remap={remap}"
        )

    def test_sparse_population_column_flags_kl_only(
        self, summary: Figure3RealizationSummary
    ) -> None:
        """Headline panel-(b) claim, guarded on the flag-fraction columns the
        figure actually shows (rows HPD, predictive-p, KL): the sparse-
        population control (column 5) elevates KL well above the
        well-specified baseline while HPD overlap and the rank-based
        predictive p-value flag ~no spikes. A threshold-calibration
        regression that started flagging HPD/p there, or dropped the KL rate,
        would fail here even though the per-event-median guard stays green.
        """
        sparse = summary.median_flag_percentages[:, 5]  # [HPD, predictive-p, KL]
        well = summary.median_flag_percentages[:, 0]
        assert sparse[2] > 15.0, f"sparse-population KL should be clearly elevated; got {sparse[2]}"
        assert sparse[2] > 2.0 * well[2], (
            f"sparse-population KL should exceed 2x the baseline; "
            f"got KL={sparse[2]}, well_KL={well[2]}"
        )
        assert sparse[0] < 2.0 and sparse[1] < 2.0, (
            "HPD overlap and predictive-p must not flag the sparse-population control; "
            f"got {sparse}"
        )

    def test_history_dependent_column_is_missed(self, summary: Figure3RealizationSummary) -> None:
        """Panel-(b) guard: the history-dependent (temporal) misfit is missed
        by all three per-spike spatial diagnostics (column 2 stays low on the
        flag-fraction columns, not merely at the per-event median)."""
        hist = summary.median_flag_percentages[:, 2]
        well = summary.median_flag_percentages[:, 0]
        assert np.all(hist < 5.0), f"history-dependent phase should stay near zero; got {hist}"
        assert np.all(hist <= well), (
            f"history-dependent flags should not exceed the baseline; got hist={hist}, well={well}"
        )

    def test_remap_is_strongly_flagged_by_all_three(
        self, summary: Figure3RealizationSummary
    ) -> None:
        """Magnitude guard: the incoherent random-remap is
        the headline positive control, so every metric must flag it well
        above both the well-specified baseline and the momentum misfit.

        A near-noop remap, or a spatially *coherent* remap (e.g. a pure
        reflection) — which is self-consistent under the decoder's own
        likelihood and correctly undetectable — would fail here. The
        percentages are smaller than the full-length figure (~37--43%)
        because ``_moderate_params`` uses short windows the trajectory only
        partly explores; bounds are set against the observed deterministic
        values (remap ~[31, 35, 24]% vs well ~[10, 11, 7]%, momentum ~[10, 14, 7]%).
        """
        well = summary.median_flag_percentages[:, 0]
        remap = summary.median_flag_percentages[:, 1]
        momentum = summary.median_flag_percentages[:, 4]
        assert np.all(remap > 10.0), f"remap should flag >10% for every metric; got {remap}"
        assert np.all(remap > 2.0 * well), (
            f"remap should flag >2x the well-specified baseline; got remap={remap}, well={well}"
        )
        assert np.all(remap > momentum), (
            "remap should flag more than momentum for every metric; "
            f"got remap={remap}, momentum={momentum}"
        )

    def test_rejects_nonpositive_realizations(self) -> None:
        with pytest.raises(ValueError, match="n_realizations"):
            estimate_realization_summary(_moderate_params(), n_realizations=0)


class TestFigure3RealizationSummaryInvariants:
    @staticmethod
    def _thresholds() -> DiagnosticThresholds:
        return DiagnosticThresholds(hpd_overlap=0.5, kl_divergence=1.0, predictive_pvalue=0.05)

    @staticmethod
    def _fractions() -> dict[str, float]:
        return {"hpd_overlap": 0.02, "predictive_pvalue": 0.05, "kl_divergence": 0.01}

    @pytest.mark.parametrize(
        "fractions",
        [
            {"hpd_overlap": 0.02, "predictive_pvalue": 0.05},
            {"hpd_overlap": 0.02, "predictive_pvalue": 0.05, "kl_divergence": 1.5},
            {"hpd_overlap": -0.1, "predictive_pvalue": 0.05, "kl_divergence": 0.01},
        ],
    )
    def test_invalid_baseline_flagged_fractions_raise(self, fractions: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="baseline_flagged_fractions"):
            Figure3RealizationSummary(
                diagnostic_thresholds=self._thresholds(),
                baseline_flagged_fractions=fractions,
                realization_flag_percentages=np.zeros((2, 3, 5)),
                realization_decoding_error=np.zeros((2, 5)),
            )

    def test_non_3d_realizations_raise(self) -> None:
        with pytest.raises(ValueError, match="realization_flag_percentages"):
            Figure3RealizationSummary(
                diagnostic_thresholds=self._thresholds(),
                baseline_flagged_fractions=self._fractions(),
                realization_flag_percentages=np.zeros((3, 5)),
                realization_decoding_error=np.zeros(5),
            )

    def test_error_column_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="realization_decoding_error"):
            Figure3RealizationSummary(
                diagnostic_thresholds=self._thresholds(),
                baseline_flagged_fractions=self._fractions(),
                realization_flag_percentages=np.zeros((2, 3, 5)),
                realization_decoding_error=np.zeros((2, 4)),
            )

    def test_error_realization_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="realization_decoding_error"):
            Figure3RealizationSummary(
                diagnostic_thresholds=self._thresholds(),
                baseline_flagged_fractions=self._fractions(),
                realization_flag_percentages=np.zeros((2, 3, 5)),
                realization_decoding_error=np.zeros((3, 5)),
            )

    def test_no_realizations_raises(self) -> None:
        with pytest.raises(ValueError, match="n_realizations"):
            Figure3RealizationSummary(
                diagnostic_thresholds=self._thresholds(),
                baseline_flagged_fractions=self._fractions(),
                realization_flag_percentages=np.zeros((0, 3, 5)),
                realization_decoding_error=np.zeros((0, 5)),
            )

    def test_medians_and_errors_come_from_the_realizations(self) -> None:
        """The summary keeps every realization; its medians cannot disagree with them."""
        rng = np.random.default_rng(0)
        flags = rng.uniform(0.0, 100.0, size=(7, 3, 5))
        decoding_error = rng.uniform(0.0, 10.0, size=(7, 5))
        summary = Figure3RealizationSummary(
            diagnostic_thresholds=self._thresholds(),
            baseline_flagged_fractions=self._fractions(),
            realization_flag_percentages=flags,
            realization_decoding_error=decoding_error,
        )
        assert summary.n_realizations == 7
        np.testing.assert_array_equal(summary.median_flag_percentages, np.median(flags, axis=0))
        np.testing.assert_array_equal(
            summary.median_decoding_error, np.median(decoding_error, axis=0)
        )
        np.testing.assert_array_equal(
            summary.flag_percentage_standard_errors, median_standard_error(flags)
        )
        np.testing.assert_array_equal(
            summary.decoding_error_standard_errors, median_standard_error(decoding_error)
        )
        assert not summary.realization_flag_percentages.flags.writeable
