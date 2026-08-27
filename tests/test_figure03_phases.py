"""Behavior tests for the figure-3 metric-dissociation phases.

These tests verify the scientific claims of the figure-3 simulation:

- The remap phase diagnostics use the decoder's remapped likelihood,
  rather than an oracle baseline rate table.
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

import numpy as np
import pytest
import statespacecheck as ssc

from statespacecheck_paper.diagnostics import (
    DiagnosticThresholds,
    compute_spike_event_diagnostics_from_rates,
)
from statespacecheck_paper.figure03_protocol import (
    PHASE_LABELS,
    Figure3Config,
    PhaseBoundary,
)
from statespacecheck_paper.figure03_simulation import (
    Figure3SimulationResult,
    _single_out_and_back_sweep,
    build_figure03_rate_tables,
    remap_place_field_centers,
    run_figure03_simulation,
    simulate_drift_phase,
    simulate_sparse_approach_phase,
)
from statespacecheck_paper.figure03_summary import (
    Figure3RealizationSummary,
    estimate_realization_summary,
)
from statespacecheck_paper.simulation import gaussian_transition_matrix, place_field_rates


def _moderate_params() -> Figure3Config:
    """Phase sizes large enough for per-phase medians to be stable but
    small enough to keep the test fast (~3 s on a laptop).
    """
    return Figure3Config(
        phase_boundaries=(600, 900, 1100, 1400, 1600, 1900, 2100, 3100),
    )


def _per_phase_medians(
    sim: Figure3SimulationResult,
) -> dict[str, tuple[float, float, float]]:
    """Return (kl_med, hpd_med, sp_med) per phase label."""
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


def test_phase_labels_and_boundaries(sim: Figure3SimulationResult) -> None:
    """``run_figure03_simulation`` emits every canonical phase in order
    and a timeline that ends at the SPARSE_POP_END boundary.
    """
    params = sim.config
    # The simulation must emit exactly the canonical phase set, in order.
    assert sim.phase_labels == PHASE_LABELS
    # Sanity-check the canonical set itself: 8 phases, with each expected
    # non-baseline condition appearing once.
    assert len(PHASE_LABELS) == 8
    for misfit in (
        "Remap Misfit",
        "History-Dependent Firing",
        "Drift Misfit",
        "Sparse Population",
    ):
        assert PHASE_LABELS.count(misfit) == 1
    boundaries = np.asarray(sim.phase_boundaries)
    end = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    assert boundaries[-1] == end
    assert np.all(np.diff(boundaries) > 0)
    x_true = np.asarray(sim.true_position)
    assert x_true.shape[0] == end


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
    """Replay is a correctly-specified observation model.

    The generative sweep spikes fire at the elevated ``replay_place_field_rate_scale``
    through the ordinary position-tuning model; the decoder's replay-window
    rate table must use the *same* tuning model (centers, width) and the
    *same* elevated scale. The invariant is this parameterization
    equivalence, not array equality of the continuous sweep evaluation
    against the grid rate table. When both match, the decoded state simply
    tracks the replayed trajectory and no metric flags a misfit.
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
    # The elevated replay scale is genuinely above the ordinary rate scale,
    # so this equivalence is not vacuously true at the baseline rate.
    baseline_normal = place_field_rates(
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

    remapped_normal_rates = place_field_rates(
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
    baseline_sparse_firing_rates = params.sparse_cell_baseline_rate_fraction * place_field_rates(
        sim.position_bins,
        np.asarray(sim.sparse_place_field_centers),
        params.sparse_place_field_std,
        sparse_scale,
    )
    remapped_firing_rates = np.hstack([remapped_normal_rates, baseline_sparse_firing_rates])
    expected = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        remapped_firing_rates,
        sim.diagnostics.event_time_ind[in_window],
        sim.diagnostics.event_cell_ind[in_window],
    )
    assert expected.per_spike_likelihood is not None

    np.testing.assert_allclose(
        sim.diagnostics.per_spike_likelihood[in_window],
        expected.per_spike_likelihood,
    )
    predictive = sim.diagnostics.predictive[sim.diagnostics.event_time_ind[in_window]]
    np.testing.assert_allclose(
        sim.diagnostics.event_hpd_overlap[in_window],
        ssc.hpd_overlap(
            predictive,
            sim.diagnostics.per_spike_likelihood[in_window],
            coverage=0.95,
        ),
        err_msg="remap HPD was not computed from the displayed event likelihood",
    )
    np.testing.assert_allclose(
        sim.diagnostics.event_kl_divergence[in_window],
        ssc.kl_divergence(
            predictive,
            sim.diagnostics.per_spike_likelihood[in_window],
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
    # the old oracle computation based on the unperturbed place fields.
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
    for name in (
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_predictive_pvalue",
    ):
        assert not np.allclose(getattr(expected, name), getattr(oracle, name)), (
            f"test fixture does not distinguish decoder and oracle values for {name}"
        )


def test_sparse_population_dissociates_kl_from_other_metrics(
    sim: Figure3SimulationResult,
) -> None:
    """Load-bearing: isolated sparse-population spikes elevate KL while HPD
    overlap and the predictive p-value remain consistent.
    """
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, _ = medians["Clean Baseline"]
    sparse_kl, sparse_hpd, sparse_p = medians["Sparse Population"]

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
    np.testing.assert_allclose(sim.true_position[w0:w1], params.sparse_position)
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
    sparse_population_firing_rates = place_field_rates(
        sim.position_bins,
        np.asarray(sim.sparse_place_field_centers),
        params.sparse_place_field_std,
        sparse_scale,
    )
    expected = compute_spike_event_diagnostics_from_rates(
        sim.diagnostics.predictive,
        sparse_population_firing_rates,
        sim.diagnostics.event_time_ind[in_window],
        # Re-index the sparse-cell columns to 0..n_sparse-1 for the K-column
        # sparse rate table.
        (sim.diagnostics.event_cell_ind[in_window] - n_normal).astype(np.intp),
    )
    np.testing.assert_allclose(
        sim.diagnostics.per_spike_likelihood[in_window],
        expected.per_spike_likelihood,
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

    # Parameterization equivalence: the decoder's sparse rate tables use the
    # same centers/width/scale as the generative model (``sparse_population_firing_rates``),
    # with the declared baseline gain below the window and the full elevated
    # rate within it. The gain is intentionally nonzero, so this is a
    # rate-regime check (low baseline vs elevated), not a zero-count check.
    rate_tables = build_figure03_rate_tables(
        sim.position_bins,
        params.place_field_centers,
        np.asarray(sim.sparse_place_field_centers),
        params,
    )
    decoder_elevated_sparse = rate_tables.sparse_population_firing_rates[:, n_normal:]
    decoder_baseline_sparse = rate_tables.baseline_firing_rates[:, n_normal:]
    np.testing.assert_allclose(decoder_elevated_sparse, sparse_population_firing_rates)
    np.testing.assert_allclose(
        decoder_baseline_sparse,
        params.sparse_cell_baseline_rate_fraction * sparse_population_firing_rates,
    )
    np.testing.assert_allclose(rate_tables.baseline_sparse_firing_rates, decoder_baseline_sparse)
    # Rate regime: the baseline gain is < 1, so out-of-window rates are
    # uniformly below the elevated in-window peak.
    assert params.sparse_cell_baseline_rate_fraction < 1.0
    peak = float(sparse_population_firing_rates.max())
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
    thresholds the way remap/drift do.
    """
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, base_sp = medians["Clean Baseline"]
    hd_kl, hd_hpd, hd_sp = medians["History-Dependent Firing"]

    # All three per-spike metrics stay near baseline — the temporal
    # misfit barely registers. Bounds are absolute (vs. baseline), not
    # relative to the remap collapse: the claim is "near baseline", and
    # a midpoint-vs-remap bound would be trivially satisfied because the
    # remap inflation is ~30x.
    assert hd_kl < 3 * base_kl, (
        "per-spike KL in the history-dependent phase should stay within "
        f"3x baseline; got baseline={base_kl:.3f}, hist-dep={hd_kl:.3f}"
    )
    assert hd_hpd > 0.9 * base_hpd, (
        "per-spike HPDO in the history-dependent phase should stay within "
        f"10% of baseline; got baseline={base_hpd:.3f}, hist-dep={hd_hpd:.3f}"
    )
    # predictive_pvalue stays in a band around baseline (neither collapsing like
    # remap nor spuriously inflating).
    assert 0.5 * base_sp < hd_sp < 1.5 * base_sp, (
        "per-spike predictive_pvalue in the history-dependent phase should stay "
        f"within +/-50% of baseline; got baseline={base_sp:.3f}, "
        f"hist-dep={hd_sp:.3f}"
    )


def test_drift_phase_inflates_kl(sim: Figure3SimulationResult) -> None:
    """The drift misfit (persistent-velocity trajectory vs. memoryless
    decoder) must produce a meaningfully larger per-spike KL than
    baseline. With the wiggly phase removed, this and the sparse-reward
    test are the only metric-dissociation regression guards left, so the
    bound is tight enough to catch a near-noop drift.
    """
    medians = _per_phase_medians(sim)
    base_kl, _, _ = medians["Clean Baseline"]
    drift_kl, _, _ = medians["Drift Misfit"]
    # Bound chosen against observed ratio (~1.38x at the moderate test
    # scale) — strict enough to catch a regression where drift becomes
    # indistinguishable from baseline, loose enough to absorb normal
    # seed-to-seed variation. Tighten if a wider session shows clearer
    # separation.
    assert drift_kl > 1.2 * base_kl, (
        f"drift phase should inflate KL by >1.2x baseline; "
        f"got base={base_kl:.3f}, drift={drift_kl:.3f}"
    )


def test_simulate_drift_phase() -> None:
    """The drift trajectory starts at ``x_last``, stays on the track, and is
    reproducible under a fixed seed."""
    params = Figure3Config()
    x = simulate_drift_phase(500, x_last=40.0, config=params, rng=np.random.default_rng(0))

    assert x.shape == (500,)
    assert x[0] == pytest.approx(40.0)
    assert np.all(x >= params.position_min) and np.all(x <= params.position_max)

    repeat = simulate_drift_phase(500, x_last=40.0, config=params, rng=np.random.default_rng(0))
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


class TestEstimateRealizationSummary:
    def test_shapes_and_determinism(self) -> None:
        """The summary is (3 metrics x 6 columns: well-specified, remap,
        history, replay, drift, sparse population), fractions are percentages,
        and the same seeds reproduce the same result."""
        params = _moderate_params()
        summary = estimate_realization_summary(params, n_realizations=3, first_random_seed=0)

        assert isinstance(summary, Figure3RealizationSummary)
        assert summary.n_realizations == 3
        assert summary.median_flag_percentages.shape == (3, 6)
        # Percentages in [0, 100].
        assert np.all(summary.median_flag_percentages >= 0.0)
        assert np.all(summary.median_flag_percentages <= 100.0)
        # Threshold is a valid, finite DiagnosticThresholds with the fixed p-value cutoff.
        assert summary.diagnostic_thresholds.predictive_pvalue == 0.05
        assert np.isfinite(summary.diagnostic_thresholds.kl_divergence)

        repeat = estimate_realization_summary(params, n_realizations=3, first_random_seed=0)
        np.testing.assert_array_equal(
            summary.median_flag_percentages, repeat.median_flag_percentages
        )
        assert (
            summary.diagnostic_thresholds.kl_divergence
            == repeat.diagnostic_thresholds.kl_divergence
        )

    def test_remap_column_is_most_flagged(self) -> None:
        """Scientific regression guard: across realizations, the remap column
        (index 1) is flagged far more than the well-specified column (index 0)
        for every metric — the headline 'all three detect remap' result, now
        on a stabilized median."""
        summary = estimate_realization_summary(
            _moderate_params(), n_realizations=5, first_random_seed=0
        )
        for row in range(3):
            assert summary.median_flag_percentages[row, 1] > summary.median_flag_percentages[row, 0]

    def test_replay_is_not_flagged(self) -> None:
        """Scientific claim: the replay event (column 3) is *not* a
        misspecification. The decoder tracks the swept trajectory, so every
        metric stays low — far below the remap positive control — even though
        the decoded position departs from the (fixed) true position."""
        summary = estimate_realization_summary(
            _moderate_params(), n_realizations=5, first_random_seed=0
        )
        replay = summary.median_flag_percentages[:, 3]
        remap = summary.median_flag_percentages[:, 1]
        assert np.all(replay < 15.0), f"replay should stay low; got {replay}"
        assert np.all(replay < 0.5 * remap), (
            f"replay must flag far less than the remap misfit; got replay={replay}, remap={remap}"
        )

    def test_sparse_population_column_flags_kl_only(self) -> None:
        """Headline panel-(b) claim, guarded on the flag-fraction columns the
        figure actually shows (rows HPD, predictive-p, KL): the sparse-
        population control (column 5) elevates KL well above the
        well-specified baseline while HPD overlap and the rank-based
        predictive p-value flag ~no spikes. A threshold-calibration
        regression that started flagging HPD/p there, or dropped the KL rate,
        would fail here even though the per-event-median guard stays green.
        """
        summary = estimate_realization_summary(
            _moderate_params(), n_realizations=5, first_random_seed=0
        )
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

    def test_history_dependent_column_is_missed(self) -> None:
        """Panel-(b) guard: the history-dependent (temporal) misfit is missed
        by all three per-spike spatial diagnostics (column 2 stays low on the
        flag-fraction columns, not merely at the per-event median)."""
        summary = estimate_realization_summary(
            _moderate_params(), n_realizations=5, first_random_seed=0
        )
        hist = summary.median_flag_percentages[:, 2]
        well = summary.median_flag_percentages[:, 0]
        assert np.all(hist < 5.0), f"history-dependent phase should stay near zero; got {hist}"
        assert np.all(hist <= well), (
            f"history-dependent flags should not exceed the baseline; got hist={hist}, well={well}"
        )

    def test_remap_is_strongly_flagged_by_all_three(self) -> None:
        """Magnitude guard (replaces the removed single-realization
        ``test_remap_phase_flags_all_three``): the incoherent random-remap is
        the headline positive control, so every metric must flag it well
        above both the well-specified baseline and the drift misfit.

        A near-noop remap, or a spatially *coherent* remap (e.g. a pure
        reflection) — which is self-consistent under the decoder's own
        likelihood and correctly undetectable — would fail here. The
        percentages are smaller than the full-length figure (~37--43%)
        because ``_moderate_params`` uses short windows the trajectory only
        partly explores; bounds are set against the observed deterministic
        values (remap ~[15, 20, 14]% vs well ~[4, 5, 3]%, drift ~[3, 5, 1]%).
        """
        summary = estimate_realization_summary(
            _moderate_params(), n_realizations=5, first_random_seed=0
        )
        well = summary.median_flag_percentages[:, 0]
        remap = summary.median_flag_percentages[:, 1]
        drift = summary.median_flag_percentages[:, 4]
        assert np.all(remap > 10.0), f"remap should flag >10% for every metric; got {remap}"
        assert np.all(remap > 2.0 * well), (
            f"remap should flag >2x the well-specified baseline; got remap={remap}, well={well}"
        )
        assert np.all(remap > drift), (
            f"remap should flag more than drift for every metric; got remap={remap}, drift={drift}"
        )

    def test_rejects_nonpositive_realizations(self) -> None:
        with pytest.raises(ValueError, match="n_realizations"):
            estimate_realization_summary(_moderate_params(), n_realizations=0)


class TestFigure3RealizationSummaryInvariants:
    @staticmethod
    def _thresholds() -> DiagnosticThresholds:
        return DiagnosticThresholds(hpd_overlap=0.5, kl_divergence=1.0, predictive_pvalue=0.05)

    def test_non_2d_median_raises(self) -> None:
        with pytest.raises(ValueError, match="median_flag_percentages"):
            Figure3RealizationSummary(
                diagnostic_thresholds=self._thresholds(),
                median_flag_percentages=np.zeros(5),
                n_realizations=2,
            )

    def test_nonpositive_realizations_raises(self) -> None:
        with pytest.raises(ValueError, match="n_realizations"):
            Figure3RealizationSummary(
                diagnostic_thresholds=self._thresholds(),
                median_flag_percentages=np.zeros((3, 5)),
                n_realizations=0,
            )
