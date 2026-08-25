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

from statespacecheck_paper.analysis import (
    DecodeParams,
    PhaseBoundary,
    Thresholds,
    compute_per_cell_diagnostics_from_rates,
    get_remapped_pf_centers,
)
from statespacecheck_paper.figure03_demo import (
    PHASE_LABELS,
    SimulationResult,
    StableSummary,
    estimate_stable_summary,
    run_figure03_simulation,
)
from statespacecheck_paper.simulation import gaussian_transition_matrix, placefield_rates


def _moderate_params() -> DecodeParams:
    """Phase sizes large enough for per-phase medians to be stable but
    small enough to keep the test fast (~3 s on a laptop).
    """
    return DecodeParams(
        phase_boundaries=(600, 900, 1100, 1400, 1600, 1900, 2100, 3100),
    )


def _per_phase_medians(
    sim: SimulationResult,
) -> dict[str, tuple[float, float, float]]:
    """Return (kl_med, hpd_med, sp_med) per phase label."""
    metrics = sim.metrics
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
        sp = float(np.nanmedian(metrics.event_spike_prob[mask]))
        out[label] = (kl, hpd, sp)
    return out


@pytest.fixture(scope="module")
def sim() -> SimulationResult:
    return run_figure03_simulation(_moderate_params(), seed=0)


def test_phase_labels_and_boundaries(sim: SimulationResult) -> None:
    """``run_figure03_simulation`` emits every canonical phase in order
    and a timeline that ends at the SPARSE_POP_END boundary.
    """
    params = sim.params
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
    x_true = np.asarray(sim.x_true)
    assert x_true.shape[0] == end


def test_remap_phase_uses_decoder_likelihood(sim: SimulationResult) -> None:
    """Remap diagnostics must use the same remapped rates as the decoder.

    This guards against combining the decoder's predictive distribution
    with unavailable baseline/oracle place fields during the remap window.
    """
    params = sim.params
    assert params.pf_centers is not None
    start = params.phase_boundaries[PhaseBoundary.REMAP_START]
    end = params.phase_boundaries[PhaseBoundary.REMAP_END]
    in_window = (sim.metrics.event_time_ind >= start) & (sim.metrics.event_time_ind < end)
    assert in_window.any(), "test simulation produced no remap-window spike events"

    remapped_normal_rates = placefield_rates(
        sim.xs,
        get_remapped_pf_centers(params.pf_centers, params.remap_from_to, active=True),
        params.pf_width,
        params.rate_scale,
    )
    sparse_scale = params.sparse_cell_peak_rate * np.sqrt(2.0 * np.pi) * params.sparse_cell_width
    baseline_sparse_rates = params.sparse_cell_baseline_gain * placefield_rates(
        sim.xs,
        np.asarray(sim.sparse_cell_centers),
        params.sparse_cell_width,
        sparse_scale,
    )
    remapped_rates = np.hstack([remapped_normal_rates, baseline_sparse_rates])
    expected = compute_per_cell_diagnostics_from_rates(
        sim.metrics.predictive,
        remapped_rates,
        sim.metrics.event_time_ind[in_window],
        sim.metrics.event_cell_ind[in_window],
    )
    assert expected.per_spike_likelihood is not None

    np.testing.assert_allclose(
        sim.metrics.per_spike_likelihood[in_window],
        expected.per_spike_likelihood,
    )
    predictive = sim.metrics.predictive[sim.metrics.event_time_ind[in_window]]
    np.testing.assert_allclose(
        sim.metrics.event_hpd_overlap[in_window],
        ssc.hpd_overlap(
            predictive,
            sim.metrics.per_spike_likelihood[in_window],
            coverage=0.95,
        ),
        err_msg="remap HPD was not computed from the displayed event likelihood",
    )
    np.testing.assert_allclose(
        sim.metrics.event_kl_divergence[in_window],
        ssc.kl_divergence(
            predictive,
            sim.metrics.per_spike_likelihood[in_window],
        ),
        err_msg="remap KL was not computed from the displayed event likelihood",
    )
    for name in (
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_spike_prob",
    ):
        np.testing.assert_allclose(
            getattr(sim.metrics, name)[in_window],
            getattr(expected, name),
            err_msg=f"remap-window {name} did not use decoder rates",
        )

    # Confirm that this fixture distinguishes decoder-rate diagnostics from
    # the old oracle computation based on the unperturbed place fields.
    baseline_rates = np.hstack(
        [
            placefield_rates(
                sim.xs,
                params.pf_centers,
                params.pf_width,
                params.rate_scale,
            ),
            baseline_sparse_rates,
        ]
    )
    oracle = compute_per_cell_diagnostics_from_rates(
        sim.metrics.predictive,
        baseline_rates,
        sim.metrics.event_time_ind[in_window],
        sim.metrics.event_cell_ind[in_window],
    )
    for name in (
        "event_hpd_overlap",
        "event_kl_divergence",
        "event_spike_prob",
    ):
        assert not np.allclose(getattr(expected, name), getattr(oracle, name)), (
            f"test fixture does not distinguish decoder and oracle values for {name}"
        )


def test_sparse_population_dissociates_kl_from_other_metrics(
    sim: SimulationResult,
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
    sim: SimulationResult,
) -> None:
    """The last phase is a fixed immobile stop, not a transition perturbation.

    The ordinary ensemble is quiet, the sparse population fires, and both its
    likelihood and the decoder prediction use the declared model. This pins
    the KL dissociation to sparse information rather than hidden mismatch.
    """
    params = sim.params
    w0 = params.phase_boundaries[PhaseBoundary.RECOVERY3_END]
    w1 = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    n_sparse = len(sim.sparse_cell_centers)
    n_normal = sim.spikes.shape[1] - n_sparse
    np.testing.assert_allclose(sim.x_true[w0:w1], params.sparse_position)
    # The ordinary ensemble is silent; only the sparse-population cells fire.
    assert sim.spikes[w0:w1, :n_normal].sum() == 0
    assert sim.spikes[w0:w1, n_normal:].sum() > 0
    assert n_sparse == params.n_sparse_cells

    in_window = (sim.metrics.event_time_ind >= w0) & (sim.metrics.event_time_ind < w1)
    assert in_window.any(), "the sparse population produced no sparse-window diagnostic events"
    assert np.all(sim.metrics.event_cell_ind[in_window] >= n_normal)

    sparse_scale = params.sparse_cell_peak_rate * np.sqrt(2.0 * np.pi) * params.sparse_cell_width
    sparse_rates = placefield_rates(
        sim.xs,
        np.asarray(sim.sparse_cell_centers),
        params.sparse_cell_width,
        sparse_scale,
    )
    expected = compute_per_cell_diagnostics_from_rates(
        sim.metrics.predictive,
        sparse_rates,
        sim.metrics.event_time_ind[in_window],
        # Re-index the sparse-cell columns to 0..n_sparse-1 for the K-column
        # sparse rate table.
        (sim.metrics.event_cell_ind[in_window] - n_normal).astype(np.intp),
    )
    np.testing.assert_allclose(
        sim.metrics.per_spike_likelihood[in_window],
        expected.per_spike_likelihood,
    )

    # No phase-specific transition is introduced: the stored prediction is
    # exactly the standard transition applied to the preceding posterior.
    event_time = int(sim.metrics.event_time_ind[np.flatnonzero(in_window)[0]])
    transition = gaussian_transition_matrix(sim.xs, params.sigx_pred)
    expected_predictive = transition @ sim.metrics.posterior[event_time - 1]
    expected_predictive /= expected_predictive.sum()
    np.testing.assert_allclose(
        sim.metrics.predictive[event_time],
        expected_predictive,
    )


def test_history_dependent_firing_per_spike_metrics_near_baseline(
    sim: SimulationResult,
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
    # spike_prob stays in a band around baseline (neither collapsing like
    # remap nor spuriously inflating).
    assert 0.5 * base_sp < hd_sp < 1.5 * base_sp, (
        "per-spike spike_prob in the history-dependent phase should stay "
        f"within +/-50% of baseline; got baseline={base_sp:.3f}, "
        f"hist-dep={hd_sp:.3f}"
    )


def test_drift_phase_inflates_kl(sim: SimulationResult) -> None:
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


# ---------------------------------------------------------------------------
# estimate_stable_summary: pooled thresholds + median per-phase fractions
# ---------------------------------------------------------------------------


class TestEstimateStableSummary:
    def test_shapes_and_determinism(self) -> None:
        """The summary is (3 metrics x 6 columns: well-specified, replay,
        remap, history, drift, sparse population), fractions are percentages, and the same
        seeds reproduce the same result."""
        params = _moderate_params()
        summary = estimate_stable_summary(params, n_realizations=3, base_seed=0)

        assert isinstance(summary, StableSummary)
        assert summary.n_realizations == 3
        assert summary.frac_median.shape == (3, 6)
        # Percentages in [0, 100].
        assert np.all(summary.frac_median >= 0.0)
        assert np.all(summary.frac_median <= 100.0)
        # Threshold is a valid, finite Thresholds with the fixed p-value cutoff.
        assert summary.thresholds.spike_prob == 0.05
        assert np.isfinite(summary.thresholds.kl_divergence)

        repeat = estimate_stable_summary(params, n_realizations=3, base_seed=0)
        np.testing.assert_array_equal(summary.frac_median, repeat.frac_median)
        assert summary.thresholds.kl_divergence == repeat.thresholds.kl_divergence

    def test_remap_column_is_most_flagged(self) -> None:
        """Scientific regression guard: across realizations, the remap column
        (index 2) is flagged far more than the well-specified column (index 0)
        for every metric — the headline 'all three detect remap' result, now
        on a stabilized median."""
        summary = estimate_stable_summary(_moderate_params(), n_realizations=5, base_seed=0)
        for row in range(3):
            assert summary.frac_median[row, 2] > summary.frac_median[row, 0]

    def test_replay_is_not_flagged(self) -> None:
        """Scientific claim: the replay event (column 1) is *not* a
        misspecification. The decoder tracks the swept trajectory, so every
        metric stays low — far below the remap positive control — even though
        the decoded position departs from the (fixed) true position."""
        summary = estimate_stable_summary(_moderate_params(), n_realizations=5, base_seed=0)
        replay = summary.frac_median[:, 1]
        remap = summary.frac_median[:, 2]
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
        summary = estimate_stable_summary(_moderate_params(), n_realizations=5, base_seed=0)
        sparse = summary.frac_median[:, 5]  # [HPD, predictive-p, KL]
        well = summary.frac_median[:, 0]
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
        by all three per-spike spatial diagnostics (column 3 stays low on the
        flag-fraction columns, not merely at the per-event median)."""
        summary = estimate_stable_summary(_moderate_params(), n_realizations=5, base_seed=0)
        hist = summary.frac_median[:, 3]
        well = summary.frac_median[:, 0]
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
        summary = estimate_stable_summary(_moderate_params(), n_realizations=5, base_seed=0)
        well = summary.frac_median[:, 0]
        remap = summary.frac_median[:, 2]
        drift = summary.frac_median[:, 4]
        assert np.all(remap > 10.0), f"remap should flag >10% for every metric; got {remap}"
        assert np.all(remap > 2.0 * well), (
            f"remap should flag >2x the well-specified baseline; got remap={remap}, well={well}"
        )
        assert np.all(remap > drift), (
            f"remap should flag more than drift for every metric; got remap={remap}, drift={drift}"
        )

    def test_rejects_nonpositive_realizations(self) -> None:
        with pytest.raises(ValueError, match="n_realizations"):
            estimate_stable_summary(_moderate_params(), n_realizations=0)


class TestStableSummaryInvariants:
    @staticmethod
    def _thresholds() -> Thresholds:
        return Thresholds(hpd_overlap=0.5, kl_divergence=1.0, spike_prob=0.05)

    def test_non_2d_median_raises(self) -> None:
        with pytest.raises(ValueError, match="frac_median"):
            StableSummary(
                thresholds=self._thresholds(),
                frac_median=np.zeros(5),
                n_realizations=2,
            )

    def test_nonpositive_realizations_raises(self) -> None:
        with pytest.raises(ValueError, match="n_realizations"):
            StableSummary(
                thresholds=self._thresholds(),
                frac_median=np.zeros((3, 5)),
                n_realizations=0,
            )
