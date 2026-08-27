"""Tests for the general Bayesian decoder (decoding module)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.decoding import (
    DecoderOverrideSchedule,
    DecoderOverrideWindow,
    _condition_on,
    _resolve_baseline_firing_rates,
    decode_with_diagnostics,
)
from statespacecheck_paper.diagnostics import (
    compute_spike_event_diagnostics_from_rates,
)
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    normalize,
    place_field_rates,
)

from ._decoder_inputs import DecoderInputs, _diag_dominant_transition


class TestDecodeWithDiagnostics:
    def test_full_output_contract(self, decoder_inputs: DecoderInputs) -> None:
        """Lock in the full set of fields + their shapes. Downstream
        plotting and cache code consumes every field, so silently
        dropping one (e.g. ``predictive``, ``per_spike_likelihood``,
        ``event_*``) is a regression even if the four headline metrics
        are intact."""
        result = decoder_inputs.call()
        n_time, n_cells = decoder_inputs.spike_counts.shape
        n_bins = decoder_inputs.position_bins.size
        # Events come from spike_counts[1:] (t=0 has no prior); count > 1 expands
        # to that many events (src/.../analysis.py:583).
        n_events = int(decoder_inputs.spike_counts[1:].sum())

        expected_shapes = {
            # Distributions over position (time × bins).
            "posterior": (n_time, n_bins),
            "predictive": (n_time, n_bins),
            "likelihood": (n_time, n_bins),
            "spike_likelihood": (n_time, n_bins),
            # Per-cell metric matrices.
            "hpd_overlap": (n_time, n_cells),
            "kl_divergence": (n_time, n_cells),
            "predictive_pvalue": (n_time, n_cells),
            # Per-spike-event arrays (count expansion in src/.../analysis.py:584).
            "per_spike_likelihood": (n_events, n_bins),
            "event_time_ind": (n_events,),
            "event_cell_ind": (n_events,),
            "event_hpd_overlap": (n_events,),
            "event_kl_divergence": (n_events,),
            "event_predictive_pvalue": (n_events,),
        }
        for name, shape in expected_shapes.items():
            arr = getattr(result, name)
            assert arr.shape == shape, f"{name} shape mismatch: got {arr.shape}, want {shape}"

    def test_t0_diagnostics_are_nan(self, decoder_inputs: DecoderInputs) -> None:
        """No prior exists at t=0, so all diagnostics are NaN."""
        result = decoder_inputs.call()
        for key in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            assert np.all(np.isnan(getattr(result, key)[0]))

    def test_nan_pattern_matches_spike_pattern(self) -> None:
        """DecodingDiagnostics are NaN exactly where a cell has no spike at that time."""
        n_bins = 11
        spike_counts = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [2, 2]])
        position_bins = np.linspace(0, 100, n_bins)
        result = decode_with_diagnostics(
            spike_counts=spike_counts,
            position_bins=position_bins,
            transition_matrix=_diag_dominant_transition(n_bins),
            place_field_centers=np.array([25.0, 75.0]),
            place_field_std=5.0,
            place_field_rate_scale=0.1,
        )

        # DecodingDiagnostics NaN at (t, cell) iff cell has no spike at that t,
        # plus all of t=0 (no prior available).
        no_spike = spike_counts == 0
        no_spike[0] = True
        for key in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            np.testing.assert_array_equal(np.isnan(getattr(result, key)), no_spike)

    def test_count_greater_than_one_expands_to_multiple_events(self) -> None:
        n_bins = 11
        spike_counts = np.zeros((4, 2), dtype=int)
        spike_counts[1, 0] = 2
        spike_counts[2, 1] = 1

        result = decode_with_diagnostics(
            spike_counts=spike_counts,
            position_bins=np.linspace(0, 100, n_bins),
            transition_matrix=_diag_dominant_transition(n_bins),
            place_field_centers=np.array([25.0, 75.0]),
            place_field_std=5.0,
            place_field_rate_scale=0.1,
        )

        np.testing.assert_array_equal(result.event_time_ind, [1, 1, 2])
        np.testing.assert_array_equal(result.event_cell_ind, [0, 0, 1])
        assert result.event_kl_divergence.shape == (3,)
        # The two count=2 spike_counts share a (time, cell), so their per-event
        # diagnostics must equal the matrix value at that bin.
        np.testing.assert_allclose(
            result.event_kl_divergence[:2],
            np.repeat(result.kl_divergence[1, 0], 2),
        )

    def test_no_spikes_produces_all_nan_diagnostics(self) -> None:
        """Edge case: zero spike_counts => all-NaN matrices and empty event arrays."""
        n_time, n_cells, n_bins = 5, 2, 11
        result = decode_with_diagnostics(
            spike_counts=np.zeros((n_time, n_cells), dtype=int),
            position_bins=np.linspace(0, 100, n_bins),
            transition_matrix=_diag_dominant_transition(n_bins),
            place_field_centers=np.array([25.0, 75.0]),
            place_field_std=5.0,
            place_field_rate_scale=0.1,
        )
        assert np.all(np.isnan(result.hpd_overlap))
        assert np.all(np.isnan(result.kl_divergence))
        assert np.all(np.isnan(result.predictive_pvalue))
        assert result.event_time_ind.shape == (0,)
        assert result.event_cell_ind.shape == (0,)
        assert result.event_kl_divergence.shape == (0,)

    def test_alternative_transition_matrix_used_only_inside_window(
        self,
        decoder_inputs: DecoderInputs,
    ) -> None:
        """A :class:`DecoderOverrideWindow` with a ``transition_matrix`` must
        (a) leave the predictive untouched before the window and
        (b) actually change it inside. A regression that ignored the
        schedule would still produce well-shaped output, so we compare
        against a baseline run instead.
        """
        n_bins = decoder_inputs.position_bins.size
        # Choose an alternative matrix that is *very* different from the
        # baseline (peak=0.9 diag-dominant). A near-uniform matrix forces
        # the predictive to spread out dramatically — easy to detect.
        alt_matrix = _diag_dominant_transition(n_bins, peak=0.05)
        window = (3, 6)
        schedule = DecoderOverrideSchedule(
            (DecoderOverrideWindow(window[0], window[1], transition_matrix=alt_matrix),)
        )

        baseline = decoder_inputs.call()
        with_alt = decoder_inputs.call(override_schedule=schedule)

        # Before the window, the two runs must be bit-identical: nothing
        # in the algorithm has diverged yet.
        np.testing.assert_array_equal(
            baseline.predictive[: window[0]], with_alt.predictive[: window[0]]
        )
        np.testing.assert_array_equal(
            baseline.posterior[: window[0]], with_alt.posterior[: window[0]]
        )

        # Inside the window, at least one timestep's predictive must
        # measurably differ (this is what the window actually controls).
        # Use a generous tolerance so we don't depend on the exact
        # magnitude — we only assert "different".
        inside = slice(*window)
        assert not np.allclose(
            baseline.predictive[inside],
            with_alt.predictive[inside],
            atol=1e-6,
        ), f"transition_matrix in {window} did not change predictive — schedule ignored?"

    def test_alt_rates_used_only_inside_window(self) -> None:
        """A :class:`DecoderOverrideWindow` with ``firing_rate_table`` must change
        every per-spike diagnostic inside the window and leave it
        untouched outside. The in-window values are checked directly
        against the shared diagnostic routine evaluated with the alternate
        table, preventing a regression to oracle baseline rates.
        """
        position_bins = np.arange(5, dtype=float)
        place_field_centers = np.array([0.0, 2.0, 4.0])
        place_field_std = 0.45
        place_field_rate_scale = 2.0
        spike_counts = np.zeros((6, 3), dtype=int)
        for time_ind, cell_ind in ((1, 0), (2, 0), (3, 1), (4, 2), (5, 0)):
            spike_counts[time_ind, cell_ind] = 1

        baseline_rates = place_field_rates(
            position_bins, place_field_centers, place_field_std, place_field_rate_scale
        )
        alt_rates = place_field_rates(
            position_bins,
            np.array([4.0, 0.0, 2.0]),
            place_field_std,
            place_field_rate_scale,
        )
        window = (2, 4)
        schedule = DecoderOverrideSchedule(
            (DecoderOverrideWindow(window[0], window[1], firing_rate_table=alt_rates),)
        )

        with_alt = decode_with_diagnostics(
            spike_counts=spike_counts,
            position_bins=position_bins,
            transition_matrix=np.eye(position_bins.size),
            place_field_centers=place_field_centers,
            place_field_std=place_field_std,
            place_field_rate_scale=place_field_rate_scale,
            override_schedule=schedule,
        )

        evt_t = with_alt.event_time_ind
        inside = (evt_t >= window[0]) & (evt_t < window[1])
        outside = ~inside

        # The in-window metrics and displayed likelihood must all be the
        # values obtained from the decoder's active rate table.
        assert inside.any(), "test fixture produced no in-window spike events"
        expected = compute_spike_event_diagnostics_from_rates(
            with_alt.predictive,
            alt_rates,
            with_alt.event_time_ind[inside],
            with_alt.event_cell_ind[inside],
        )
        assert expected.per_spike_likelihood is not None
        np.testing.assert_allclose(
            with_alt.per_spike_likelihood[inside],
            expected.per_spike_likelihood,
        )
        for name in (
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_predictive_pvalue",
        ):
            np.testing.assert_allclose(
                getattr(with_alt, name)[inside],
                getattr(expected, name),
                err_msg=f"in-window {name} was not computed from firing_rate_table",
            )

        expected_outside = compute_spike_event_diagnostics_from_rates(
            with_alt.predictive,
            baseline_rates,
            with_alt.event_time_ind[outside],
            with_alt.event_cell_ind[outside],
        )
        assert expected_outside.per_spike_likelihood is not None
        np.testing.assert_allclose(
            with_alt.per_spike_likelihood[outside],
            expected_outside.per_spike_likelihood,
        )
        for name in (
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_predictive_pvalue",
        ):
            np.testing.assert_allclose(
                getattr(with_alt, name)[outside],
                getattr(expected_outside, name),
                err_msg=f"out-of-window {name} did not use baseline rates",
            )

        # The fixture independently distinguishes decoder and oracle rates
        # for every output, rather than relying on one metric to change.
        oracle = compute_spike_event_diagnostics_from_rates(
            with_alt.predictive,
            baseline_rates,
            with_alt.event_time_ind[inside],
            with_alt.event_cell_ind[inside],
        )
        for name in (
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_predictive_pvalue",
            "per_spike_likelihood",
        ):
            assert not np.allclose(getattr(expected, name), getattr(oracle, name)), (
                f"test fixture does not distinguish decoder and oracle values for {name}"
            )

        # Dense per-cell matrices mirror the event arrays after the window
        # overwrite. The one-event-per-coordinate fixture makes this exact.
        for dense_name, event_name in (
            ("hpd_overlap", "event_hpd_overlap"),
            ("kl_divergence", "event_kl_divergence"),
            ("predictive_pvalue", "event_predictive_pvalue"),
        ):
            np.testing.assert_allclose(
                getattr(with_alt, dense_name)[
                    with_alt.event_time_ind,
                    with_alt.event_cell_ind,
                ],
                getattr(with_alt, event_name),
            )

    def test_predictive_uses_column_stochastic_orientation(self) -> None:
        """The one-step predictive must marginalize as ``T @ post`` for the
        column-stochastic transition built by ``gaussian_transition_matrix``
        (column j is the distribution over next states given current state j).

        With a uniform prior at t=0 and an *asymmetric* transition, the
        correct orientation yields a predictive proportional to the row sums
        of ``T``; the transposed orientation (``post @ T``) would instead
        leave the uniform prior unchanged because the columns sum to 1. The
        symmetric fixtures used elsewhere cannot tell these apart, so this
        pins the orientation directly.
        """
        # Asymmetric but column-stochastic transition on a 3-bin grid.
        transition = np.array(
            [
                [0.8, 0.1, 0.0],
                [0.2, 0.6, 0.3],
                [0.0, 0.3, 0.7],
            ]
        )
        np.testing.assert_allclose(transition.sum(axis=0), 1.0)  # column-stochastic
        assert not np.allclose(transition, transition.T)  # genuinely asymmetric

        n_time = 3
        result = decode_with_diagnostics(
            spike_counts=np.zeros((n_time, 1), dtype=int),
            position_bins=np.array([0.0, 1.0, 2.0]),
            transition_matrix=transition,
            place_field_centers=np.array([1.0]),
            place_field_std=1.0,
            place_field_rate_scale=0.1,
        )

        # post[0] is the flat prior; predictive[1] = normalize(T @ post[0]).
        expected = normalize(transition @ result.posterior[0])
        np.testing.assert_allclose(result.predictive[1], expected, atol=1e-12)

        # The transposed orientation would return the uniform prior unchanged;
        # confirm the predictive actually moved so the check above is
        # discriminating rather than vacuous.
        assert not np.allclose(result.predictive[1], result.posterior[0])


class TestDecoderOverrideWindow:
    def test_start_not_before_end_raises(self) -> None:
        """``start >= end`` is rejected at construction."""
        with pytest.raises(ValueError, match="start < end"):
            DecoderOverrideWindow(10, 10)
        with pytest.raises(ValueError, match="start < end"):
            DecoderOverrideWindow(10, 5)

    def test_negative_rate_table_raises(self) -> None:
        """A negative rate table is rejected before it can become NaN
        likelihoods downstream."""
        bad = np.full((21, 3), 0.05)
        bad[0, 0] = -1.0
        with pytest.raises(ValueError, match="finite and non-negative"):
            DecoderOverrideWindow(3, 7, firing_rate_table=bad)

    def test_nonfinite_rate_table_raises(self) -> None:
        """A non-finite rate table is rejected at construction."""
        bad = np.full((21, 3), 0.05)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite and non-negative"):
            DecoderOverrideWindow(3, 7, firing_rate_table=bad)


class TestDecoderOverrideSchedule:
    def test_empty_schedule_has_no_window_anywhere(self) -> None:
        assert DecoderOverrideSchedule().window_at(0) is None
        assert DecoderOverrideSchedule().window_at(10_000) is None

    def test_window_at_returns_containing_window(self) -> None:
        sched = DecoderOverrideSchedule(
            (DecoderOverrideWindow(10, 20), DecoderOverrideWindow(30, 40))
        )
        assert sched.window_at(15) is sched.windows[0]
        assert sched.window_at(35) is sched.windows[1]
        # Half-open: end is exclusive, start inclusive.
        assert sched.window_at(10) is sched.windows[0]
        assert sched.window_at(20) is None
        assert sched.window_at(25) is None

    def test_overlapping_windows_raise(self) -> None:
        """Overlapping windows are rejected — diagnostics could not pick a
        single rate table for the overlap."""
        with pytest.raises(ValueError, match="must not overlap"):
            DecoderOverrideSchedule((DecoderOverrideWindow(10, 25), DecoderOverrideWindow(20, 30)))


class TestConditionOn:
    """The dynamax/non_local_detector-style Bayesian update helper."""

    def test_uniform_prior_yields_softmax_of_loglik(self) -> None:
        """With a uniform prior, the posterior must equal softmax(ll)."""
        n_bins = 8
        prior = np.full(n_bins, 1.0 / n_bins)
        ll = np.linspace(-5.0, 0.0, n_bins)
        new_probs, log_norm = _condition_on(prior, ll)

        # Posterior matches softmax(ll) — the log-sum-exp shift in
        # _condition_on is mathematically equivalent.
        expected = np.exp(ll - ll.max())
        expected = expected / expected.sum()
        np.testing.assert_allclose(new_probs, expected, rtol=1e-12, atol=1e-15)

        # Sums to 1.
        np.testing.assert_allclose(new_probs.sum(), 1.0, rtol=1e-12)

        # Log marginal: ll_max + log(sum exp(ll - ll_max)). For uniform
        # prior this is just log_sumexp(ll) - log(n_bins).
        from scipy.special import logsumexp

        np.testing.assert_allclose(log_norm, logsumexp(ll) - np.log(n_bins), rtol=1e-12)

    def test_handles_all_neg_inf_loglik(self) -> None:
        """All ``-inf`` likelihoods are a degenerate observation: the
        model assigns zero probability to the spike everywhere on the
        grid. The helper falls back to a uniform posterior and signals
        the situation via ``log_norm = -inf``; the caller in
        ``decode_with_diagnostics`` then surfaces the count via a
        ``RuntimeWarning``.
        """
        n_bins = 5
        prior = np.full(n_bins, 1.0 / n_bins)
        ll = np.full(n_bins, -np.inf)
        new_probs, log_norm = _condition_on(prior, ll)
        # Uniform fallback, properly normalized.
        np.testing.assert_allclose(new_probs, 1.0 / n_bins, rtol=1e-12)
        np.testing.assert_allclose(new_probs.sum(), 1.0, rtol=1e-12)
        # Zero-probability signal is exactly -inf so callers can compare directly.
        assert log_norm == -np.inf

    def test_retains_tiny_finite_prior_likelihood_overlap(self) -> None:
        """A tiny but finite overlap is a valid Bayesian update, not a
        numerical-underflow condition that permits a uniform reset."""
        n_bins = 8
        prior = np.zeros(n_bins)
        prior[0] = 1.0
        ll = np.full(n_bins, -1000.0)
        ll[-1] = 0.0  # likelihood mass at the opposite end of the grid
        new_probs, log_norm = _condition_on(prior, ll)
        np.testing.assert_array_equal(new_probs, prior)
        assert log_norm == pytest.approx(-1000.0)

    def test_exactly_disjoint_support_uses_zero_probability_fallback(self) -> None:
        """Fallback is reserved for exact zero joint support."""
        n_bins = 4
        prior = np.array([1.0, 0.0, 0.0, 0.0])
        ll = np.array([-np.inf, 0.0, -1.0, -2.0])
        new_probs, log_norm = _condition_on(prior, ll)
        np.testing.assert_allclose(new_probs, 1.0 / n_bins)
        assert log_norm == -np.inf

    @pytest.mark.parametrize(
        "prior,ll",
        [
            (np.array([np.nan, 1.0]), np.zeros(2)),
            (np.array([-1.0, 2.0]), np.zeros(2)),
            (np.array([0.0, 0.0]), np.zeros(2)),
            (np.array([0.5, 0.5]), np.array([0.0, np.nan])),
        ],
    )
    def test_invalid_inputs_do_not_silently_become_uniform(
        self,
        prior: np.ndarray,
        ll: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="finite nonnegative 1D distribution"):
            _condition_on(prior, ll)

    def test_extreme_loglik_does_not_underflow(self) -> None:
        """Likelihoods with -800 magnitude (would underflow exp(-800)
        to zero in float64) must still produce a normalized posterior.
        This is the regime the linear-space code's reset-to-uniform
        branch was guarding against.
        """
        n_bins = 10
        prior = np.full(n_bins, 1.0 / n_bins)
        # Likelihoods centered around -800 — exp() would underflow but
        # the log-sum-exp shift handles it.
        ll = np.array(
            [-800.0, -795.0, -790.0, -785.0, -780.0, -782.0, -787.0, -792.0, -797.0, -802.0]
        )
        new_probs, log_norm = _condition_on(prior, ll)
        assert np.all(np.isfinite(new_probs))
        np.testing.assert_allclose(new_probs.sum(), 1.0, rtol=1e-12)
        # log_norm carries the magnitude through, finite.
        assert np.isfinite(log_norm)
        assert log_norm < -700  # Same order of magnitude as the input.


class TestDecodeWithDiagnosticsLogSpace:
    """Stress tests for the log-space rewrite of decode_with_diagnostics.

    The previous implementation reset the posterior to uniform when
    ``prior * combined_likelihood`` underflowed to zero. The log-space
    rewrite removes that branch; the finite-overlap numerical-underflow
    failure mode cannot occur. An exactly impossible observation retains
    a separately tested fallback. These tests pin that distinction so a
    future refactor cannot silently reintroduce the arbitrary reset.
    """

    def test_posterior_sums_to_one_at_every_step(self) -> None:
        """Algorithmic correctness: at every time step the posterior
        must be a proper probability distribution.
        """
        rng = np.random.default_rng(0)
        n_time, n_cells, n_bins = 50, 3, 21
        spike_counts = rng.poisson(1.0, size=(n_time, n_cells))
        position_bins = np.linspace(0.0, 100.0, n_bins)
        transition_matrix = gaussian_transition_matrix(position_bins, step_std=2.0)
        place_field_centers = np.array([25.0, 50.0, 75.0])
        results = decode_with_diagnostics(
            spike_counts,
            position_bins,
            transition_matrix,
            place_field_centers,
            place_field_std=10.0,
            place_field_rate_scale=0.1,
        )
        posterior = results.posterior
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12)

    def test_extreme_prior_likelihood_mismatch_yields_meaningful_posterior(self) -> None:
        """Stress test for an extreme but finite overlap: place a narrow prior at
        one end of the grid then drive the decoder with spike_counts whose
        place-field rate is concentrated at the *other* end.

        On the pre-refactor code this configuration triggered the
        ``posterior_sum < 1e-300`` reset-to-uniform branch. The
        log-space implementation must instead produce a normalized,
        non-uniform posterior at every step.
        """
        n_time, n_cells, n_bins = 30, 2, 51
        position_bins = np.linspace(0.0, 100.0, n_bins)
        # Narrow transition kernel so the prior stays concentrated.
        transition_matrix = gaussian_transition_matrix(position_bins, step_std=0.5)
        # Two cells with place fields at x≈90 — far from the
        # bias-initialized posterior which mostly accumulates near 0.
        place_field_centers = np.array([88.0, 92.0])
        # Both cells fire every time step → strong likelihood signal at
        # x≈90.
        spike_counts = np.ones((n_time, n_cells), dtype=np.int_)

        results = decode_with_diagnostics(
            spike_counts,
            position_bins,
            transition_matrix,
            place_field_centers,
            place_field_std=2.0,
            place_field_rate_scale=5.0,
        )
        posterior = results.posterior

        # Every step's posterior sums to 1 (no underflow, no reset).
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10)

        # Posterior is not uniform — at least one step's distribution
        # is meaningfully concentrated. The old reset-to-uniform branch
        # would have made every transitioning step uniform, so non-
        # uniformity at any step rules out the silent fallback.
        uniform = 1.0 / n_bins
        max_dev = float(np.max(np.abs(posterior - uniform)))
        assert max_dev > 0.05, (
            f"posterior is uniformly flat (max |Δ uniform| = {max_dev:.4f}); "
            f"the log-space rewrite may have collapsed to the old reset-to-"
            f"uniform behaviour."
        )

        # The mass should ultimately concentrate near the place-field
        # centers under sustained firing — sanity check the filter is
        # working at all.
        final_posterior = posterior[-1]
        peak_bin = int(np.argmax(final_posterior))
        peak_pos = position_bins[peak_bin]
        assert peak_pos > 80.0, (
            f"final posterior peak at x={peak_pos:.1f}, expected near 90 "
            f"under sustained firing at PF centers (88, 92)."
        )

    def test_impossible_observation_emits_summary_warning(self) -> None:
        """The per-step ``_condition_on`` ``-inf`` path is covered by
        ``TestConditionOn``; the *summary* warning emitted post-loop
        is not asserted anywhere. Force an impossible observation via a
        place-field center so far from the decoder grid that every
        bin's rate underflows to 0.0."""
        n_time, n_cells, n_bins = 8, 1, 21
        position_bins = np.linspace(0.0, 100.0, n_bins)
        transition_matrix = gaussian_transition_matrix(position_bins, step_std=2.0)
        # PF center so far from position_bins that exp(-d^2 / 2*place_field_std^2)
        # underflows to exactly 0.0 — every bin's rate is 0.0.
        place_field_centers = np.array([1e6])
        spike_counts = np.ones((n_time, n_cells), dtype=np.int_)

        with pytest.warns(
            RuntimeWarning,
            match=r"observation had zero probability on the prior support at \d+ timestep",
        ):
            decode_with_diagnostics(
                spike_counts,
                position_bins,
                transition_matrix,
                place_field_centers,
                place_field_std=5.0,
                place_field_rate_scale=1.0,
            )


class TestDecoderOverrideWindowTightening:
    """Write-protect, shape validation invariants."""

    @pytest.mark.parametrize(
        ("field", "kwargs_factory"),
        [
            ("firing_rate_table", lambda arr: {"firing_rate_table": arr}),
            ("transition_matrix", lambda arr: {"transition_matrix": arr}),
        ],
    )
    def test_arrays_are_write_protected(self, field: str, kwargs_factory: Any) -> None:
        """``frozen=True`` only blocks rebinding the field; the array
        itself must also be read-only so callers can't bypass
        validation by mutating in place."""
        arr = np.eye(5) * 0.5 + 0.1 if field == "transition_matrix" else np.full((5, 3), 0.1)
        w = DecoderOverrideWindow(10, 20, **kwargs_factory(arr))
        stored = getattr(w, field)
        assert stored is not None
        assert stored.flags.writeable is False
        with pytest.raises(ValueError, match="read-only|assignment destination"):
            stored[0, 0] = 999.0

    def test_caller_array_not_mutated_by_construction(self) -> None:
        """Defensive copy: caller's original array stays writable."""
        rates = np.full((5, 3), 0.1)
        original_id = id(rates)
        DecoderOverrideWindow(10, 20, firing_rate_table=rates)
        assert id(rates) == original_id
        assert rates.flags.writeable is True

    def test_validate_against_accepts_matching_shape(self) -> None:
        rates = np.full((5, 3), 0.1)
        w = DecoderOverrideWindow(10, 20, firing_rate_table=rates)
        w.validate_against(n_bins=5, n_cells=3)  # does not raise

    def test_validate_against_raises_on_mismatched_decoder_rates_shape(self) -> None:
        rates = np.full((5, 3), 0.1)
        w = DecoderOverrideWindow(10, 20, firing_rate_table=rates)
        with pytest.raises(ValueError, match=r"firing_rate_table shape"):
            w.validate_against(n_bins=7, n_cells=3)

    def test_validate_against_raises_on_mismatched_transition_shape(self) -> None:
        transition = np.eye(5)
        w = DecoderOverrideWindow(10, 20, transition_matrix=transition)
        with pytest.raises(ValueError, match=r"transition_matrix shape"):
            w.validate_against(n_bins=7, n_cells=3)


class TestStoredLikelihoodNormalization:
    """The log-space rewrite stores ``combined_likelihood_all`` and
    ``spike_likelihood_all`` via a bespoke shift-and-normalize. A
    regression that left these unnormalized would still pass shape
    contracts but distort every downstream HPD / KL on the displayed
    likelihood. Pin the normalization directly.
    """

    def test_combined_likelihood_sums_to_one_at_every_step(
        self, decoder_inputs: DecoderInputs
    ) -> None:
        result = decoder_inputs.call()
        likelihood = result.likelihood
        np.testing.assert_allclose(
            likelihood.sum(axis=1),
            1.0,
            rtol=1e-10,
            atol=1e-12,
            err_msg="combined_likelihood_all is not row-normalized",
        )

    def test_spike_likelihood_sums_to_one_at_spike_steps(
        self, decoder_inputs: DecoderInputs
    ) -> None:
        """Steps with at least one spike must produce a normalized
        spike-only likelihood. NaN at no-spike steps is acceptable (and
        documented)."""
        result = decoder_inputs.call()
        spike_lik = result.spike_likelihood
        spike_counts = decoder_inputs.spike_counts
        spike_steps = np.where(spike_counts.sum(axis=1) > 0)[0]
        # Skip t=0 which is a flat-initialized row, not a likelihood
        # over any observation.
        spike_steps = spike_steps[spike_steps > 0]
        assert spike_steps.size, "test fixture produced no spike_counts"
        sums = spike_lik[spike_steps].sum(axis=1)
        np.testing.assert_allclose(
            sums,
            1.0,
            rtol=1e-10,
            atol=1e-12,
            err_msg="spike_likelihood_all is not row-normalized at spike steps",
        )


class TestDecoderValidatesScheduleShapes:
    """``decode_with_diagnostics`` must invoke ``validate_against`` on
    every schedule entry before running the time loop. A regression
    that drops the call would only fail later with a cryptic
    broadcasting error inside ``poisson.logpmf``.
    """

    def test_mismatched_decoder_rates_raises_before_decode(
        self, decoder_inputs: DecoderInputs
    ) -> None:
        n_bins = decoder_inputs.position_bins.size
        wrong_shape = np.full((n_bins + 3, decoder_inputs.spike_counts.shape[1]), 0.1)
        schedule = DecoderOverrideSchedule(
            (DecoderOverrideWindow(2, 5, firing_rate_table=wrong_shape),)
        )
        with pytest.raises(ValueError, match=r"firing_rate_table shape"):
            decoder_inputs.call(override_schedule=schedule)

    def test_mismatched_transition_matrix_raises_before_decode(
        self, decoder_inputs: DecoderInputs
    ) -> None:
        n_bins = decoder_inputs.position_bins.size
        wrong_transition = np.eye(n_bins + 1)
        schedule = DecoderOverrideSchedule(
            (DecoderOverrideWindow(2, 5, transition_matrix=wrong_transition),)
        )
        with pytest.raises(ValueError, match=r"transition_matrix shape"):
            decoder_inputs.call(override_schedule=schedule)


class TestLogSpaceReferenceComparison:
    """Tighter regression guard for the log-space posterior update:
    compare against an independent log-space reference computed step by
    step. A regression that subtly resets to uniform on a subset of
    steps would pass the earlier "max_dev > 0.05" smoke check but fail
    here.
    """

    def test_posterior_matches_independent_log_space_reference(
        self, decoder_inputs: DecoderInputs
    ) -> None:
        from scipy.stats import poisson as _poisson

        from statespacecheck_paper.simulation import place_field_rates

        spike_counts = decoder_inputs.spike_counts
        position_bins = decoder_inputs.position_bins
        n_time, _ = spike_counts.shape
        n_bins = position_bins.size

        # Use an *asymmetric* column-stochastic transition (upward drift) so
        # the reference is a genuine orientation guard. The fixture's
        # ``_diag_dominant_transition`` is symmetric, so ``T @ post`` and
        # ``post @ T`` coincide and a production regression to the wrong
        # orientation would slip through. With this matrix the two differ.
        transition = np.eye(n_bins) * 0.7 + np.eye(n_bins, k=-1) * 0.3
        transition /= transition.sum(axis=0, keepdims=True)  # column-stochastic
        assert not np.allclose(transition, transition.T)  # genuinely asymmetric

        # Reference: linear-space prior, log-space combined likelihood,
        # softmax-shift normalization. No reset-to-uniform branch.
        rates = place_field_rates(
            position_bins,
            decoder_inputs.place_field_centers,
            decoder_inputs.place_field_std,
            decoder_inputs.place_field_rate_scale,
        )
        ref_post = np.zeros((n_time, n_bins))
        ref_post[0] = np.ones(n_bins) / n_bins
        for t in range(1, n_time):
            # ``transition`` is column-stochastic (column j = P(next | current
            # = j)), so the predictive marginal is ``T @ post``. This mirrors
            # the production convention and keeps the reference an independent
            # check of orientation, not just of the log-space arithmetic.
            prior = transition @ ref_post[t - 1]
            prior = prior / prior.sum()
            log_lik = _poisson.logpmf(spike_counts[t][None, :], rates).sum(axis=1)
            ll_max = float(np.max(log_lik))
            assert np.isfinite(ll_max)
            weighted = prior * np.exp(log_lik - ll_max)
            norm = weighted.sum()
            assert norm > 0
            ref_post[t] = weighted / norm

        result = decoder_inputs.call(transition_matrix=transition)
        np.testing.assert_allclose(result.posterior, ref_post, rtol=1e-10, atol=1e-12)


class TestResolveBaselineFiringRates:
    def test_wrong_shape_raises(self) -> None:
        position_bins = np.linspace(0, 100, 5)
        place_field_centers = np.array([25.0, 75.0])
        with pytest.raises(ValueError, match="does not match the decoder grid"):
            _resolve_baseline_firing_rates(
                np.ones((4, 2)), position_bins, place_field_centers, 5.0, 0.1, n_bins=5, n_cells=2
            )

    def test_negative_rate_raises(self) -> None:
        position_bins = np.linspace(0, 100, 5)
        place_field_centers = np.array([25.0, 75.0])
        bad = np.ones((5, 2))
        bad[0, 0] = -1.0
        with pytest.raises(ValueError, match="finite, non-negative"):
            _resolve_baseline_firing_rates(
                bad, position_bins, place_field_centers, 5.0, 0.1, n_bins=5, n_cells=2
            )

    def test_nonfinite_rate_raises(self) -> None:
        position_bins = np.linspace(0, 100, 5)
        place_field_centers = np.array([25.0, 75.0])
        bad = np.ones((5, 2))
        bad[1, 1] = np.inf
        with pytest.raises(ValueError, match="finite, non-negative"):
            _resolve_baseline_firing_rates(
                bad, position_bins, place_field_centers, 5.0, 0.1, n_bins=5, n_cells=2
            )

    def test_valid_table_returned_unchanged(self) -> None:
        position_bins = np.linspace(0, 100, 5)
        place_field_centers = np.array([25.0, 75.0])
        good = np.abs(np.random.default_rng(0).random((5, 2)))
        out = _resolve_baseline_firing_rates(
            good, position_bins, place_field_centers, 5.0, 0.1, n_bins=5, n_cells=2
        )
        assert np.array_equal(out, good)

    def test_none_builds_from_place_fields(self) -> None:
        position_bins = np.linspace(0, 100, 5)
        place_field_centers = np.array([25.0, 75.0])
        built = _resolve_baseline_firing_rates(
            None, position_bins, place_field_centers, 5.0, 0.1, n_bins=5, n_cells=2
        )
        assert built.shape == (5, 2)
        assert np.array_equal(
            built, place_field_rates(position_bins, place_field_centers, 5.0, 0.1)
        )


class TestDecoderApiContract:
    """Durable names/order contract for the decoder API, so a future rename
    that silently changes the public surface fails here."""

    def test_decode_with_diagnostics_parameter_names_and_order(self) -> None:
        import inspect

        params = list(inspect.signature(decode_with_diagnostics).parameters)
        assert params == [
            "spike_counts",
            "position_bins",
            "transition_matrix",
            "place_field_centers",
            "place_field_std",
            "place_field_rate_scale",
            "override_schedule",
            "baseline_firing_rates",
        ]

    def test_decoder_override_window_fields(self) -> None:
        import dataclasses

        names = {f.name for f in dataclasses.fields(DecoderOverrideWindow)}
        assert {"start", "end", "transition_matrix", "firing_rate_table"} <= names
        # Old field name must be gone.
        assert "decoder_rates" not in names
