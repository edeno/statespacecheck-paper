"""Tests for analysis module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.analysis import (
    DecodeParams,
    MisfitSchedule,
    MisfitWindow,
    Thresholds,
    Transformed,
    _condition_on,
    compute_per_cell_diagnostics_from_rates,
    compute_thresholds,
    decode_and_diagnostics,
    get_remapped_pf_centers,
    likelihood_grid_for_counts,
    transform_metrics,
)
from statespacecheck_paper.figure03_demo import PHASE_LABELS, SimulationResult
from statespacecheck_paper.simulation import gaussian_transition_matrix

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderInputs:
    """Bundle of inputs for ``decode_and_diagnostics``."""

    spikes: np.ndarray
    xs: np.ndarray
    transition_matrix: np.ndarray
    pf_centers: np.ndarray
    pf_width: float
    rate_scale: float

    def call(self, **overrides: Any) -> dict:
        kwargs: dict[str, Any] = {
            "spikes": self.spikes,
            "xs": self.xs,
            "transition_matrix": self.transition_matrix,
            "pf_centers": self.pf_centers,
            "pf_width": self.pf_width,
            "rate_scale": self.rate_scale,
        }
        kwargs.update(overrides)
        return decode_and_diagnostics(**kwargs)


def _diag_dominant_transition(n_bins: int, peak: float = 0.9) -> np.ndarray:
    return np.eye(n_bins) * peak + (1.0 - peak) / n_bins


@pytest.fixture
def decoder_inputs() -> DecoderInputs:
    """Small reproducible decoder problem with no misfit schedule."""
    rng = np.random.default_rng(42)
    n_time, n_cells, n_bins = 10, 3, 21
    return DecoderInputs(
        spikes=rng.poisson(1.0, size=(n_time, n_cells)),
        xs=np.linspace(0, 100, n_bins),
        transition_matrix=_diag_dominant_transition(n_bins),
        pf_centers=np.array([25.0, 50.0, 75.0]),
        pf_width=5.0,
        rate_scale=0.1,
    )


@pytest.fixture
def metrics_2d() -> dict[str, np.ndarray]:
    """Standard (n_time, n_cells) metrics dict for transform/threshold tests."""
    rng = np.random.default_rng(42)
    return {
        "hpd_overlap": rng.uniform(0.5, 1.0, (100, 5)),
        "kl_divergence": rng.uniform(0.0, 2.0, (100, 5)),
        "spike_prob": rng.uniform(0.0, 1.0, (100, 5)),
    }


# ---------------------------------------------------------------------------
# DecodeParams
# ---------------------------------------------------------------------------


class TestDecodeParams:
    def test_post_init_initializes_pf_centers_to_grid(self) -> None:
        params = DecodeParams()
        np.testing.assert_array_equal(params.pf_centers, np.arange(0, 101, 10, dtype=float))

    def test_post_init_respects_provided_pf_centers(self) -> None:
        custom = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        params = DecodeParams(pf_centers=custom)
        np.testing.assert_array_equal(params.pf_centers, custom)

    def test_remap_window_returns_start_end_tuple(self) -> None:
        params = DecodeParams(
            phase_boundaries=(1000, 2000, 14_000, 18_000, 22_000, 26_000, 30_000, 32_000)
        )
        assert params.remap_window == (1000, 2000)


# ---------------------------------------------------------------------------
# likelihood_grid_for_counts
# ---------------------------------------------------------------------------


class TestLikelihoodGridForCounts:
    @pytest.fixture
    def grid_args(self) -> dict[str, Any]:
        return {
            "xs": np.linspace(0, 100, 21),
            "pf_centers": np.array([25.0, 50.0, 75.0]),
            "pf_width": 5.0,
            "rate_scale": 0.1,
        }

    def test_shape_matches_n_bins_and_n_cells(self, grid_args: dict) -> None:
        likelihood = likelihood_grid_for_counts(counts=np.array([2, 1, 3]), **grid_args)
        assert likelihood.shape == (21, 3)

    def test_normalized_per_cell(self, grid_args: dict) -> None:
        likelihood = likelihood_grid_for_counts(counts=np.array([2, 1, 3]), **grid_args)
        np.testing.assert_allclose(likelihood.sum(axis=0), 1.0, rtol=1e-5)

    def test_zero_counts_still_normalized(self, grid_args: dict) -> None:
        """Zero spikes shouldn't produce NaN/Inf or break normalization."""
        likelihood = likelihood_grid_for_counts(counts=np.zeros(3, dtype=int), **grid_args)
        assert np.isfinite(likelihood).all()
        np.testing.assert_allclose(likelihood.sum(axis=0), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# get_remapped_pf_centers
# ---------------------------------------------------------------------------


class TestGetRemappedPfCenters:
    def test_inactive_returns_input_unchanged_without_copy(self) -> None:
        """active=False is the hot path — must avoid the copy."""
        pf_centers = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        result = get_remapped_pf_centers(pf_centers, (0, 1), active=False)
        np.testing.assert_array_equal(result, pf_centers)
        assert result is pf_centers

    def test_active_returns_copy_and_does_not_mutate_input(self) -> None:
        pf_centers = np.array([0.0, 10.0, 20.0])
        result = get_remapped_pf_centers(pf_centers, (0, 1), active=True)
        assert result is not pf_centers
        np.testing.assert_array_equal(pf_centers, [0.0, 10.0, 20.0])

    def test_single_remapping_assigns_dst_center_to_src(self) -> None:
        pf_centers = np.array([0.0, 10.0, 20.0, 30.0])
        result = get_remapped_pf_centers(pf_centers, (2, 0), active=True)
        np.testing.assert_array_equal(result, [0.0, 10.0, 0.0, 30.0])

    def test_multiple_remappings_use_original_dst_values(self) -> None:
        """Bidirectional swap (0->1, 1->0) must use *original* values, not
        sequentially overwritten ones — otherwise both end up with the same
        center."""
        pf_centers = np.array([0.0, 10.0, 20.0, 30.0])
        result = get_remapped_pf_centers(pf_centers, ((0, 1), (1, 0)), active=True)
        np.testing.assert_array_equal(result, [10.0, 0.0, 20.0, 30.0])

    def test_default_bidirectional_swap_pattern(self) -> None:
        """The DecodeParams default is six bidirectional swaps; verify the
        full pattern preserves the swap semantics across all 10 cells."""
        pf_centers = np.arange(10) * 10.0
        remap_from_to = ((0, 9), (1, 8), (2, 7), (9, 0), (8, 1), (7, 2))
        result = get_remapped_pf_centers(pf_centers, remap_from_to, active=True)
        expected = np.array([90.0, 80.0, 70.0, 30.0, 40.0, 50.0, 60.0, 20.0, 10.0, 0.0])
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# decode_and_diagnostics
# ---------------------------------------------------------------------------


class TestDecodeAndDiagnostics:
    def test_full_output_contract(self, decoder_inputs: DecoderInputs) -> None:
        """Lock in the full set of returned keys + their shapes. Downstream
        plotting and cache code consumes the *whole* dict, so silently
        dropping any key (e.g. ``predictive``, ``per_spike_likelihood``,
        ``event_*``) is a regression even if the four headline metrics are
        intact."""
        result = decoder_inputs.call()
        n_time, n_cells = decoder_inputs.spikes.shape
        n_bins = decoder_inputs.xs.size
        # Events come from spikes[1:] (t=0 has no prior); count > 1 expands
        # to that many events (src/.../analysis.py:583).
        n_events = int(decoder_inputs.spikes[1:].sum())

        expected_shapes = {
            # Distributions over position (time × bins).
            "posterior": (n_time, n_bins),
            "predictive": (n_time, n_bins),
            "likelihood": (n_time, n_bins),
            "spike_likelihood": (n_time, n_bins),
            # Per-cell metric matrices.
            "hpd_overlap": (n_time, n_cells),
            "kl_divergence": (n_time, n_cells),
            "spike_prob": (n_time, n_cells),
            # Per-spike-event arrays (count expansion in src/.../analysis.py:584).
            "per_spike_likelihood": (n_events, n_bins),
            "spike_time_ind": (n_events,),
            "spike_cell_ind": (n_events,),
            "event_time_ind": (n_events,),
            "event_cell_ind": (n_events,),
            "event_hpd_overlap": (n_events,),
            "event_kl_divergence": (n_events,),
            "event_spike_prob": (n_events,),
        }
        # No missing keys, no surprise extras.
        assert set(result) == set(expected_shapes)
        for key, shape in expected_shapes.items():
            assert result[key].shape == shape, (
                f"{key} shape mismatch: got {result[key].shape}, want {shape}"
            )

    def test_t0_diagnostics_are_nan(self, decoder_inputs: DecoderInputs) -> None:
        """No prior exists at t=0, so all diagnostics are NaN."""
        result = decoder_inputs.call()
        for key in ("hpd_overlap", "kl_divergence", "spike_prob"):
            assert np.all(np.isnan(result[key][0]))

    def test_nan_pattern_matches_spike_pattern(self) -> None:
        """Diagnostics are NaN exactly where a cell has no spike at that time."""
        n_bins = 11
        spikes = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [2, 2]])
        xs = np.linspace(0, 100, n_bins)
        result = decode_and_diagnostics(
            spikes=spikes,
            xs=xs,
            transition_matrix=_diag_dominant_transition(n_bins),
            pf_centers=np.array([25.0, 75.0]),
            pf_width=5.0,
            rate_scale=0.1,
        )

        # Diagnostics NaN at (t, cell) iff cell has no spike at that t,
        # plus all of t=0 (no prior available).
        no_spike = spikes == 0
        no_spike[0] = True
        for key in ("hpd_overlap", "kl_divergence", "spike_prob"):
            np.testing.assert_array_equal(np.isnan(result[key]), no_spike)

    def test_count_greater_than_one_expands_to_multiple_events(self) -> None:
        n_bins = 11
        spikes = np.zeros((4, 2), dtype=int)
        spikes[1, 0] = 2
        spikes[2, 1] = 1

        result = decode_and_diagnostics(
            spikes=spikes,
            xs=np.linspace(0, 100, n_bins),
            transition_matrix=_diag_dominant_transition(n_bins),
            pf_centers=np.array([25.0, 75.0]),
            pf_width=5.0,
            rate_scale=0.1,
        )

        np.testing.assert_array_equal(result["spike_time_ind"], [1, 1, 2])
        np.testing.assert_array_equal(result["spike_cell_ind"], [0, 0, 1])
        assert result["event_kl_divergence"].shape == (3,)
        # The two count=2 spikes share a (time, cell), so their per-event
        # diagnostics must equal the matrix value at that bin.
        np.testing.assert_allclose(
            result["event_kl_divergence"][:2],
            np.repeat(result["kl_divergence"][1, 0], 2),
        )

    def test_no_spikes_produces_all_nan_diagnostics(self) -> None:
        """Edge case: zero spikes => all-NaN matrices and empty event arrays."""
        n_time, n_cells, n_bins = 5, 2, 11
        result = decode_and_diagnostics(
            spikes=np.zeros((n_time, n_cells), dtype=int),
            xs=np.linspace(0, 100, n_bins),
            transition_matrix=_diag_dominant_transition(n_bins),
            pf_centers=np.array([25.0, 75.0]),
            pf_width=5.0,
            rate_scale=0.1,
        )
        assert np.all(np.isnan(result["hpd_overlap"]))
        assert np.all(np.isnan(result["kl_divergence"]))
        assert np.all(np.isnan(result["spike_prob"]))
        assert result["event_time_ind"].shape == (0,)
        assert result["event_cell_ind"].shape == (0,)
        assert result["event_kl_divergence"].shape == (0,)

    def test_alternative_transition_matrix_used_only_inside_window(
        self,
        decoder_inputs: DecoderInputs,
    ) -> None:
        """A :class:`MisfitWindow` with a ``transition_matrix`` must
        (a) leave the predictive untouched before the window and
        (b) actually change it inside. A regression that ignored the
        schedule would still produce well-shaped output, so we compare
        against a baseline run instead.
        """
        n_bins = decoder_inputs.xs.size
        # Choose an alternative matrix that is *very* different from the
        # baseline (peak=0.9 diag-dominant). A near-uniform matrix forces
        # the predictive to spread out dramatically — easy to detect.
        alt_matrix = _diag_dominant_transition(n_bins, peak=0.05)
        window = (3, 6)
        schedule = MisfitSchedule(
            (MisfitWindow(window[0], window[1], transition_matrix=alt_matrix),)
        )

        baseline = decoder_inputs.call()
        with_alt = decoder_inputs.call(misfit_schedule=schedule)

        # Before the window, the two runs must be bit-identical: nothing
        # in the algorithm has diverged yet.
        np.testing.assert_array_equal(
            baseline["predictive"][: window[0]], with_alt["predictive"][: window[0]]
        )
        np.testing.assert_array_equal(
            baseline["posterior"][: window[0]], with_alt["posterior"][: window[0]]
        )

        # Inside the window, at least one timestep's predictive must
        # measurably differ (this is what the window actually controls).
        # Use a generous tolerance so we don't depend on the exact
        # magnitude — we only assert "different".
        inside = slice(*window)
        assert not np.allclose(
            baseline["predictive"][inside],
            with_alt["predictive"][inside],
            atol=1e-6,
        ), f"transition_matrix in {window} did not change predictive — schedule ignored?"

    def test_alt_rates_used_only_inside_window(self, decoder_inputs: DecoderInputs) -> None:
        """A :class:`MisfitWindow` with ``decoder_rates`` must change
        the per-spike likelihood inside the window and leave it
        untouched outside. Compared against a baseline run so a
        regression that ignored the schedule cannot pass on output
        shape alone.
        """
        # Flat-rate table that's clearly different from the Gaussian-PF
        # baseline — used here only to exercise the schedule plumbing.
        alt_rates = np.full((decoder_inputs.xs.size, 3), 0.05)
        window = (3, 7)
        schedule = MisfitSchedule((MisfitWindow(window[0], window[1], decoder_rates=alt_rates),))

        baseline = decoder_inputs.call()
        with_alt = decoder_inputs.call(misfit_schedule=schedule)

        evt_t = with_alt["event_time_ind"]
        # Events strictly before the window are unaffected — the filter
        # state has not diverged yet. (Events *after* the window
        # legitimately differ: the in-window posterior updates carry
        # forward, so we do not check those.)
        before = evt_t < window[0]
        inside = (evt_t >= window[0]) & (evt_t < window[1])

        # Pre-window per-spike likelihood is bit-identical: the
        # likelihood panel reads from the decoder's rate table and is
        # untouched by the schedule outside the window.
        np.testing.assert_array_equal(
            baseline["per_spike_likelihood"][before],
            with_alt["per_spike_likelihood"][before],
            err_msg="pre-window per_spike_likelihood differs — schedule leaked outside window",
        )

        # At least one in-window event's per-spike likelihood
        # row differs from the baseline. The schedule swaps the
        # decoder's rate table in-window, so the displayed likelihood
        # must change.
        assert inside.any(), "test fixture produced no in-window spike events"
        diff = with_alt["per_spike_likelihood"][inside] - baseline["per_spike_likelihood"][inside]
        assert np.any(np.abs(diff) > 0.0), (
            "alt rates produced no in-window change in per_spike_likelihood; "
            "the schedule may be near-noop."
        )


# ---------------------------------------------------------------------------
# MisfitWindow / MisfitSchedule
# ---------------------------------------------------------------------------


class TestMisfitWindow:
    def test_start_not_before_end_raises(self) -> None:
        """``start >= end`` is rejected at construction."""
        with pytest.raises(ValueError, match="start < end"):
            MisfitWindow(10, 10)
        with pytest.raises(ValueError, match="start < end"):
            MisfitWindow(10, 5)

    def test_negative_rate_table_raises(self) -> None:
        """A negative rate table is rejected before it can become NaN
        likelihoods downstream."""
        bad = np.full((21, 3), 0.05)
        bad[0, 0] = -1.0
        with pytest.raises(ValueError, match="finite and non-negative"):
            MisfitWindow(3, 7, decoder_rates=bad)

    def test_nonfinite_rate_table_raises(self) -> None:
        """A non-finite rate table is rejected at construction."""
        bad = np.full((21, 3), 0.05)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite and non-negative"):
            MisfitWindow(3, 7, decoder_rates=bad)


class TestMisfitSchedule:
    def test_empty_schedule_has_no_window_anywhere(self) -> None:
        assert MisfitSchedule().window_at(0) is None
        assert MisfitSchedule().window_at(10_000) is None

    def test_window_at_returns_containing_window(self) -> None:
        sched = MisfitSchedule((MisfitWindow(10, 20), MisfitWindow(30, 40)))
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
            MisfitSchedule((MisfitWindow(10, 25), MisfitWindow(20, 30)))


# ---------------------------------------------------------------------------
# Thresholds / compute_thresholds
# ---------------------------------------------------------------------------


class TestComputeThresholds:
    def test_thresholds_match_quantile_definitions(self, metrics_2d: dict[str, np.ndarray]) -> None:
        baseline_end = 50
        thresholds = compute_thresholds(metrics_2d, baseline_end=baseline_end)

        expected_hpdo = np.nanquantile(metrics_2d["hpd_overlap"][:baseline_end].ravel(), 0.01)
        expected_kl = np.nanquantile(metrics_2d["kl_divergence"][:baseline_end].ravel(), 0.99)
        assert thresholds.hpd_overlap == pytest.approx(expected_hpdo)
        assert thresholds.kl_divergence == pytest.approx(expected_kl)
        # spike_prob is fixed at 0.05 (matches MATLAB), not data-driven.
        assert thresholds.spike_prob == 0.05

    def test_handles_partial_nan_baseline(self) -> None:
        """NaNs in the baseline must be ignored, not propagate to thresholds."""
        n_time, n_cells = 20, 3
        hpdo = np.full((n_time, n_cells), 0.8)
        hpdo[:5] = np.nan
        metrics: dict[str, np.ndarray] = {
            "hpd_overlap": hpdo,
            "kl_divergence": np.full((n_time, n_cells), 1.0),
            "spike_prob": np.full((n_time, n_cells), 0.5),
        }
        thresholds = compute_thresholds(metrics, baseline_end=10)
        assert not np.isnan(thresholds.hpd_overlap)
        assert not np.isnan(thresholds.kl_divergence)

    def test_baseline_end_is_keyword_only(self, metrics_2d: dict[str, np.ndarray]) -> None:
        """Passing baseline_end positionally must fail — the argument is
        keyword-only so callers can't accidentally omit it via the prior
        ``None`` default that silently used the whole recording."""
        # Cast to Any to probe the runtime contract without the static
        # type checker rejecting the deliberately-wrong call.
        unchecked: Any = compute_thresholds
        with pytest.raises(TypeError, match="positional"):
            unchecked(metrics_2d, 50)

    def test_all_nan_hpd_baseline_raises(self) -> None:
        """An all-NaN baseline slice would produce a NaN threshold and
        every downstream ``metric < threshold`` comparison would silently
        evaluate False. Raise instead."""
        n_time, n_cells = 20, 3
        metrics: dict[str, np.ndarray] = {
            "hpd_overlap": np.full((n_time, n_cells), np.nan),
            "kl_divergence": np.full((n_time, n_cells), 1.0),
            "spike_prob": np.full((n_time, n_cells), 0.5),
        }
        with pytest.raises(ValueError, match="hpd_overlap baseline slice"):
            compute_thresholds(metrics, baseline_end=10)

    def test_all_nan_kl_baseline_raises(self) -> None:
        n_time, n_cells = 20, 3
        metrics: dict[str, np.ndarray] = {
            "hpd_overlap": np.full((n_time, n_cells), 0.8),
            "kl_divergence": np.full((n_time, n_cells), np.nan),
            "spike_prob": np.full((n_time, n_cells), 0.5),
        }
        with pytest.raises(ValueError, match="kl_divergence baseline slice"):
            compute_thresholds(metrics, baseline_end=10)


# ---------------------------------------------------------------------------
# Transformed / transform_metrics
# ---------------------------------------------------------------------------


class TestTransformMetrics:
    def test_transformations_match_documented_formulas(self) -> None:
        metrics = {
            "hpd_overlap": np.array([[0.5, 0.8], [0.9, 0.7]]),
            "kl_divergence": np.array([[1.0, 4.0], [9.0, 16.0]]),
            "spike_prob": np.array([[0.1, 0.5], [0.01, 0.05]]),
        }
        thresholds = Thresholds(hpd_overlap=0.6, kl_divergence=5.0, spike_prob=0.05)
        eps1, eps2 = 1e-2, 1e-10

        transformed = transform_metrics(metrics, thresholds, eps1=eps1, eps2=eps2)

        np.testing.assert_allclose(
            transformed.hpd_overlap, -np.log10(metrics["hpd_overlap"] + eps1)
        )
        np.testing.assert_allclose(transformed.kl_divergence, np.sqrt(metrics["kl_divergence"]))
        np.testing.assert_allclose(transformed.spike_prob, -np.log10(metrics["spike_prob"] + eps2))
        assert transformed.hpd_overlap_threshold == pytest.approx(
            -np.log10(thresholds.hpd_overlap + eps1)
        )
        assert transformed.kl_divergence_threshold == pytest.approx(
            np.sqrt(thresholds.kl_divergence)
        )
        assert transformed.spike_prob_threshold == pytest.approx(
            -np.log10(thresholds.spike_prob + eps2)
        )

    def test_nan_inputs_propagate_to_outputs(self) -> None:
        """NaNs in metrics must remain NaN after transformation."""
        nan_mask = np.array([[False, True], [False, False]])
        metrics = {
            "hpd_overlap": np.where(nan_mask, np.nan, 0.5),
            "kl_divergence": np.where(nan_mask, np.nan, 1.0),
            "spike_prob": np.where(nan_mask, np.nan, 0.05),
        }
        thresholds = Thresholds(hpd_overlap=0.6, kl_divergence=5.0, spike_prob=0.05)
        transformed = transform_metrics(metrics, thresholds)
        assert np.array_equal(np.isnan(transformed.hpd_overlap), nan_mask)
        assert np.array_equal(np.isnan(transformed.kl_divergence), nan_mask)
        assert np.array_equal(np.isnan(transformed.spike_prob), nan_mask)

    def test_default_eps_yields_finite_outputs(self) -> None:
        metrics = {
            "hpd_overlap": np.array([[0.5, 0.8]]),
            "kl_divergence": np.array([[1.0, 4.0]]),
            "spike_prob": np.array([[0.1, 0.05]]),
        }
        thresholds = Thresholds(hpd_overlap=0.6, kl_divergence=5.0, spike_prob=0.05)
        transformed = transform_metrics(metrics, thresholds)
        assert np.isfinite(transformed.hpd_overlap).all()
        assert np.isfinite(transformed.kl_divergence).all()
        assert np.isfinite(transformed.spike_prob).all()


class TestTransformedDataclass:
    def test_construction_preserves_arrays_and_thresholds(self) -> None:
        hpd_overlap = np.array([[1.0, 2.0]])
        transformed = Transformed(
            hpd_overlap=hpd_overlap,
            kl_divergence=hpd_overlap.copy(),
            spike_prob=hpd_overlap.copy(),
            hpd_overlap_threshold=1.5,
            kl_divergence_threshold=1.0,
            spike_prob_threshold=3.0,
        )
        np.testing.assert_array_equal(transformed.hpd_overlap, hpd_overlap)
        assert transformed.hpd_overlap_threshold == 1.5
        assert transformed.kl_divergence_threshold == 1.0
        assert transformed.spike_prob_threshold == 3.0


# ---------------------------------------------------------------------------
# _condition_on (Bayesian update helper) and the log-space decode_and_diagnostics
# ---------------------------------------------------------------------------


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
        ``decode_and_diagnostics`` then surfaces the count via a
        ``RuntimeWarning``.
        """
        n_bins = 5
        prior = np.full(n_bins, 1.0 / n_bins)
        ll = np.full(n_bins, -np.inf)
        new_probs, log_norm = _condition_on(prior, ll)
        # Uniform fallback, properly normalized.
        np.testing.assert_allclose(new_probs, 1.0 / n_bins, rtol=1e-12)
        np.testing.assert_allclose(new_probs.sum(), 1.0, rtol=1e-12)
        # Underflow signal is exactly -inf so callers can ``== -np.inf``.
        assert log_norm == -np.inf

    def test_handles_prior_likelihood_disjoint(self) -> None:
        """A finite but vanishingly small overlap also takes the
        fallback path: ``weighted.sum() < eps`` triggers the explicit
        uniform reset. Prior concentrated at one end, likelihood at the
        other.
        """
        n_bins = 8
        prior = np.zeros(n_bins)
        prior[0] = 1.0
        ll = np.full(n_bins, -1000.0)
        ll[-1] = 0.0  # likelihood mass at the opposite end of the grid
        new_probs, log_norm = _condition_on(prior, ll)
        np.testing.assert_allclose(new_probs, 1.0 / n_bins, rtol=1e-12)
        assert log_norm == -np.inf

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


class TestDecodeAndDiagnosticsLogSpace:
    """Stress tests for the log-space rewrite of decode_and_diagnostics.

    The previous implementation reset the posterior to uniform when
    ``prior * combined_likelihood`` underflowed to zero. The log-space
    rewrite removes that branch; the failure mode it guarded against
    cannot occur. These tests pin that property so a future refactor
    can't silently reintroduce the reset.
    """

    def test_posterior_sums_to_one_at_every_step(self) -> None:
        """Algorithmic correctness: at every time step the posterior
        must be a proper probability distribution.
        """
        rng = np.random.default_rng(0)
        n_time, n_cells, n_bins = 50, 3, 21
        spikes = rng.poisson(1.0, size=(n_time, n_cells))
        xs = np.linspace(0.0, 100.0, n_bins)
        transition_matrix = gaussian_transition_matrix(xs, sig=2.0)
        pf_centers = np.array([25.0, 50.0, 75.0])
        results = decode_and_diagnostics(
            spikes, xs, transition_matrix, pf_centers, pf_width=10.0, rate_scale=0.1
        )
        posterior = results["posterior"]
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12)

    def test_extreme_prior_likelihood_mismatch_yields_meaningful_posterior(self) -> None:
        """Stress test for the underflow regime: place a narrow prior at
        one end of the grid then drive the decoder with spikes whose
        place-field rate is concentrated at the *other* end.

        On the pre-refactor code this configuration triggered the
        ``posterior_sum < 1e-300`` reset-to-uniform branch. The
        log-space implementation must instead produce a normalized,
        non-uniform posterior at every step.
        """
        n_time, n_cells, n_bins = 30, 2, 51
        xs = np.linspace(0.0, 100.0, n_bins)
        # Narrow transition kernel so the prior stays concentrated.
        transition_matrix = gaussian_transition_matrix(xs, sig=0.5)
        # Two cells with place fields at x≈90 — far from the
        # bias-initialized posterior which mostly accumulates near 0.
        pf_centers = np.array([88.0, 92.0])
        # Both cells fire every time step → strong likelihood signal at
        # x≈90.
        spikes = np.ones((n_time, n_cells), dtype=np.int_)

        results = decode_and_diagnostics(
            spikes, xs, transition_matrix, pf_centers, pf_width=2.0, rate_scale=5.0
        )
        posterior = results["posterior"]

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
        peak_pos = xs[peak_bin]
        assert peak_pos > 80.0, (
            f"final posterior peak at x={peak_pos:.1f}, expected near 90 "
            f"under sustained firing at PF centers (88, 92)."
        )

    def test_underflow_emits_summary_warning(self) -> None:
        """The per-step ``_condition_on`` ``-inf`` path is covered by
        ``TestConditionOn``; the *summary* warning emitted by
        ``decode_and_diagnostics`` post-loop is not asserted anywhere.

        Force the underflow path by placing the cell's gaussian place
        field so far from the decoder grid that every bin's rate
        underflows to exactly 0.0. ``poisson.logpmf(k=1, mu=0) = -inf``
        at every bin → ``ll_max = -inf`` → ``_condition_on`` returns
        the uniform-fallback sentinel at every spike timestep.

        Reverting the post-loop warning emit makes this test fail,
        leaving the operator with no signal that some posterior steps
        ran the uniform fallback.
        """
        n_time, n_cells, n_bins = 8, 1, 21
        xs = np.linspace(0.0, 100.0, n_bins)
        transition_matrix = gaussian_transition_matrix(xs, sig=2.0)
        # PF center so far from xs that exp(-d^2 / 2*pf_width^2)
        # underflows to exactly 0.0 — every bin's rate is 0.0.
        pf_centers = np.array([1e6])
        spikes = np.ones((n_time, n_cells), dtype=np.int_)

        with pytest.warns(
            RuntimeWarning,
            match=r"prior/likelihood overlap underflowed at \d+ timestep",
        ):
            decode_and_diagnostics(
                spikes, xs, transition_matrix, pf_centers, pf_width=5.0, rate_scale=1.0
            )


# ---------------------------------------------------------------------------
# compute_per_cell_diagnostics_from_rates
# ---------------------------------------------------------------------------


class TestComputePerCellDiagnosticsFromRates:
    """Direct tests for the per-cell diagnostics helper.

    The zero-rate-row branch at analysis.py:1244-1249 fills uniform
    ``1/n_cells`` on bins where every cell's expected rate is zero
    (sparse real-data coverage). The branch was added during the
    previous review cycle but had no direct test — the function was
    only exercised transitively through ``decode_and_diagnostics``,
    where synthetic Gaussian place-field rates never produce a zero
    row.

    Note: only ``event_spike_prob`` depends on ``cell_fraction_per_bin``,
    which is what the fix protects. ``event_hpd_overlap`` and
    ``event_kl_divergence`` compute against the per-cell Poisson
    likelihood directly and can still be ``inf`` when the predictive
    has mass at a bin where the cell has zero rate — that is correct
    behavior, not the bug the zero-rate-row branch addresses.
    """

    def test_zero_rate_row_keeps_event_spike_prob_finite(self) -> None:
        n_time, n_bins, n_cells = 10, 5, 3
        rng = np.random.default_rng(0)
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)

        # One bin row is entirely zero — the previously-broken path.
        rates = np.full((n_bins, n_cells), 0.5)
        rates[2, :] = 0.0

        spike_time_ind = np.array([0, 1, 5], dtype=np.intp)
        spike_cell_ind = np.array([0, 1, 2], dtype=np.intp)

        result = compute_per_cell_diagnostics_from_rates(
            predictive, rates, spike_time_ind, spike_cell_ind, coverage=0.95
        )
        # Reverting the if-zero-rows-fill-uniform fix lets the zero
        # row reach normalize() and produces a non-distribution row in
        # cell_fraction_per_bin, which propagates to event_spike_prob.
        assert np.all(np.isfinite(result["event_spike_prob"]))

    def test_fully_degenerate_rates_still_produces_finite_spike_prob(self) -> None:
        """Pathological case: every rate row is zero. The uniform
        fallback covers every row, so ``event_spike_prob`` stays finite
        even though the result is statistically meaningless. Catches a
        regression that would let ``normalize``'s eps-clamp leak through."""
        n_time, n_bins, n_cells = 5, 3, 2
        predictive = np.full((n_time, n_bins), 1.0 / n_bins)
        rates = np.zeros((n_bins, n_cells))
        spike_time_ind = np.array([0], dtype=np.intp)
        spike_cell_ind = np.array([0], dtype=np.intp)
        result = compute_per_cell_diagnostics_from_rates(
            predictive, rates, spike_time_ind, spike_cell_ind, coverage=0.95
        )
        assert np.all(np.isfinite(result["event_spike_prob"]))


# ---------------------------------------------------------------------------
# Phase 2 invariants: phase_boundaries tuple, MisfitWindow tightening
# ---------------------------------------------------------------------------


class TestDecodeParamsPhaseBoundaries:
    """The phase ladder collapsed from 8 ``T_*`` fields to one
    ``phase_boundaries`` tuple. Lock the invariants in.
    """

    def test_default_boundaries_match_documented_defaults(self) -> None:
        params = DecodeParams()
        assert params.phase_boundaries == (
            6_000,
            10_000,
            14_000,
            18_000,
            22_000,
            26_000,
            30_000,
            32_000,
        )

    def test_phase_boundaries_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must have 8 entries"):
            DecodeParams(phase_boundaries=(1, 2, 3))

    def test_phase_boundaries_non_monotonic_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            DecodeParams(
                phase_boundaries=(100, 200, 200, 400, 500, 600, 700, 800),
            )

    def test_phase_boundaries_equal_consecutive_raises(self) -> None:
        """Equal consecutive entries (zero-width phase) must reject too."""
        with pytest.raises(ValueError, match="strictly increasing"):
            DecodeParams(
                phase_boundaries=(100, 200, 300, 300, 500, 600, 700, 800),
            )

    @pytest.mark.parametrize(
        ("name", "index"),
        [
            ("T_remap_start", 0),
            ("T_remap_end", 1),
            ("T_recovery1_end", 2),
            ("T_hist_dep_end", 3),
            ("T_recovery2_end", 4),
            ("T_drift_end", 5),
            ("T_recovery3_end", 6),
            ("T_wide_dynamics_end", 7),
        ],
    )
    def test_named_accessor_indexes_into_tuple(self, name: str, index: int) -> None:
        boundaries = (100, 200, 300, 400, 500, 600, 700, 800)
        params = DecodeParams(phase_boundaries=boundaries)
        assert getattr(params, name) == boundaries[index]


class TestMisfitWindowTightening:
    """Write-protect, shape validation invariants."""

    @pytest.mark.parametrize(
        ("field", "kwargs_factory"),
        [
            ("decoder_rates", lambda arr: {"decoder_rates": arr}),
            ("transition_matrix", lambda arr: {"transition_matrix": arr}),
        ],
    )
    def test_arrays_are_write_protected(self, field: str, kwargs_factory: Any) -> None:
        """``frozen=True`` only blocks rebinding the field; the array
        itself must also be read-only so callers can't bypass
        validation by mutating in place."""
        arr = np.eye(5) * 0.5 + 0.1 if field == "transition_matrix" else np.full((5, 3), 0.1)
        w = MisfitWindow(10, 20, **kwargs_factory(arr))
        stored = getattr(w, field)
        assert stored is not None
        assert stored.flags.writeable is False
        with pytest.raises(ValueError, match="read-only|assignment destination"):
            stored[0, 0] = 999.0

    def test_caller_array_not_mutated_by_construction(self) -> None:
        """Defensive copy: caller's original array stays writable."""
        rates = np.full((5, 3), 0.1)
        original_id = id(rates)
        MisfitWindow(10, 20, decoder_rates=rates)
        assert id(rates) == original_id
        assert rates.flags.writeable is True

    def test_validate_against_accepts_matching_shape(self) -> None:
        rates = np.full((5, 3), 0.1)
        w = MisfitWindow(10, 20, decoder_rates=rates)
        w.validate_against(n_bins=5, n_cells=3)  # does not raise

    def test_validate_against_raises_on_mismatched_decoder_rates_shape(self) -> None:
        rates = np.full((5, 3), 0.1)
        w = MisfitWindow(10, 20, decoder_rates=rates)
        with pytest.raises(ValueError, match=r"decoder_rates shape"):
            w.validate_against(n_bins=7, n_cells=3)

    def test_validate_against_raises_on_mismatched_transition_shape(self) -> None:
        transition = np.eye(5)
        w = MisfitWindow(10, 20, transition_matrix=transition)
        with pytest.raises(ValueError, match=r"transition_matrix shape"):
            w.validate_against(n_bins=7, n_cells=3)


class TestSimulationResultDataclass:
    """The ``TypedDict`` → frozen-dataclass conversion brought length
    validation. Cover the success contract and the failure modes."""

    def test_valid_construction_succeeds(self) -> None:
        """Happy path: a well-formed SimulationResult constructs cleanly,
        coerces list inputs to tuple, and exposes attribute access on
        every field."""

        n_bins = 5
        n_time = 10
        sim = SimulationResult(
            params=DecodeParams(),
            xs=np.linspace(0.0, 100.0, n_bins),
            x_true=np.zeros(n_time),
            spikes=np.zeros((n_time, 1), dtype=np.int_),
            metrics={},
            phase_labels=PHASE_LABELS,
            phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
        )
        # Sequence fields coerced to tuple by __post_init__.
        assert isinstance(sim.phase_labels, tuple)
        assert isinstance(sim.phase_boundaries, tuple)
        # Attribute access works (the migration test).
        assert sim.xs.shape == (n_bins,)
        assert sim.x_true.shape == (n_time,)

    def test_phase_labels_wrong_order_raises(self) -> None:
        n_bins = 5
        n_time = 10
        bogus_labels = tuple(reversed(PHASE_LABELS))
        with pytest.raises(ValueError, match="phase_labels must equal PHASE_LABELS"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics={},
                phase_labels=bogus_labels,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )

    def test_phase_boundary_length_mismatch_raises(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="phase_boundaries length"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics={},
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3),  # wrong length
            )

    def test_spikes_and_x_true_timeline_mismatch_raises(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="spikes timeline"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time + 1, 1), dtype=np.int_),  # off by one
                metrics={},
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )

    def test_final_boundary_must_equal_timeline_length(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="final phase boundary"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics={},
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time + 1),
            )

    def test_metrics_with_wrong_time_indexed_leading_dim_raises(self) -> None:
        """The ``TIME_INDEXED_METRIC_KEYS`` loop in ``__post_init__``
        rejects a dense metric array whose leading dim doesn't match
        ``x_true``'s timeline. Reverting that loop lets a mis-shaped
        ``posterior`` slip through and the figure-3 pipeline runs with
        misaligned indices."""
        n_bins = 5
        n_time = 10
        # Wrong leading dim: posterior is one row longer than x_true.
        bad_metrics: dict[str, np.ndarray] = {
            "posterior": np.zeros((n_time + 1, n_bins)),
        }
        with pytest.raises(ValueError, match=r"metrics\['posterior'\] has leading dim"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics=bad_metrics,
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )


# ---------------------------------------------------------------------------
# Phase 3 follow-ups: stored-likelihood normalization, end-to-end
# validate_against, log-space reference comparison
# ---------------------------------------------------------------------------


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
        likelihood = result["likelihood"]
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
        spike_lik = result["spike_likelihood"]
        spikes = decoder_inputs.spikes
        spike_steps = np.where(spikes.sum(axis=1) > 0)[0]
        # Skip t=0 which is a flat-initialized row, not a likelihood
        # over any observation.
        spike_steps = spike_steps[spike_steps > 0]
        assert spike_steps.size, "test fixture produced no spikes"
        sums = spike_lik[spike_steps].sum(axis=1)
        np.testing.assert_allclose(
            sums,
            1.0,
            rtol=1e-10,
            atol=1e-12,
            err_msg="spike_likelihood_all is not row-normalized at spike steps",
        )


class TestDecoderValidatesScheduleShapes:
    """``decode_and_diagnostics`` must invoke ``validate_against`` on
    every schedule entry before running the time loop. A regression
    that drops the call would only fail later with a cryptic
    broadcasting error inside ``poisson.logpmf``.
    """

    def test_mismatched_decoder_rates_raises_before_decode(
        self, decoder_inputs: DecoderInputs
    ) -> None:
        n_bins = decoder_inputs.xs.size
        wrong_shape = np.full((n_bins + 3, decoder_inputs.spikes.shape[1]), 0.1)
        schedule = MisfitSchedule((MisfitWindow(2, 5, decoder_rates=wrong_shape),))
        with pytest.raises(ValueError, match=r"decoder_rates shape"):
            decoder_inputs.call(misfit_schedule=schedule)

    def test_mismatched_transition_matrix_raises_before_decode(
        self, decoder_inputs: DecoderInputs
    ) -> None:
        n_bins = decoder_inputs.xs.size
        wrong_transition = np.eye(n_bins + 1)
        schedule = MisfitSchedule((MisfitWindow(2, 5, transition_matrix=wrong_transition),))
        with pytest.raises(ValueError, match=r"transition_matrix shape"):
            decoder_inputs.call(misfit_schedule=schedule)


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

        from statespacecheck_paper.simulation import placefield_rates

        spikes = decoder_inputs.spikes
        xs = decoder_inputs.xs
        transition = decoder_inputs.transition_matrix
        n_time, _ = spikes.shape
        n_bins = xs.size

        # Reference: linear-space prior, log-space combined likelihood,
        # softmax-shift normalization. No reset-to-uniform branch.
        rates = placefield_rates(
            xs,
            decoder_inputs.pf_centers,
            decoder_inputs.pf_width,
            decoder_inputs.rate_scale,
        )
        ref_post = np.zeros((n_time, n_bins))
        ref_post[0] = np.ones(n_bins) / n_bins
        for t in range(1, n_time):
            prior = ref_post[t - 1] @ transition
            prior = prior / prior.sum()
            log_lik = _poisson.logpmf(spikes[t][None, :], rates).sum(axis=1)
            ll_max = float(np.max(log_lik))
            assert np.isfinite(ll_max)
            weighted = prior * np.exp(log_lik - ll_max)
            norm = weighted.sum()
            assert norm > 0
            ref_post[t] = weighted / norm

        result = decoder_inputs.call()
        np.testing.assert_allclose(result["posterior"], ref_post, rtol=1e-10, atol=1e-12)
