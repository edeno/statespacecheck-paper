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
    _merge_diagnostics,
    compute_thresholds,
    decode_and_diagnostics,
    get_remapped_pf_centers,
    likelihood_grid_for_counts,
    transform_metrics,
)

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
        """A :class:`MisfitWindow` with ``decoder_rates``/``diagnostic_rates``
        must leave per-event diagnostics untouched for events outside the
        window and change at least one event inside it. Compared against a
        baseline run so a regression that ignored the schedule cannot pass
        on output shape alone.
        """
        # Flat-rate table that's clearly different from the Gaussian-PF
        # baseline — used here only to exercise the schedule plumbing.
        alt_rates = np.full((decoder_inputs.xs.size, 3), 0.05)
        window = (3, 7)
        schedule = MisfitSchedule(
            (
                MisfitWindow(
                    window[0],
                    window[1],
                    decoder_rates=alt_rates,
                    diagnostic_rates=alt_rates,
                ),
            )
        )

        baseline = decoder_inputs.call()
        with_alt = decoder_inputs.call(misfit_schedule=schedule)

        evt_t = with_alt["event_time_ind"]
        # Events strictly before the window are unaffected — the filter
        # state has not diverged yet. (Events *after* the window
        # legitimately differ: the in-window posterior updates carry
        # forward, so we do not check those.)
        before = evt_t < window[0]
        inside = (evt_t >= window[0]) & (evt_t < window[1])

        np.testing.assert_array_equal(
            baseline["event_kl_divergence"][before],
            with_alt["event_kl_divergence"][before],
        )
        # At least one in-window event's KL changed *meaningfully* (median
        # |Δ| above 0.01). Bare ``not np.allclose`` would fire on float
        # noise too; the magnitude bound catches a near-noop schedule that
        # still produces float-different outputs.
        assert inside.any(), "test fixture produced no in-window spike events"
        diff = with_alt["event_kl_divergence"][inside] - baseline["event_kl_divergence"][inside]
        median_abs_diff = float(np.median(np.abs(diff)))
        assert median_abs_diff > 0.01, (
            f"alt rates produced trivially small in-window diagnostic change; "
            f"median |Δ KL| = {median_abs_diff:.4f}. The schedule may be near-noop."
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

    @pytest.mark.parametrize("field", ["decoder_rates", "diagnostic_rates"])
    def test_negative_rate_table_raises(self, field: str) -> None:
        """A negative rate table is rejected before it can become NaN
        likelihoods downstream."""
        bad = np.full((21, 3), 0.05)
        bad[0, 0] = -1.0
        with pytest.raises(ValueError, match="finite and non-negative"):
            MisfitWindow(3, 7, **{field: bad})

    @pytest.mark.parametrize("field", ["decoder_rates", "diagnostic_rates"])
    def test_nonfinite_rate_table_raises(self, field: str) -> None:
        """A non-finite rate table is rejected at construction."""
        bad = np.full((21, 3), 0.05)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite and non-negative"):
            MisfitWindow(3, 7, **{field: bad})


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
# _merge_diagnostics
# ---------------------------------------------------------------------------


class TestMergeDiagnostics:
    @staticmethod
    def _batch(values: list[float]) -> dict[str, np.ndarray]:
        """A minimal per-event diagnostic batch with distinct sentinel values."""
        arr = np.asarray(values, dtype=float)
        return {
            "event_hpd_overlap": arr,
            "event_kl_divergence": arr + 100.0,
            "event_spike_prob": arr + 200.0,
        }

    def test_mixed_groups_reassemble_in_original_event_order(self) -> None:
        """Events split across rate-table groups are scattered back in order."""
        spike_time_ind = np.array([1, 2, 3, 4], dtype=np.intp)
        spike_cell_ind = np.array([0, 1, 0, 1], dtype=np.intp)
        # Group 0 covers events 0 and 3; group 1 covers events 1 and 2.
        event_group = np.array([0, 1, 1, 0], dtype=np.intp)
        per_group = [self._batch([10.0, 13.0]), self._batch([11.0, 12.0])]

        merged = _merge_diagnostics(
            n_time=5,
            n_cells=2,
            spike_time_ind=spike_time_ind,
            spike_cell_ind=spike_cell_ind,
            event_group=event_group,
            per_group=per_group,
        )
        # Original event order is 10, 11, 12, 13.
        np.testing.assert_array_equal(merged["event_hpd_overlap"], [10.0, 11.0, 12.0, 13.0])
        # Dense matrix carries each event at its (time, cell) slot; all
        # other slots stay NaN.
        dense = merged["hpd_overlap"]
        assert dense.shape == (5, 2)
        assert dense[1, 0] == 10.0
        assert dense[2, 1] == 11.0
        assert dense[3, 0] == 12.0
        assert dense[4, 1] == 13.0
        assert np.isnan(dense[0, 0])

    @pytest.mark.parametrize("all_group1", [True, False])
    def test_all_events_in_one_group(self, all_group1: bool) -> None:
        """A grouping that puts every event in one group still merges
        correctly, with the other group empty."""
        n = 3
        spike_time_ind = np.arange(1, n + 1, dtype=np.intp)
        spike_cell_ind = np.zeros(n, dtype=np.intp)
        event_group = np.full(n, 1 if all_group1 else 0, dtype=np.intp)
        populated = self._batch([1.0, 2.0, 3.0])
        empty = self._batch([])
        per_group = [empty, populated] if all_group1 else [populated, empty]
        merged = _merge_diagnostics(
            n_time=n + 1,
            n_cells=1,
            spike_time_ind=spike_time_ind,
            spike_cell_ind=spike_cell_ind,
            event_group=event_group,
            per_group=per_group,
        )
        np.testing.assert_array_equal(merged["event_kl_divergence"], [101.0, 102.0, 103.0])

    def test_length_mismatch_raises(self) -> None:
        """A batch whose event count disagrees with its group fails loud."""
        spike_time_ind = np.array([1, 2, 3], dtype=np.intp)
        spike_cell_ind = np.zeros(3, dtype=np.intp)
        event_group = np.array([0, 1, 1], dtype=np.intp)
        with pytest.raises(ValueError, match="group 1 has 1 events but the mask expects 2"):
            _merge_diagnostics(
                n_time=4,
                n_cells=1,
                spike_time_ind=spike_time_ind,
                spike_cell_ind=spike_cell_ind,
                event_group=event_group,
                per_group=[self._batch([10.0]), self._batch([11.0])],  # group 1 expects 2
            )


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
        """All -inf likelihoods would be a degenerate observation (no
        bin has finite probability). The helper should not produce NaN
        — it falls back via the ll_max guard and the eps in the
        denominator.
        """
        n_bins = 5
        prior = np.full(n_bins, 1.0 / n_bins)
        ll = np.full(n_bins, -np.inf)
        new_probs, log_norm = _condition_on(prior, ll)
        # Result is finite — no NaN, no inf.
        assert np.all(np.isfinite(new_probs))
        # In this degenerate case the helper returns the prior re-scaled
        # by 1.0 (since exp(-inf - 0) = 0 everywhere → weighted = 0 →
        # new_probs = 0 / eps = 0; but ll_max is masked to 0 by the
        # finite-check). Acceptable failure mode; document via the
        # assertion that the math at least doesn't propagate NaN.
        assert np.isfinite(log_norm) or log_norm == -np.inf

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
        from statespacecheck_paper.simulation import gaussian_transition_matrix

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
        from statespacecheck_paper.simulation import gaussian_transition_matrix

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
    """Phase-2 additions: write-protect, shape validation, pairing invariant."""

    def test_diagnostic_rates_without_decoder_rates_raises(self) -> None:
        bad = np.full((5, 3), 0.1)
        with pytest.raises(
            ValueError,
            match="diagnostic_rates is set but decoder_rates is None",
        ):
            MisfitWindow(10, 20, diagnostic_rates=bad)

    def test_decoder_rates_array_is_write_protected(self) -> None:
        """``frozen=True`` only blocks rebinding the field; the array
        itself must also be read-only so callers can't bypass
        validation by mutating in place."""
        rates = np.full((5, 3), 0.1)
        w = MisfitWindow(10, 20, decoder_rates=rates)
        assert w.decoder_rates is not None
        assert w.decoder_rates.flags.writeable is False
        with pytest.raises(ValueError, match="read-only|assignment destination"):
            w.decoder_rates[0, 0] = 999.0

    def test_caller_array_not_mutated_by_construction(self) -> None:
        """Defensive copy: caller's original array stays writable."""
        rates = np.full((5, 3), 0.1)
        original_id = id(rates)
        MisfitWindow(10, 20, decoder_rates=rates)
        assert id(rates) == original_id
        assert rates.flags.writeable is True

    def test_validate_against_accepts_matching_shape(self) -> None:
        rates = np.full((5, 3), 0.1)
        w = MisfitWindow(10, 20, decoder_rates=rates, diagnostic_rates=rates)
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
    validation. Cover the failure modes."""

    def test_phase_labels_wrong_order_raises(self) -> None:
        from statespacecheck_paper.figure03_demo import PHASE_LABELS, SimulationResult

        n_bins = 5
        n_time = 10
        bogus_labels = list(reversed(PHASE_LABELS))
        with pytest.raises(ValueError, match="phase_labels must equal PHASE_LABELS"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics={},
                phase_labels=bogus_labels,
                phase_boundaries=[1, 2, 3, 4, 5, 6, 7, n_time],
            )

    def test_phase_boundary_length_mismatch_raises(self) -> None:
        from statespacecheck_paper.figure03_demo import PHASE_LABELS, SimulationResult

        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="phase_boundaries length"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics={},
                phase_labels=list(PHASE_LABELS),
                phase_boundaries=[1, 2, 3],  # wrong length
            )

    def test_spikes_and_x_true_timeline_mismatch_raises(self) -> None:
        from statespacecheck_paper.figure03_demo import PHASE_LABELS, SimulationResult

        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="spikes timeline"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time + 1, 1), dtype=np.int_),  # off by one
                metrics={},
                phase_labels=list(PHASE_LABELS),
                phase_boundaries=[1, 2, 3, 4, 5, 6, 7, n_time],
            )

    def test_final_boundary_must_equal_timeline_length(self) -> None:
        from statespacecheck_paper.figure03_demo import PHASE_LABELS, SimulationResult

        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="final phase boundary"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics={},
                phase_labels=list(PHASE_LABELS),
                phase_boundaries=[1, 2, 3, 4, 5, 6, 7, n_time + 1],
            )
