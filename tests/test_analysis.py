"""Tests for analysis module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.analysis import (
    DecodeParams,
    Thresholds,
    Transformed,
    _merge_diagnostics,
    compute_thresholds,
    decode_and_diagnostics,
    get_remapped_pf_centers,
    likelihood_grid_for_counts,
    transform_metrics,
)
from statespacecheck_paper.simulation import wiggly_flat_rates

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
    remap_window: tuple[int, int]
    remap_from_to: tuple[int, int]

    def call(self, **overrides: Any) -> dict:
        kwargs: dict[str, Any] = {
            "spikes": self.spikes,
            "xs": self.xs,
            "transition_matrix": self.transition_matrix,
            "pf_centers": self.pf_centers,
            "pf_width": self.pf_width,
            "rate_scale": self.rate_scale,
            "remap_window": self.remap_window,
            "remap_from_to": self.remap_from_to,
        }
        kwargs.update(overrides)
        return decode_and_diagnostics(**kwargs)


def _diag_dominant_transition(n_bins: int, peak: float = 0.9) -> np.ndarray:
    return np.eye(n_bins) * peak + (1.0 - peak) / n_bins


@pytest.fixture
def decoder_inputs() -> DecoderInputs:
    """Small reproducible decoder problem with no remapping inside the run."""
    rng = np.random.default_rng(42)
    n_time, n_cells, n_bins = 10, 3, 21
    return DecoderInputs(
        spikes=rng.poisson(1.0, size=(n_time, n_cells)),
        xs=np.linspace(0, 100, n_bins),
        transition_matrix=_diag_dominant_transition(n_bins),
        pf_centers=np.array([25.0, 50.0, 75.0]),
        pf_width=5.0,
        rate_scale=0.1,
        remap_window=(20, 20),  # outside range -> no remap
        remap_from_to=(0, 1),
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
        params = DecodeParams(T_remap_start=1000, T_remap_end=2000)
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
            remap_window=(10, 10),
            remap_from_to=(0, 1),
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
            remap_window=(10, 10),
            remap_from_to=(0, 1),
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
            remap_window=(10, 10),
            remap_from_to=(0, 1),
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
        """The ``transition_matrix_inflated``/``inflate_window`` kwarg pair
        must (a) leave the predictive untouched before the window and
        (b) actually change it inside. A regression that ignored the kwarg
        would still produce well-shaped output, so we compare against a
        baseline run instead.
        """
        matrix_kwarg = "transition_matrix_inflated"
        window_kwarg = "inflate_window"
        n_bins = decoder_inputs.xs.size
        # Choose an alternative matrix that is *very* different from the
        # baseline (peak=0.9 diag-dominant). A near-uniform matrix forces
        # the predictive to spread out dramatically — easy to detect.
        alt_matrix = _diag_dominant_transition(n_bins, peak=0.05)
        window = (3, 6)

        baseline = decoder_inputs.call()
        with_alt = decoder_inputs.call(**{matrix_kwarg: alt_matrix, window_kwarg: window})

        # Before the window, the two runs must be bit-identical: nothing
        # in the algorithm has diverged yet.
        np.testing.assert_array_equal(
            baseline["predictive"][: window[0]], with_alt["predictive"][: window[0]]
        )
        np.testing.assert_array_equal(
            baseline["posterior"][: window[0]], with_alt["posterior"][: window[0]]
        )

        # Inside the window, at least one timestep's predictive must
        # measurably differ (this is the line the kwarg actually
        # controls). Use a generous tolerance so we don't depend on the
        # exact magnitude — we only assert "different".
        inside = slice(*window)
        assert not np.allclose(
            baseline["predictive"][inside],
            with_alt["predictive"][inside],
            atol=1e-6,
        ), f"{matrix_kwarg} in {window} did not change predictive — kwarg ignored?"

    def test_wiggly_rates_used_only_inside_window(self, decoder_inputs: DecoderInputs) -> None:
        """``wiggly_rates``/``wiggly_window`` must leave per-event diagnostics
        untouched for events outside the window and change at least one
        event inside it. Compared against a baseline run so a regression
        that ignored the kwarg cannot pass on output shape alone.
        """
        wiggly = wiggly_flat_rates(decoder_inputs.xs, n_cells=3)
        window = (3, 7)

        baseline = decoder_inputs.call()
        with_wiggly = decoder_inputs.call(wiggly_rates=wiggly, wiggly_window=window)

        evt_t = with_wiggly["event_time_ind"]
        # Events strictly before the window are unaffected — the filter
        # state has not diverged yet. (Events *after* the window
        # legitimately differ: the in-window posterior updates carry
        # forward, so we do not check those.)
        before = evt_t < window[0]
        inside = (evt_t >= window[0]) & (evt_t < window[1])

        np.testing.assert_array_equal(
            baseline["event_kl_divergence"][before],
            with_wiggly["event_kl_divergence"][before],
        )
        # At least one in-window event's KL changed (the wiggly likelihood
        # is genuinely different from the Gaussian-PF likelihood).
        assert inside.any(), "test fixture produced no in-window spike events"
        assert not np.allclose(
            baseline["event_kl_divergence"][inside],
            with_wiggly["event_kl_divergence"][inside],
            atol=1e-6,
        ), "wiggly_rates did not change in-window diagnostics — kwarg ignored?"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"transition_matrix_inflated": np.eye(21)},
                "must be provided together",
            ),
            ({"inflate_window": (2, 5)}, "must be provided together"),
            ({"wiggly_window": (2, 5)}, "must be provided together"),
        ],
    )
    def test_unpaired_window_kwarg_raises(
        self, decoder_inputs: DecoderInputs, kwargs: dict, match: str
    ) -> None:
        """Passing only one half of a matrix/window pair must fail loud —
        a silent fallback would produce a plausible-but-wrong figure.
        """
        with pytest.raises(ValueError, match=match):
            decoder_inputs.call(**kwargs)

    def test_wiggly_rates_with_negative_entries_raises(self, decoder_inputs: DecoderInputs) -> None:
        """A hand-built wiggly_rates table with negative entries is rejected
        before it can become NaN likelihoods downstream."""
        bad = wiggly_flat_rates(decoder_inputs.xs, n_cells=3)
        bad[0, 0] = -1.0
        with pytest.raises(ValueError, match="finite and non-negative"):
            decoder_inputs.call(wiggly_rates=bad, wiggly_window=(3, 7))


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

    def test_mixed_mask_reassembles_in_original_event_order(self) -> None:
        """Events split by an in/out mask are scattered back in order."""
        spike_time_ind = np.array([1, 2, 3, 4], dtype=np.intp)
        spike_cell_ind = np.array([0, 1, 0, 1], dtype=np.intp)
        in_window = np.array([False, True, True, False])
        # diag_out covers events 0 and 3; diag_in covers events 1 and 2.
        diag_out = self._batch([10.0, 13.0])
        diag_in = self._batch([11.0, 12.0])

        merged = _merge_diagnostics(
            n_time=5,
            n_cells=2,
            spike_time_ind=spike_time_ind,
            spike_cell_ind=spike_cell_ind,
            in_window=in_window,
            diag_out=diag_out,
            diag_in=diag_in,
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

    @pytest.mark.parametrize("all_in", [True, False])
    def test_all_in_or_all_out_mask(self, all_in: bool) -> None:
        """A mask that puts every event on one side still merges correctly."""
        n = 3
        spike_time_ind = np.arange(1, n + 1, dtype=np.intp)
        spike_cell_ind = np.zeros(n, dtype=np.intp)
        in_window = np.full(n, all_in)
        populated = self._batch([1.0, 2.0, 3.0])
        empty = self._batch([])
        merged = _merge_diagnostics(
            n_time=n + 1,
            n_cells=1,
            spike_time_ind=spike_time_ind,
            spike_cell_ind=spike_cell_ind,
            in_window=in_window,
            diag_out=empty if all_in else populated,
            diag_in=populated if all_in else empty,
        )
        np.testing.assert_array_equal(merged["event_kl_divergence"], [101.0, 102.0, 103.0])

    def test_length_mismatch_raises(self) -> None:
        """A batch whose event count disagrees with the mask fails loud."""
        spike_time_ind = np.array([1, 2, 3], dtype=np.intp)
        spike_cell_ind = np.zeros(3, dtype=np.intp)
        in_window = np.array([False, True, True])
        with pytest.raises(ValueError, match="diag_in has 1 events but the mask expects 2"):
            _merge_diagnostics(
                n_time=4,
                n_cells=1,
                spike_time_ind=spike_time_ind,
                spike_cell_ind=spike_cell_ind,
                in_window=in_window,
                diag_out=self._batch([10.0]),
                diag_in=self._batch([11.0]),  # mask expects 2, not 1
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
