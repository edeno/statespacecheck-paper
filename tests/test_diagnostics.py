"""Tests for the shared diagnostics module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import statespacecheck as ssc

from statespacecheck_paper.diagnostics import (
    INCLUSIVE_FLAG_COMPARISONS,
    METRIC_FLAG_DIRECTIONS,
    DecodingDiagnostics,
    DiagnosticThresholds,
    FlagDirection,
    SpikeEventDiagnostics,
    compute_baseline_diagnostic_thresholds,
    compute_spike_event_diagnostics_from_rates,
    expand_spike_events,
    flag_mask,
)


@pytest.fixture
def metrics_2d() -> dict[str, np.ndarray]:
    """Standard (n_time, n_cells) metrics dict for threshold tests."""
    rng = np.random.default_rng(42)
    return {
        "hpd_overlap": rng.uniform(0.5, 1.0, (100, 5)),
        "kl_divergence": rng.uniform(0.0, 2.0, (100, 5)),
        "predictive_pvalue": rng.uniform(0.0, 1.0, (100, 5)),
    }


# ---------------------------------------------------------------------------
# DiagnosticThresholds / compute_baseline_diagnostic_thresholds
# ---------------------------------------------------------------------------


class TestComputeBaselineDiagnosticThresholds:
    def test_thresholds_match_quantile_definitions(self, metrics_2d: dict[str, np.ndarray]) -> None:
        baseline_end = 50
        thresholds = compute_baseline_diagnostic_thresholds(
            metrics_2d, baseline_end_index=baseline_end
        )

        expected_hpdo = np.nanquantile(metrics_2d["hpd_overlap"][:baseline_end].ravel(), 0.01)
        expected_kl = np.nanquantile(metrics_2d["kl_divergence"][:baseline_end].ravel(), 0.99)
        assert thresholds.hpd_overlap == pytest.approx(expected_hpdo)
        assert thresholds.kl_divergence == pytest.approx(expected_kl)
        # predictive_pvalue is a fixed rank-statistic cutoff, not data-driven.
        assert thresholds.predictive_pvalue == 0.05

    def test_handles_partial_nan_baseline(self) -> None:
        """NaNs in the baseline must be ignored, not propagate to thresholds."""
        n_time, n_cells = 20, 3
        hpdo = np.full((n_time, n_cells), 0.8)
        hpdo[:5] = np.nan
        metrics: dict[str, np.ndarray] = {
            "hpd_overlap": hpdo,
            "kl_divergence": np.full((n_time, n_cells), 1.0),
            "predictive_pvalue": np.full((n_time, n_cells), 0.5),
        }
        thresholds = compute_baseline_diagnostic_thresholds(metrics, baseline_end_index=10)
        assert not np.isnan(thresholds.hpd_overlap)
        assert not np.isnan(thresholds.kl_divergence)

    def test_baseline_end_index_is_keyword_only(self, metrics_2d: dict[str, np.ndarray]) -> None:
        """Passing baseline_end_index positionally must fail — the argument is
        keyword-only so callers can't accidentally omit it via the prior
        ``None`` default that silently used the whole recording."""
        # Cast to Any to probe the runtime contract without the static
        # type checker rejecting the deliberately-wrong call.
        unchecked: Any = compute_baseline_diagnostic_thresholds
        with pytest.raises(TypeError, match="positional"):
            unchecked(metrics_2d, 50)

    @pytest.mark.parametrize("case", ["past_end", "zero", "negative"])
    def test_baseline_end_index_out_of_range_raises(
        self, metrics_2d: dict[str, np.ndarray], case: str
    ) -> None:
        """An index past the end would silently use the whole recording and a
        negative one would drop its last rows; both, and zero, raise."""
        n_time = metrics_2d["hpd_overlap"].shape[0]
        index = {"past_end": n_time + 1, "zero": 0, "negative": -1}[case]
        with pytest.raises(ValueError, match="baseline_end_index must be in"):
            compute_baseline_diagnostic_thresholds(metrics_2d, baseline_end_index=index)

    def test_baseline_end_index_may_cover_the_recording(
        self, metrics_2d: dict[str, np.ndarray]
    ) -> None:
        n_time = metrics_2d["hpd_overlap"].shape[0]
        thresholds = compute_baseline_diagnostic_thresholds(metrics_2d, baseline_end_index=n_time)
        assert thresholds.hpd_overlap == pytest.approx(
            np.nanquantile(metrics_2d["hpd_overlap"].ravel(), 0.01)
        )

    def test_all_nan_hpd_baseline_raises(self) -> None:
        """An all-NaN baseline slice would produce a NaN threshold and
        every downstream ``metric < threshold`` comparison would silently
        evaluate False. Raise instead."""
        n_time, n_cells = 20, 3
        metrics: dict[str, np.ndarray] = {
            "hpd_overlap": np.full((n_time, n_cells), np.nan),
            "kl_divergence": np.full((n_time, n_cells), 1.0),
            "predictive_pvalue": np.full((n_time, n_cells), 0.5),
        }
        with pytest.raises(ValueError, match="hpd_overlap baseline slice"):
            compute_baseline_diagnostic_thresholds(metrics, baseline_end_index=10)

    def test_all_nan_kl_baseline_raises(self) -> None:
        n_time, n_cells = 20, 3
        metrics: dict[str, np.ndarray] = {
            "hpd_overlap": np.full((n_time, n_cells), 0.8),
            "kl_divergence": np.full((n_time, n_cells), np.nan),
            "predictive_pvalue": np.full((n_time, n_cells), 0.5),
        }
        with pytest.raises(ValueError, match="kl_divergence baseline slice"):
            compute_baseline_diagnostic_thresholds(metrics, baseline_end_index=10)


class TestDiagnosticThresholdsInvariants:
    """Range validation at construction. Reverting any branch lets a
    NaN or out-of-range threshold slip through and silently make
    downstream ``metric < threshold`` comparisons evaluate False."""

    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
    def test_hpd_overlap_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"hpd_overlap must lie in \[0, 1\]"):
            DiagnosticThresholds(hpd_overlap=bad, kl_divergence=0.0, predictive_pvalue=0.05)

    @pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
    def test_kl_divergence_non_finite_or_negative_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"kl_divergence must be finite and non-negative"):
            DiagnosticThresholds(hpd_overlap=0.5, kl_divergence=bad, predictive_pvalue=0.05)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
    def test_predictive_pvalue_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"predictive_pvalue must lie in \[0, 1\]"):
            DiagnosticThresholds(hpd_overlap=0.5, kl_divergence=0.0, predictive_pvalue=bad)

    def test_boundary_values_accepted(self) -> None:
        """The closed-interval boundaries [0, 1] must construct cleanly."""
        DiagnosticThresholds(hpd_overlap=0.0, kl_divergence=0.0, predictive_pvalue=0.0)
        DiagnosticThresholds(hpd_overlap=1.0, kl_divergence=0.0, predictive_pvalue=1.0)

    def test_is_frozen(self) -> None:
        """Frozen so a downstream consumer cannot rebind a field mid-pipeline."""
        from dataclasses import FrozenInstanceError

        t: Any = DiagnosticThresholds(hpd_overlap=0.5, kl_divergence=0.0, predictive_pvalue=0.05)
        with pytest.raises(FrozenInstanceError):
            t.hpd_overlap = 0.7


class TestDecodingDiagnosticsInvariants:
    """``DecodingDiagnostics.__post_init__`` validates shape and value ranges
    on every field. Exercise the most-likely-to-regress branches
    directly so a future "loosen the check" change fails here, not
    later as a NaN downstream."""

    def _kwargs(
        self, *, n_time: int = 4, n_bins: int = 3, n_cells: int = 2, n_spikes: int = 1
    ) -> dict[str, np.ndarray]:
        posterior = np.full((n_time, n_bins), 1.0 / n_bins)
        return dict(
            posterior=posterior,
            predictive=posterior.copy(),
            combined_likelihood=posterior.copy(),
            hpd_overlap=np.zeros((n_time, n_cells)),
            kl_divergence=np.zeros((n_time, n_cells)),
            predictive_pvalue=np.zeros((n_time, n_cells)),
            event_time_ind=np.zeros(n_spikes, dtype=np.intp),
            event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
            event_hpd_overlap=np.zeros(n_spikes),
            event_kl_divergence=np.zeros(n_spikes),
            event_predictive_pvalue=np.zeros(n_spikes),
            event_likelihood=np.zeros((n_spikes, n_bins)),
        )

    def test_predictive_shape_mismatch_raises(self) -> None:
        kwargs = self._kwargs(n_time=4, n_bins=3)
        kwargs["predictive"] = np.zeros((5, 3))  # wrong leading dim
        with pytest.raises(ValueError, match=r"DecodingDiagnostics\.predictive shape"):
            DecodingDiagnostics(**kwargs)

    def test_per_event_shape_mismatch_raises(self) -> None:
        kwargs = self._kwargs(n_spikes=3)
        kwargs["event_kl_divergence"] = np.zeros(4)  # wrong leading dim
        with pytest.raises(ValueError, match=r"DecodingDiagnostics\.event_kl_divergence shape"):
            DecodingDiagnostics(**kwargs)

    def test_posterior_must_be_2d(self) -> None:
        kwargs = self._kwargs()
        kwargs["posterior"] = np.zeros(12)  # 1-D
        with pytest.raises(ValueError, match=r"DecodingDiagnostics\.posterior must be 2-D"):
            DecodingDiagnostics(**kwargs)

    def test_hpd_overlap_out_of_range_raises(self) -> None:
        """A buggy decoder shipping ``hpd_overlap > 1`` is caught at the
        producer boundary, not silently propagated into HPD overlap
        statistics that look fine at first glance."""
        kwargs = self._kwargs()
        kwargs["hpd_overlap"] = np.full((4, 2), 1.5)
        with pytest.raises(ValueError, match=r"DecodingDiagnostics\.hpd_overlap: values above 1"):
            DecodingDiagnostics(**kwargs)

    def test_kl_divergence_negative_raises(self) -> None:
        kwargs = self._kwargs()
        kwargs["kl_divergence"] = np.full((4, 2), -0.5)
        with pytest.raises(ValueError, match=r"DecodingDiagnostics\.kl_divergence: values below 0"):
            DecodingDiagnostics(**kwargs)

    def test_nan_in_dense_field_is_allowed(self) -> None:
        """NaN at (t, cell) without a spike is legitimate; the range
        check must let it through."""
        kwargs = self._kwargs()
        kwargs["hpd_overlap"][0, :] = np.nan
        kwargs["kl_divergence"][0, :] = np.nan
        kwargs["predictive_pvalue"][0, :] = np.nan
        DecodingDiagnostics(**kwargs)  # does not raise

    @pytest.mark.parametrize(
        ("field", "bad_value", "message"),
        [
            ("event_hpd_overlap", np.nan, "NaN found in a required per-event value"),
            ("event_predictive_pvalue", 1.5, "values above 1.0"),
            ("event_kl_divergence", -np.inf, "-inf is not a valid diagnostic value"),
            ("predictive_pvalue", -0.5, "values below 0.0"),
        ],
    )
    def test_out_of_range_metric_names_the_field(
        self, field: str, bad_value: float, message: str
    ) -> None:
        # The per-event arrays feed the manuscript means, which no longer
        # re-check them, so these range checks are their only guard.
        kwargs = self._kwargs()
        kwargs[field].flat[0] = bad_value
        with pytest.raises(ValueError, match=rf"DecodingDiagnostics\.{field}: {message}"):
            DecodingDiagnostics(**kwargs)

    def test_infinite_event_kl_divergence_is_allowed(self) -> None:
        kwargs = self._kwargs()
        kwargs["event_kl_divergence"][0] = np.inf
        DecodingDiagnostics(**kwargs)  # does not raise


class TestSpikeEventDiagnosticsInvariants:
    """All-or-nothing on the dense matrices is the load-bearing
    invariant of ``SpikeEventDiagnostics``; cover it directly so a
    future caller can't supply ``hpd_overlap`` without
    ``kl_divergence`` and have downstream code mistake the
    None as "include_dense_matrices=False"."""

    def test_partial_dense_matrices_rejected(self) -> None:
        n_spikes, n_time, n_cells, n_bins = 2, 4, 2, 3
        with pytest.raises(ValueError, match="all-or-nothing"):
            SpikeEventDiagnostics(
                event_time_ind=np.zeros(n_spikes, dtype=np.intp),
                event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
                event_hpd_overlap=np.zeros(n_spikes),
                event_kl_divergence=np.zeros(n_spikes),
                event_predictive_pvalue=np.zeros(n_spikes),
                hpd_overlap=np.zeros((n_time, n_cells)),
                kl_divergence=None,  # only some dense matrices supplied
                predictive_pvalue=np.zeros((n_time, n_cells)),
                event_likelihood=np.zeros((n_spikes, n_bins)),
            )

    @pytest.mark.parametrize(
        ("field", "bad_value", "message"),
        [
            ("event_hpd_overlap", 1.5, "values above 1.0"),
            ("event_predictive_pvalue", -0.5, "values below 0.0"),
            ("event_kl_divergence", np.nan, "NaN found in a required per-event value"),
            ("hpd_overlap", -np.inf, "-inf is not a valid diagnostic value"),
            ("predictive_pvalue", np.inf, r"\+inf is not permitted"),
            ("kl_divergence", -1.0, "values below 0.0"),
        ],
    )
    def test_out_of_range_metric_names_the_field(
        self, field: str, bad_value: float, message: str
    ) -> None:
        n_spikes, n_time, n_cells, n_bins = 2, 4, 2, 3
        arrays: dict[str, Any] = {
            "event_hpd_overlap": np.full(n_spikes, 0.5),
            "event_kl_divergence": np.full(n_spikes, np.inf),
            "event_predictive_pvalue": np.full(n_spikes, 0.5),
            "hpd_overlap": np.full((n_time, n_cells), np.nan),
            "kl_divergence": np.full((n_time, n_cells), np.nan),
            "predictive_pvalue": np.full((n_time, n_cells), np.nan),
        }
        arrays[field].flat[0] = bad_value
        with pytest.raises(ValueError, match=rf"SpikeEventDiagnostics\.{field}: {message}"):
            SpikeEventDiagnostics(
                event_time_ind=np.zeros(n_spikes, dtype=np.intp),
                event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
                event_likelihood=np.zeros((n_spikes, n_bins)),
                **arrays,
            )


# ---------------------------------------------------------------------------
# compute_spike_event_diagnostics_from_rates
# ---------------------------------------------------------------------------


class TestComputeSpikeEventDiagnosticsFromRates:
    """The adapter around ``statespacecheck.event_diagnostics``; the diagnostics
    themselves are tested in the statespacecheck package."""

    @pytest.fixture
    def inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        predictive = rng.dirichlet(np.ones(6), size=5)  # (n_time, n_bins)
        rates = rng.random((6, 3)) + 0.1  # (n_bins, n_cells)
        # Two spikes share (time 1, cell 2); time 3 has no spikes.
        spike_time_ind = np.array([0, 1, 1, 4], dtype=np.intp)
        spike_cell_ind = np.array([0, 2, 2, 1], dtype=np.intp)
        return predictive, rates, spike_time_ind, spike_cell_ind

    def test_matches_package_and_scatters_into_dense_matrices(
        self, inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        predictive, rates, time_ind, cell_ind = inputs
        result = compute_spike_event_diagnostics_from_rates(predictive, rates, time_ind, cell_ind)
        expected = ssc.event_diagnostics(
            predictive, rates, time_ind, cell_ind, return_likelihood=True
        )
        spiked = np.zeros((5, 3), dtype=bool)
        spiked[time_ind, cell_ind] = True
        for name in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            values = getattr(expected, name)
            np.testing.assert_array_equal(getattr(result, f"event_{name}"), values)
            dense = getattr(result, name)
            assert dense.shape == (5, 3)
            np.testing.assert_array_equal(dense[time_ind, cell_ind], values)
            assert np.all(np.isnan(dense[~spiked]))
        np.testing.assert_array_equal(result.event_likelihood, expected.likelihood)

    def test_dense_matrices_omitted_on_request(
        self, inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        predictive, rates, time_ind, cell_ind = inputs
        result = compute_spike_event_diagnostics_from_rates(
            predictive, rates, time_ind, cell_ind, include_dense_matrices=False
        )
        assert result.hpd_overlap is None
        assert result.kl_divergence is None
        assert result.predictive_pvalue is None
        assert result.event_likelihood is None
        assert result.event_hpd_overlap.shape == (4,)


# ---------------------------------------------------------------------------
# expand_spike_events
# ---------------------------------------------------------------------------


def test_expand_spike_events_repeats_multi_spike_bins_in_row_major_order() -> None:
    spike_counts = np.array([[1, 0, 2], [0, 0, 0], [3, 1, 0]])
    time_ind, cell_ind = expand_spike_events(spike_counts)
    assert time_ind.dtype == np.intp and cell_ind.dtype == np.intp
    np.testing.assert_array_equal(time_ind, [0, 0, 0, 2, 2, 2, 2])
    np.testing.assert_array_equal(cell_ind, [0, 2, 2, 0, 0, 0, 1])
    np.testing.assert_array_equal(
        np.bincount(time_ind * 3 + cell_ind, minlength=9), spike_counts.ravel()
    )


def test_expand_spike_events_without_spikes_is_empty() -> None:
    time_ind, cell_ind = expand_spike_events(np.zeros((4, 2), dtype=np.int64))
    assert time_ind.shape == (0,) and cell_ind.shape == (0,)


# ---------------------------------------------------------------------------
# flag_mask / METRIC_FLAG_DIRECTIONS
# ---------------------------------------------------------------------------


class TestFlagMask:
    @pytest.mark.parametrize(
        ("direction", "expected"),
        [
            ("below", [True, True, False, False]),
            ("above", [False, True, True, False]),
        ],
    )
    def test_inclusive_at_threshold_and_never_flags_nan(
        self, direction: FlagDirection, expected: list[bool]
    ) -> None:
        values = np.array([0.0, 0.5, 1.0, np.nan])
        np.testing.assert_array_equal(flag_mask(values, 0.5, direction), expected)

    def test_positive_infinity_is_flagged_above(self) -> None:
        values = np.array([np.inf, 1.0])
        np.testing.assert_array_equal(flag_mask(values, 2.0, "above"), [True, False])

    def test_bad_direction_raises(self) -> None:
        unchecked: Any = flag_mask
        with pytest.raises(ValueError, match="direction"):
            unchecked(np.array([1.0]), 0.5, "sideways")

    def test_every_metric_direction_has_a_recorded_comparison(self) -> None:
        assert METRIC_FLAG_DIRECTIONS == {
            "hpd_overlap": "below",
            "predictive_pvalue": "below",
            "kl_divergence": "above",
        }
        assert set(INCLUSIVE_FLAG_COMPARISONS) == set(METRIC_FLAG_DIRECTIONS.values())
