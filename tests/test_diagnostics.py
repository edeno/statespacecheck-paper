"""Tests for the shared diagnostics module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import statespacecheck as ssc

from statespacecheck_paper.diagnostics import (
    DecodingDiagnostics,
    DiagnosticThresholds,
    SpikeEventDiagnostics,
    compute_baseline_diagnostic_thresholds,
    compute_spike_event_diagnostics_from_rates,
)

from ._decoder_inputs import DecoderInputs


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

    def test_accepts_diagnostics_object(self, decoder_inputs: DecoderInputs) -> None:
        """``compute_baseline_diagnostic_thresholds`` accepts either a
        ``DecodingDiagnostics`` or a plain dict (union back-compat for
        synthetic test fixtures). Pin the DecodingDiagnostics branch so it
        stays exercised."""
        diagnostics = decoder_inputs.call()
        thresholds = compute_baseline_diagnostic_thresholds(diagnostics, baseline_end_index=5)
        # Same call shape with a dict — results must agree.
        as_dict = {
            "hpd_overlap": diagnostics.hpd_overlap,
            "kl_divergence": diagnostics.kl_divergence,
            "predictive_pvalue": diagnostics.predictive_pvalue,
        }
        from_dict = compute_baseline_diagnostic_thresholds(as_dict, baseline_end_index=5)
        assert thresholds.hpd_overlap == pytest.approx(from_dict.hpd_overlap)
        assert thresholds.kl_divergence == pytest.approx(from_dict.kl_divergence)
        assert thresholds.predictive_pvalue == from_dict.predictive_pvalue


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
            likelihood=posterior.copy(),
            spike_likelihood=posterior.copy(),
            hpd_overlap=np.zeros((n_time, n_cells)),
            kl_divergence=np.zeros((n_time, n_cells)),
            predictive_pvalue=np.zeros((n_time, n_cells)),
            event_time_ind=np.zeros(n_spikes, dtype=np.intp),
            event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
            event_hpd_overlap=np.zeros(n_spikes),
            event_kl_divergence=np.zeros(n_spikes),
            event_predictive_pvalue=np.zeros(n_spikes),
            per_spike_likelihood=np.zeros((n_spikes, n_bins)),
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
                per_spike_likelihood=np.zeros((n_spikes, n_bins)),
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
        np.testing.assert_array_equal(result.per_spike_likelihood, expected.likelihood)

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
        assert result.per_spike_likelihood is None
        assert result.event_hpd_overlap.shape == (4,)
