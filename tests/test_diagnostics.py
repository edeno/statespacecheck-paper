"""Tests for the shared diagnostics module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statespacecheck_paper.diagnostics import (
    DecodingDiagnostics,
    DiagnosticThresholds,
    SpikeEventDiagnostics,
    _compute_spike_event_predictive_pvalue_rank,
    compute_baseline_diagnostic_thresholds,
    compute_normalized_event_likelihood,
    compute_predictive_mark_probabilities,
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
# predictive mark probabilities
# ---------------------------------------------------------------------------


class TestComputePredictiveMarkProbabilities:
    def test_integrates_raw_intensities_before_normalizing(self) -> None:
        """Population intensity varies by state, so averaging conditional
        cell fractions would give [0.7, 0.3]. The event-weighted
        predictive distribution must instead be [5/6, 1/6].
        """
        prior = np.array([0.5, 0.5])
        rates = np.array([[9.0, 1.0], [1.0, 1.0]])

        mark_probs = compute_predictive_mark_probabilities(prior, rates)

        assert_allclose(mark_probs, [5.0 / 6.0, 1.0 / 6.0])

    def test_global_intensity_scale_does_not_change_distribution(self) -> None:
        prior = np.array([0.5, 0.3, 0.2])
        rates = np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]])
        baseline = compute_predictive_mark_probabilities(prior, rates)
        assert_allclose(compute_predictive_mark_probabilities(prior, 17.0 * rates), baseline)

    def test_zero_total_intensity_raises(self) -> None:
        prior = np.array([0.5, 0.5])
        rates = np.zeros((2, 3))
        with pytest.raises(ValueError, match="total event intensity is zero"):
            compute_predictive_mark_probabilities(prior, rates)

    def test_zero_total_intensity_row_raises_in_time_series(self) -> None:
        """A per-time predictive with a zero-intensity row is undefined for that row."""
        # Row 1 places all mass on bin 1, whose mark intensities are all zero.
        predictive = np.array([[1.0, 0.0], [0.0, 1.0]])  # (n_time, n_bins)
        rates = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])  # (n_bins, n_marks)
        with pytest.raises(ValueError, match="zero total"):
            compute_predictive_mark_probabilities(predictive, rates)

    def test_nonfinite_total_after_reduction_raises(self) -> None:
        """Finite expected intensities must not normalize by an overflowed sum."""
        prior = np.array([0.5, 0.5])
        rates = np.full((2, 2), 1e308)
        with pytest.raises(ValueError, match="total event intensity is non-finite"):
            compute_predictive_mark_probabilities(prior, rates)

    def test_nonfinite_expected_intensity_after_matrix_product_raises(self) -> None:
        """Overflow during state integration must fail before normalization."""
        predictive = np.array([[1.0, 1.0]])
        rates = np.full((2, 1), 1e308)
        with pytest.raises(ValueError, match="expected mark intensities are non-finite"):
            compute_predictive_mark_probabilities(predictive, rates)


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
    """Direct tests for the per-spike-event diagnostics helper."""

    def test_integrates_raw_rates_before_normalizing(self) -> None:
        """Regression test for the original MATLAB normalization-order bug.

        Averaging state-conditional cell fractions would assign the less
        likely cell rank 0.3. Conditioning the latent state on an event by
        integrating raw rates first gives the correct rank 1/6.
        """
        predictive = np.array([[0.5, 0.5], [0.5, 0.5]])
        rates = np.array([[9.0, 1.0], [1.0, 1.0]])
        spike_time_ind = np.array([0, 1], dtype=np.intp)
        spike_cell_ind = np.array([0, 1], dtype=np.intp)

        result = compute_spike_event_diagnostics_from_rates(
            predictive, rates, spike_time_ind, spike_cell_ind, coverage=0.95
        )
        np.testing.assert_allclose(result.event_predictive_pvalue, [1.0, 1.0 / 6.0])

    def test_zero_rate_row_contributes_no_event_mass(self) -> None:
        """A state with zero population rate contributes no mass after
        conditioning on an event; equal rates elsewhere keep cells tied.
        """
        predictive = np.array([[0.2, 0.5, 0.3]])
        rates = np.array([[0.5, 0.5], [0.0, 0.0], [0.5, 0.5]])
        result = compute_spike_event_diagnostics_from_rates(
            predictive,
            rates,
            np.array([0], dtype=np.intp),
            np.array([0], dtype=np.intp),
            coverage=0.95,
        )
        np.testing.assert_allclose(result.event_predictive_pvalue, 1.0, atol=1e-12)

    def test_fully_degenerate_rates_raise(self) -> None:
        """An observed spike is impossible under an all-zero rate table."""
        n_time, n_bins, n_cells = 5, 3, 2
        predictive = np.full((n_time, n_bins), 1.0 / n_bins)
        rates = np.zeros((n_bins, n_cells))
        spike_time_ind = np.array([0], dtype=np.intp)
        spike_cell_ind = np.array([0], dtype=np.intp)
        with pytest.raises(ValueError, match="zero at every position"):
            compute_spike_event_diagnostics_from_rates(
                predictive, rates, spike_time_ind, spike_cell_ind, coverage=0.95
            )


class TestComputeNormalizedEventLikelihood:
    def test_matches_normalized_intensity_and_rows_sum_to_one(self) -> None:
        rates = np.array([[2.0, 0.5, 1.0], [0.1, 0.4, 0.2]])
        out = compute_normalized_event_likelihood(rates)

        expected = rates / rates.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(out, expected)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0)

    def test_global_intensity_scale_does_not_change_event_likelihood(self) -> None:
        rates = np.array([[2.0, 0.5, 1.0], [0.1, 0.4, 0.2]])
        expected = compute_normalized_event_likelihood(rates)

        np.testing.assert_allclose(compute_normalized_event_likelihood(17.0 * rates), expected)

    def test_degenerate_zero_rate_row_raises(self) -> None:
        rates = np.array([[2.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="zero at every position"):
            compute_normalized_event_likelihood(rates)

    def test_tiny_but_informative_rates_keep_their_shape(self) -> None:
        # Rates far below any absolute threshold still have a well-defined
        # shape: they must normalize to their ratio, not collapse to uniform.
        rates = np.array([[1e-20, 2e-20, 4e-20]])
        out = compute_normalized_event_likelihood(rates)

        expected = np.array([1.0, 2.0, 4.0]) / 7.0
        np.testing.assert_allclose(out[0], expected, rtol=1e-6)
        assert not np.allclose(out[0], np.full(3, 1.0 / 3.0))


class TestSpikeEventPredictivePvalueRankTolerance:
    def test_sub_atol_tie_does_not_flip_rank(self) -> None:
        """Two cells whose predictive contributions differ by less than the
        ``rank_atol`` slack must receive the *same* rank. A one-hot predictive
        row lets the contributions be set directly via the rate table; the pair
        at 0.30 and 0.30 + delta (delta < rank_atol) is bracketed by a clearly
        larger and a clearly smaller cell, so without the tolerance the two
        events would land on different ranks (0.35 vs 0.65+delta) instead of
        tying.
        """
        n_bins, n_cells = 4, 4
        delta = 1e-15  # below rank_atol ~ eps*n_bins*16*max_contrib ~ 5e-15
        contributions = np.array([0.05, 0.30, 0.30 + delta, 0.35 - delta])
        # rank_atol must exceed the near-tie gap for the tie to hold.
        rank_atol = float(np.finfo(float).eps * n_bins * 16) * float(contributions.max())
        assert delta < rank_atol

        rates = np.zeros((n_bins, n_cells))
        rates[0] = contributions  # only bin 0 carries intensity
        pred = np.zeros((2, n_bins))
        pred[:, 0] = 1.0  # both events sit on bin 0 -> identical contributions
        cell_ind = np.array([1, 2], dtype=np.intp)  # near-tied pair (0.30, 0.30+delta)

        ranks = _compute_spike_event_predictive_pvalue_rank(pred, rates, cell_ind)

        assert ranks[0] == ranks[1]  # the sub-atol difference does not flip rank
        assert 0.0 < ranks[0] < 1.0  # discriminating: neither everything nor nothing

    def test_values_in_unit_range(self) -> None:
        rng = np.random.default_rng(1)
        n_time, n_bins, n_cells = 30, 12, 5
        pred = rng.dirichlet(np.ones(n_bins), size=n_time)
        rates = rng.random((n_bins, n_cells))
        cell_ind = rng.integers(0, n_cells, size=n_time).astype(np.intp)

        ranks = _compute_spike_event_predictive_pvalue_rank(pred, rates, cell_ind)

        assert ranks.shape == (n_time,)
        # Rank is a cumulative probability mass: bounded in [0, 1], allowing the
        # tiny FP overshoot above 1 the reduction can produce for the top cell
        # (matches the tolerance in test_figure04_diagnostics).
        assert np.all(ranks >= 0.0)
        assert np.all(ranks <= 1.0 + 1e-9)
