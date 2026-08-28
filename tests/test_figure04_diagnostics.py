"""Tests for Figure-4 real-data goodness-of-fit diagnostics."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import statespacecheck_paper.figure04_diagnostics as figure04_diagnostics
from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_diagnostics import (
    compute_flag_confusion,
    compute_model_diagnostics,
    compute_running_average,
    compute_spike_event_diagnostics,
    gaussian_smooth,
)


def _xarray_results(
    posterior_data: np.ndarray,
    name: str,
    state_bins: pd.MultiIndex | np.ndarray | None = None,
) -> xr.Dataset:
    """Build a 2-variable Dataset matching the on-disk results layout."""
    n_time, n_state_bins = posterior_data.shape
    if state_bins is None:
        state_bins = np.arange(n_state_bins)
    return xr.Dataset(
        {
            name: xr.DataArray(
                posterior_data,
                dims=["time", "state_bins"],
                coords={"time": np.arange(n_time), "state_bins": state_bins},
            )
        }
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def per_cell_setup(rng: np.random.Generator) -> dict[str, Any]:
    """Standard inputs for ``compute_spike_event_diagnostics``."""
    n_time, n_bins, n_cells = 100, 50, 10
    return {
        "n_time": n_time,
        "n_bins": n_bins,
        "n_cells": n_cells,
        "predictive": rng.dirichlet(np.ones(n_bins), size=n_time),
        "place_fields": rng.random((n_cells, n_bins)) * 10 + 0.1,
        "spike_counts": rng.poisson(0.5, (n_time, n_cells)).astype(np.int64),
    }


# ---------------------------------------------------------------------------
# gaussian_smooth
# ---------------------------------------------------------------------------


class TestGaussianSmooth:
    def test_output_shape_matches_input(self, rng: np.random.Generator) -> None:
        data = rng.standard_normal(1000)
        result = gaussian_smooth(data, sigma=0.01, sampling_frequency=1000)
        assert result.shape == data.shape

    def test_smoothing_reduces_variance_of_noise(self, rng: np.random.Generator) -> None:
        data = rng.standard_normal(1000)
        result = gaussian_smooth(data, sigma=0.02, sampling_frequency=500)
        assert result.var() < data.var()


# ---------------------------------------------------------------------------
# compute_spike_event_diagnostics
# ---------------------------------------------------------------------------


class TestComputePerCellDiagnostics:
    def test_shapes_and_keys(self, per_cell_setup: dict) -> None:
        result = compute_spike_event_diagnostics(
            per_cell_setup["predictive"],
            per_cell_setup["spike_counts"],
            per_cell_setup["place_fields"],
        )
        for key in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            arr = getattr(result, key)
            assert arr is not None
            assert arr.shape == (per_cell_setup["n_time"], per_cell_setup["n_cells"])

    def test_nan_exactly_where_no_spikes(self, per_cell_setup: dict) -> None:
        result = compute_spike_event_diagnostics(
            per_cell_setup["predictive"],
            per_cell_setup["spike_counts"],
            per_cell_setup["place_fields"],
        )
        no_spike = per_cell_setup["spike_counts"] == 0
        for key in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            arr = getattr(result, key)
            assert arr is not None
            assert np.all(np.isnan(arr[no_spike]))

    @pytest.mark.parametrize("metric", ["hpd_overlap", "predictive_pvalue"])
    def test_metric_in_unit_range_with_gaussian_place_fields(
        self, rng: np.random.Generator, metric: str
    ) -> None:
        """HPD overlap and predictive_pvalue are bounded in [0, 1] (allowing tiny
        floating-point overshoot above 1 for predictive_pvalue)."""
        n_time, n_bins, n_cells = 100, 50, 10
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        # Gaussian place fields ensure the event spike-prob rank stays well-defined.
        place_fields = np.zeros((n_cells, n_bins))
        for j, center in enumerate(np.linspace(5, n_bins - 5, n_cells)):
            place_fields[j] = np.exp(-0.5 * ((np.arange(n_bins) - center) / 5) ** 2)
        place_fields = place_fields * 10 + 0.1
        spike_counts = np.ones((n_time, n_cells), dtype=np.int64)

        result = compute_spike_event_diagnostics(predictive, spike_counts, place_fields)
        arr = getattr(result, metric)
        assert arr is not None
        valid = arr[~np.isnan(arr)]
        assert (valid >= 0.0).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_diagnostics_only_at_spike_times(self, rng: np.random.Generator) -> None:
        n_time, n_bins, n_cells = 20, 10, 3
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        place_fields = rng.random((n_cells, n_bins)) * 10 + 0.1
        spike_counts = np.zeros((n_time, n_cells), dtype=np.int64)
        spike_counts[[0, 5, 10], 0] = 1
        spike_counts[[2, 7], 1] = 1
        spike_counts[15, 2] = 1

        result = compute_spike_event_diagnostics(predictive, spike_counts, place_fields)
        # Diagnostics finite at spike times, NaN elsewhere — single check.
        assert result.hpd_overlap is not None
        np.testing.assert_array_equal(np.isnan(result.hpd_overlap), spike_counts == 0)

    def test_duplicate_spikes_in_same_bin_are_separate_events(
        self, rng: np.random.Generator
    ) -> None:
        """Two spikes in one (time, cell) bin must yield two event entries
        with the same value as the bin's matrix entry."""
        n_time, n_bins, n_cells = 4, 8, 1
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        place_fields = rng.random((n_cells, n_bins)) + 0.1
        spike_counts = np.zeros((n_time, n_cells), dtype=np.int64)
        spike_counts[1, 0] = 2

        result = compute_spike_event_diagnostics(
            predictive,
            spike_counts,
            place_fields,
            spike_times=[np.array([1.10, 1.20])],
            time=np.arange(n_time, dtype=np.float64),
        )

        assert result.event_time is not None
        assert result.kl_divergence is not None
        np.testing.assert_allclose(result.event_time, [1.10, 1.20])
        np.testing.assert_array_equal(result.event_time_ind, [1, 1])
        np.testing.assert_array_equal(result.event_cell_ind, [0, 0])
        for event_key in ("event_hpd_overlap", "event_kl_divergence", "event_predictive_pvalue"):
            assert getattr(result, event_key).shape == (2,)
        np.testing.assert_allclose(
            result.event_kl_divergence,
            np.repeat(result.kl_divergence[1, 0], 2),
        )


class TestMeanPerSpikeLikelihoodByTime:
    def test_averages_normalized_fields_weighted_by_spike_count(self) -> None:
        from statespacecheck_paper.figure04_diagnostics import (
            mean_per_spike_likelihood_by_time,
        )

        # cell 0 concentrates at bin 0, cell 1 at bin 2 (unnormalized).
        place_fields = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        spike_counts = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.int64)

        mean_lik, has_spikes = mean_per_spike_likelihood_by_time(spike_counts, place_fields)

        expected = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(mean_lik, expected)
        np.testing.assert_array_equal(has_spikes, np.array([True, True, False]))

    def test_uses_normalized_poisson_likelihood_not_raw_rate(self) -> None:
        from scipy.stats import poisson

        from statespacecheck_paper.figure04_diagnostics import (
            mean_per_spike_likelihood_by_time,
        )

        # One cell, two bins, with rates large enough that the Poisson
        # exp(-rate) factor matters. Raw-rate normalization would give
        # [0.8, 0.2]; the diagnostics normalize Poisson(k=1, mu=rate).
        place_fields = np.array([[2.0, 0.5]])
        spike_counts = np.array([[1]], dtype=np.int64)

        mean_lik, _ = mean_per_spike_likelihood_by_time(spike_counts, place_fields)

        pmf = poisson.pmf(k=1, mu=place_fields[0])
        expected = pmf / pmf.sum()
        np.testing.assert_allclose(mean_lik[0], expected)
        assert not np.allclose(mean_lik[0], place_fields[0] / place_fields[0].sum())

    def test_matches_diagnostics_per_spike_likelihood(self) -> None:
        # Parity across the two entry points that both claim to plot/consume
        # the same per-spike likelihood: the plotting helper (place fields,
        # (n_cells, n_bins)) and the diagnostics (rates, (n_bins, n_cells)).
        from statespacecheck_paper.diagnostics import (
            compute_spike_event_diagnostics_from_rates,
        )
        from statespacecheck_paper.figure04_diagnostics import (
            mean_per_spike_likelihood_by_time,
        )

        place_fields = np.array([[2.0, 0.5, 1.0], [0.1, 0.4, 0.3]])  # (n_cells, n_bins)
        spike_counts = np.array([[0, 1]], dtype=np.int64)  # one spike from cell 1 at t=0
        mean_lik, _ = mean_per_spike_likelihood_by_time(spike_counts, place_fields)

        predictive = np.full((1, 3), 1.0 / 3.0)
        diag = compute_spike_event_diagnostics_from_rates(
            predictive,
            place_fields.T,
            np.array([0], dtype=np.intp),
            np.array([1], dtype=np.intp),
            include_dense_matrices=True,
        )
        assert diag.per_spike_likelihood is not None
        np.testing.assert_allclose(mean_lik[0], diag.per_spike_likelihood[0])


# ---------------------------------------------------------------------------
# position-marginal model diagnostics
# ---------------------------------------------------------------------------


def _two_state_model(
    place_fields: np.ndarray,
    position_bins: np.ndarray,
    interior_mask: np.ndarray,
) -> MagicMock:
    model = MagicMock()
    model.observation_models = [
        MagicMock(environment_name="", encoding_group=0),
        MagicMock(environment_name="", encoding_group=0),
    ]
    model.encoding_model_ = {("", 0): {"place_fields": place_fields}}
    environment = MagicMock()
    environment.place_bin_centers_ = position_bins[:, np.newaxis]
    model.environments = [environment]
    model.is_track_interior_state_bins_ = np.tile(interior_mask, 2)
    return model


class TestComputeModelDiagnostics:
    def test_marginalizes_state_and_uses_one_shared_place_field_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        position_bins = np.array([0.0, 1.0, 2.0])
        interior_mask = np.array([True, False, True])
        place_fields = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model = _two_state_model(place_fields, position_bins, interior_mask)

        posterior_per_state = np.array(
            [
                [[0.10, np.nan, 0.20], [0.30, np.nan, 0.40]],
                [[0.25, np.nan, 0.15], [0.35, np.nan, 0.25]],
            ]
        )
        state_bins = pd.MultiIndex.from_product(
            [["Continuous", "Fragmented"], position_bins],
            names=["state", "position"],
        )
        results = _xarray_results(
            posterior_per_state.reshape(2, -1),
            "predictive_posterior",
            state_bins=state_bins,
        )

        captured: dict[str, Any] = {}
        sentinel = MagicMock()

        def _capture(
            predictive_posterior: np.ndarray,
            spike_counts: np.ndarray,
            diagnostic_place_fields: np.ndarray,
            **kwargs: Any,
        ) -> MagicMock:
            captured["predictive"] = predictive_posterior
            captured["place_fields"] = diagnostic_place_fields
            captured["kwargs"] = kwargs
            return sentinel

        monkeypatch.setattr(figure04_diagnostics, "compute_spike_event_diagnostics", _capture)
        spike_counts = np.zeros((2, 2), dtype=np.int64)
        time = np.array([0.0, 0.002])
        result = compute_model_diagnostics(model, results, spike_counts, time)

        assert result is sentinel
        np.testing.assert_allclose(
            captured["predictive"],
            posterior_per_state[:, :, interior_mask].sum(axis=1),
        )
        np.testing.assert_allclose(captured["place_fields"], place_fields[:, interior_mask])
        assert captured["kwargs"]["time"] is time


# ---------------------------------------------------------------------------
# compute_running_average
# ---------------------------------------------------------------------------


class TestComputeRunningAverage:
    def test_output_shape_matches_evaluation_time(self, rng: np.random.Generator) -> None:
        n_time = 100
        time = np.linspace(0, 1, n_time)
        event_times = rng.uniform(0, 1, 200)
        event_values = rng.random(200)
        running_avg, time_out = compute_running_average(
            event_times, event_values, time, window_size=0.1
        )
        assert running_avg.shape == (n_time,)
        np.testing.assert_array_equal(time_out, time)

    def test_nan_event_value_raises(self) -> None:
        n_time = 100
        time = np.linspace(0, 1, n_time)
        with pytest.raises(ValueError, match="Every spike event"):
            compute_running_average(
                np.array([0.2, 0.4]), np.array([1.0, np.nan]), time, window_size=0.1
            )

    def test_no_events_yields_all_nan_output(self) -> None:
        n_time = 100
        time = np.linspace(0, 1, n_time)
        running_avg, _ = compute_running_average(np.array([]), np.array([]), time, window_size=0.1)
        assert np.all(np.isnan(running_avg))

    def test_larger_window_smooths_more(self, rng: np.random.Generator) -> None:
        n_time = 1000
        time = np.linspace(0, 1, n_time)
        event_times = time.copy()
        event_values = rng.random(n_time)
        small, _ = compute_running_average(event_times, event_values, time, window_size=0.01)
        large, _ = compute_running_average(event_times, event_values, time, window_size=0.1)
        assert np.nanvar(large) < np.nanvar(small)

    def test_event_inputs_count_duplicates_at_same_time(self) -> None:
        """Two events at the same time both contribute to the running mean."""
        time = np.array([0.0, 1.0, 2.0])
        running_avg, _ = compute_running_average(
            np.array([1.0, 1.0]),
            np.array([1.0, 3.0]),
            time,
            window_size=0.1,
        )
        assert np.isnan(running_avg[0])
        assert running_avg[1] == 2.0
        assert np.isnan(running_avg[2])


# ---------------------------------------------------------------------------
# compute_flag_confusion
# ---------------------------------------------------------------------------


def _diag_from_events(
    *,
    hpd: np.ndarray | None = None,
    kl: np.ndarray | None = None,
    sp: np.ndarray | None = None,
) -> SpikeEventDiagnostics:
    """Minimal ``SpikeEventDiagnostics`` carrying only the per-spike event arrays.

    ``compute_flag_confusion`` reads a single ``event_*`` array; the rest of the
    dataclass is required by the constructor but unused here.
    """
    present = [a for a in (hpd, kl, sp) if a is not None]
    n = present[0].shape[0]
    zeros = np.zeros(n)
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n, dtype=np.intp),
        event_cell_ind=np.zeros(n, dtype=np.intp),
        event_hpd_overlap=hpd if hpd is not None else zeros,
        event_kl_divergence=kl if kl is not None else zeros,
        event_predictive_pvalue=sp if sp is not None else zeros,
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        per_spike_likelihood=None,
    )


class TestComputeFlagConfusion:
    def test_below_direction_counts_and_rescue_rate(self) -> None:
        a = _diag_from_events(hpd=np.array([0.01, 0.02, 0.10, 0.20, 0.03]))
        b = _diag_from_events(hpd=np.array([0.01, 0.20, 0.02, 0.20, 0.20]))
        conf = compute_flag_confusion(a, b, "hpd_overlap", 0.05, worse_when="below")
        assert (conf.n, conf.both, conf.a_only, conf.b_only, conf.neither) == (5, 1, 2, 1, 1)
        assert conf.both + conf.a_only + conf.b_only + conf.neither == conf.n
        assert conf.rescue_rate == pytest.approx(2 / 3)

    def test_above_direction(self) -> None:
        a = _diag_from_events(kl=np.array([5.0, 6.0, 1.0, 2.0]))
        b = _diag_from_events(kl=np.array([5.0, 1.0, 7.0, 1.0]))
        conf = compute_flag_confusion(a, b, "kl_divergence", 4.0, worse_when="above")
        assert (conf.both, conf.a_only, conf.b_only, conf.neither) == (1, 1, 1, 1)
        assert conf.rescue_rate == pytest.approx(0.5)

    def test_threshold_values_are_inclusive(self) -> None:
        hpd_a = _diag_from_events(hpd=np.array([0.05, 0.10]))
        hpd_b = _diag_from_events(hpd=np.array([0.10, 0.05]))
        hpd_conf = compute_flag_confusion(hpd_a, hpd_b, "hpd_overlap", 0.05, worse_when="below")
        assert (hpd_conf.a_only, hpd_conf.b_only) == (1, 1)

        kl_a = _diag_from_events(kl=np.array([4.0, 3.0]))
        kl_b = _diag_from_events(kl=np.array([3.0, 4.0]))
        kl_conf = compute_flag_confusion(kl_a, kl_b, "kl_divergence", 4.0, worse_when="above")
        assert (kl_conf.a_only, kl_conf.b_only) == (1, 1)

    def test_nan_event_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="required per-event value"):
            _diag_from_events(hpd=np.array([0.01, np.nan, 0.02]))

    def test_rescue_rate_nan_when_a_flags_nothing(self) -> None:
        a = _diag_from_events(hpd=np.array([0.5, 0.6]))  # none at or below 0.05
        b = _diag_from_events(hpd=np.array([0.01, 0.6]))
        conf = compute_flag_confusion(a, b, "hpd_overlap", 0.05, worse_when="below")
        assert conf.a_only == 0 and conf.both == 0
        assert np.isnan(conf.rescue_rate)

    def test_rejects_bad_direction(self) -> None:
        a = _diag_from_events(hpd=np.array([0.1]))
        bad: Any = "sideways"
        with pytest.raises(ValueError, match="worse_when"):
            compute_flag_confusion(a, a, "hpd_overlap", 0.05, worse_when=bad)

    def test_rejects_length_mismatch(self) -> None:
        a = _diag_from_events(hpd=np.array([0.1, 0.2]))
        b = _diag_from_events(hpd=np.array([0.1]))
        with pytest.raises(ValueError, match="same set of spike events"):
            compute_flag_confusion(a, b, "hpd_overlap", 0.05, worse_when="below")
